# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.gqa_model import GQAModel
from tasks.gqa_data import GQADataset, GQATorchDataset, GQAEvaluator

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False, skip_semantics=False) -> DataTuple:
    dset = GQADataset(splits)
    tset = GQATorchDataset(dset, skip_semantics)
    evaluator = GQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class GQA:
    def __init__(self):
        self.train_tuple = get_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            valid_bsize = 2048 if args.multiGPU else 512
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False, skip_semantics=True
            )
        else:
            self.valid_tuple = None

        self.model = GQAModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Losses and optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
            if args.task_nsp_qfpm:
              self.task_optim = BertAdam(list(self.model.parameters()),
                                    lr=args.lr,
                                    warmup=0.1,
                                    t_total=t_total)
        else:
            self.optim = args.optimizer(list(self.model.parameters()), args.lr)

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        # Pre train the NSP head
        if args.task_nsp_qfpm or args.task_mlm_qfpm:
          print(f"Pretraining for {args.epochs} epochs")
          for epoch in range(args.epochs):          
            epoch_pretrain_loss = 0
            epoch_nsp_loss = 0
            epoch_mlm_loss = 0
            for i, (ques_id, feats, boxes, sent, sem_query, sem_matched, target) in iter_wrapper(enumerate(loader)):
              self.model.train()
              self.task_optim.zero_grad()

              feats, boxes, sem_matched, target = feats.cuda(), boxes.cuda(), sem_matched.cuda(), target.cuda()
              _, logit_nsp_qfpm, logit_mlm_qfpm, masked_lang_labels = self.model(feats, boxes, sent, sem_query)
              
              loss = 0
              if args.task_nsp_qfpm:
                nsp_qfpm_loss =  self.mce_loss(logit_nsp_qfpm, sem_matched) 
                loss += nsp_qfpm_loss
                epoch_nsp_loss += nsp_qfpm_loss.detach()
              
              if args.task_mlm_qfpm:
                # masked_lang_labels: [batch, max_length], logit_mlm_qfpm: [batch, max_length, vocab_size]
                vocab_size = logit_mlm_qfpm.shape[2]
                masked_lm_loss = self.mce_loss(
                  logit_mlm_qfpm.view(-1, vocab_size),
                  masked_lang_labels.view(-1)
                )
                loss += masked_lm_loss                
                epoch_mlm_loss += masked_lm_loss.detach()
              
              loss.backward()
              nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
              self.task_optim.step()
              epoch_pretrain_loss += loss.detach()

            print(f"Total loss for epoch = {epoch_pretrain_loss} ... NSP = {epoch_nsp_loss} ... MLM = {epoch_mlm_loss}")      
            
        best_valid = 0.
        args.task_nsp_qfpm = False
        args.task_mlm_qfpm = False
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, sem_query, sem_matched, target) in iter_wrapper(enumerate(loader)):
                sem_query = [''] * len(sem_query)
                self.model.train()
                self.optim.zero_grad()

                feats, boxes, sem_matched, target = feats.cuda(), boxes.cuda(), sem_matched.cuda(), target.cuda()
                logit_qa = self.model(feats, boxes, sent, sem_query)
                assert logit_qa.dim() == target.dim() == 2
                if args.mce_loss:
                    max_value, target = target.max(1)
                    loss = self.mce_loss(logit_qa, target) * logit_qa.size(1)
                else:
                    loss = self.bce_loss(logit_qa, target)
                    loss = loss * logit_qa.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit_qa.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent, sem_query, sem_matched = datum_tuple[:6]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit_qa = self.model(feats, boxes, sent, sem_query)
                score, label = logit_qa.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans = self.predict(eval_tuple, dump)
        return evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, sem_query, sem_matched, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        for key in list(state_dict.keys()):
            if '.module' in key:
                state_dict[key.replace('.module', '')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    # Build Class
    gqa = GQA()

    # Load Model
    if args.load is not None:
        gqa.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'submit' in args.test:
            gqa.predict(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'submit_predict.json')
            )
        if 'testdev' in args.test:
            result = gqa.evaluate(
                get_tuple('testdev', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'testdev_predict.json')
            )
            print(result)
    else:
        # print("Train Oracle: %0.2f" % (gqa.oracle_score(gqa.train_tuple) * 100))
        print('Splits in Train data:', gqa.train_tuple.dataset.splits)
        if gqa.valid_tuple is not None:
            print('Splits in Valid data:', gqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (gqa.oracle_score(gqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        gqa.train(gqa.train_tuple, gqa.valid_tuple)


