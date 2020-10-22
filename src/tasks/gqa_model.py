# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU, BertPreTrainingHeads, BertConfig

# Max length including <bos> and <eos>
MAX_GQA_LENGTH = 100


class GQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_GQA_LENGTH,
            mode='xl'
        )
        hid_dim = self.lxrt_encoder.dim
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

        if args.task_nsp_qfpm or args.task_mlm_qfpm:
          self.qfpm = BertPreTrainingHeads(BertConfig(vocab_size_or_config_json_file = 30522),
                                           self.lxrt_encoder.model.bert.embeddings.word_embeddings.weight)
        
    def forward(self, vis_feat, vis_pos, sent, sem_queries):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param semantic_queries: (b,) Type -- list of string semantic queries corresponding to the sent
        :return: (b, num_answer) The logit of each answers.
        """
        ((lang_feats, visn_feats), pooled_output), masked_labels = self.lxrt_encoder(sent, (vis_feat, vis_pos), semantic_queries=sem_queries)
        logit = self.logit_fc(pooled_output)

        last_raw_token = lang_feats[:,MAX_GQA_LENGTH-1,:]
        first_raw_token = lang_feats[:,0,:]

        if args.task_nsp_qfpm or args.task_mlm_qfpm:
          lang_prediction_scores, nsp_prediction_score = self.qfpm(lang_feats, first_raw_token)
          return logit, nsp_prediction_score, lang_prediction_scores, masked_labels
        
        return logit


