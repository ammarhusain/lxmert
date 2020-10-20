# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_GQA_LENGTH = 150


class GQAModel(nn.Module):
    def __init__(self, num_answers, task_nsp_qfpm=True):
        super().__init__()
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_GQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        ##AH TODO: Add a masked LM head here
        self.task_nsp_qfpm = task_nsp_qfpm

        if self.task_nsp_qfpm is True:
          self.fc_nsp_qfpm = nn.Sequential(
              nn.Linear(hid_dim, hid_dim * 2),
              GeLU(),
              BertLayerNorm(hid_dim * 2, eps=1e-12),
              nn.Linear(hid_dim * 2, 2)
          )
          self.fc_nsp_qfpm.apply(self.lxrt_encoder.model.init_bert_weights)
        
    def forward(self, vis_feat, vis_pos, sent, sem_queries):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param semantic_queries: (b,) Type -- list of string semantic queries corresponding to the sent
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(sent, (vis_feat, vis_pos), semantic_queries=sem_queries)
        logit = self.logit_fc(x)

        if self.task_nsp_qfpm is True:
          logit_nsp = self.fc_nsp_qfpm(x)
          return logit, logit_nsp
        
        return logit


