# coding=utf-8
# Copyright 2019 project LXRT.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random

import torch
import torch.nn as nn

from lxrt.tokenization import BertTokenizer
from lxrt.modeling import LXRTFeatureExtraction, VISUAL_CONFIG
from param import args


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, masked_labels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.masked_labels = masked_labels

def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        # Do not mask CLS or SEP tokens
        if tokens[i] is "[CLS]" or tokens[i] is "[SEP]":
          output_label.append(-1)
          continue
          
        prob = random.random()
        # mask token with probability
        ratio = args.word_mask_rate
        if prob < ratio:
            prob /= ratio

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def convert_sents_to_features(sents, max_seq_length, tokenizer, semantic_queries=None):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    # make sure we have semantic queries for all sentences
    if semantic_queries is not None:
      assert(len(sents) == len(semantic_queries))
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())
        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] 
        segment_ids = [0] * len(tokens)

        tokens_b = tokenizer.tokenize(semantic_queries[i].strip())
        if len(tokens_b) > 0:
          tokens += tokens_b + ["[SEP]"]
          segment_ids_b = [1] * (len(tokens_b) + 1)
          segment_ids += segment_ids_b
  
        if len(tokens) > max_seq_length:
          #print(f"Tokens = {len(tokens)} are longer than max_seq_length {max_seq_length}")
          tokens = tokens[:max_seq_length]
          segment_ids = segment_ids[:max_seq_length]
        
        masked_labels = [-1] * len(tokens)
        if args.task_mlm_qfpm:
          tokens, masked_labels = random_word(tokens, tokenizer)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        neg_one_padding = [-1] * len(padding)

        #print(f'{tokenizer.convert_tokens_to_ids(["[CLS]"])}  &&&&^&^&^&^&^&^&^ ')
        input_ids += padding
        #input_ids[-1] = 101 # ["CLS"] token id
        input_mask += padding
        #input_mask[-1] = 1
        segment_ids += padding
        masked_labels += neg_one_padding
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(masked_labels) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              masked_labels=masked_labels))
    return features


def set_visual_config(args):
    VISUAL_CONFIG.l_layers = args.llayers
    VISUAL_CONFIG.x_layers = args.xlayers
    VISUAL_CONFIG.r_layers = args.rlayers


class LXRTEncoder(nn.Module):
    def __init__(self, args, max_seq_length, mode='x'):
        super().__init__()
        self.max_seq_length = max_seq_length
        set_visual_config(args)

        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

       # Build LXRT Model
        self.model = LXRTFeatureExtraction.from_pretrained(
            "bert-base-uncased",
            mode=mode
        )

        if args.from_scratch:
            print("initializing all the weights")
            self.model.apply(self.model.init_bert_weights)
          

    def multi_gpu(self):
        self.model = nn.DataParallel(self.model)

    @property
    def dim(self):
        return 768

    def forward(self, sents, vis_feats, visual_attention_mask=None, semantic_queries=None):
        train_features = convert_sents_to_features(
            sents, self.max_seq_length, self.tokenizer, semantic_queries=semantic_queries)

        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()
        masked_labels = torch.tensor([f.masked_labels for f in train_features], dtype=torch.long).cuda()

        output = self.model(input_ids, segment_ids, input_mask,
                            visual_feats=vis_feats,
                            visual_attention_mask=visual_attention_mask)
        
        return output, masked_labels

    def save(self, path):
        torch.save(self.model.state_dict(),
                   os.path.join("%s_LXRT.pth" % path))

    def load(self, path):
        # Load state_dict from snapshot file
        print("Load LXMERT pre-trained model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        self.model.load_state_dict(state_dict, strict=False)




