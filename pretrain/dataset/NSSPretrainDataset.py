# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""GPT2 style dataset."""

import numpy as np
import torch
import random

from .IndexDataset import MMapIndexedDataset, _build_index_mappings


class NSSPretrainDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, name, domain, document_ids_in_splits, context_indexed_dataset: MMapIndexedDataset,
                 target_indexed_dataset: MMapIndexedDataset, max_seq_length, max_dec_length):

        self.name = name
        self.domain = domain
        self.context_indexed_dataset = context_indexed_dataset
        self.target_indexed_dataset = target_indexed_dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_dec_length = max_dec_length

        # Checks
        assert np.min(document_ids_in_splits) >= 0
        assert np.max(
            document_ids_in_splits) < context_indexed_dataset.sizes.shape[0]
        self.doc_idx = document_ids_in_splits
        random.shuffle(self.doc_idx)

    def __len__(self):
        return self.doc_idx.shape[0]

    def __getitem__(self, idx):
        # Get the shuffled index.
        # NOTE: We do not get shuffle idx because the documents are already shuffled

        idx = self.doc_idx[idx]

        contexts = self.context_indexed_dataset.get(idx)
        targets = self.target_indexed_dataset.get(idx)

        contexts = [int(x) for x in contexts]

        attention_mask = [1]*len(contexts)

        targets = [int(x) for x in targets]

        labels = targets[1:]
        targets = targets[:-1]
        decoder_atten_mask = [1] * len(labels)
        loss_ids = [1, 1, 1]

        if len(contexts) < self.max_seq_length:
            contexts = contexts + [self.tokenizer.pad_token_id] * (self.max_seq_length - len(contexts))
            attention_mask = attention_mask + [0] * (self.max_seq_length - len(attention_mask))
        if len(labels) < self.max_dec_length:
            targets = targets + [self.tokenizer.pad_token_id] * (self.max_dec_length - len(targets))
            labels = labels + [self.tokenizer.pad_token_id] * (self.max_dec_length - len(labels))
            decoder_atten_mask = decoder_atten_mask + [0] * (self.max_dec_length - len(decoder_atten_mask))
            loss_ids = loss_ids + [0] * (self.max_dec_length - len(loss_ids))

        attention_mask = attention_mask[:self.max_seq_length]
        contexts = contexts[:self.max_seq_length]
        labels = labels[:self.max_dec_length]
        targets = targets[:self.max_dec_length]
        decoder_atten_mask = decoder_atten_mask[:self.max_dec_length]

        return {
            "input_ids": contexts,
            "attention_mask": attention_mask,
            "decoder_input_ids": targets,
            "labels": labels,
            "decoder_attention_mask": decoder_atten_mask,
            "loss_ids": loss_ids
        }