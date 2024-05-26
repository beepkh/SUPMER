# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Defines different way of processing the data."""
import sys 
import torch
from typing import List, Tuple

class MLMProcessor(torch.nn.Module):
    """Process the data for a model which is pretrained with the masked
    language modeling loss."""
    def __init__(self, tokenizer, tokenized_verbalizers, max_seq_length, max_dec_length,
        processor, mask_length=None, train_classifier=False):
        super(MLMProcessor, self).__init__()
        self.tokenizer = tokenizer 
        self.tokenized_verbalizers = tokenized_verbalizers
        self.max_seq_length = max_seq_length
        self.max_dec_length = max_dec_length
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.processor = processor 
        # In case of using soft_pet, we path the mask_length so we use the one passed.
        self.mask_length = mask_length
        self.train_classifier = train_classifier 

    # copied from pet 
    def seq_length(self, parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    # copied from pet 
    def remove_last(self, parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    # copied from pet 
    def truncate(self, parts_a: List[Tuple[str, bool]], parts_b: List[Tuple[str, bool]]):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self.seq_length(parts_a) + self.seq_length(parts_b)
        total_len += self.tokenizer.num_special_tokens_to_add(bool(parts_b))
        num_tokens_to_remove = total_len - self.max_seq_length

        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self.seq_length(parts_a, only_shortenable=True) > self.seq_length(parts_b, only_shortenable=True):
                self.remove_last(parts_a)
            else:
                self.remove_last(parts_b)

    def tokenize(self, text_list):
        """Gets a list of Text enteries and tokenize them, returns the output as a list of tuples with the
        tokenized text and the shortenable entry."""
        return [(self.tokenizer.encode(text.text, add_special_tokens=False), text.shortenable) for text in text_list]

    def get_tokens(self, tuple_list):
        if not tuple_list:
            return None 
        return [token_id for part, _ in tuple_list for token_id in part]

    def preprocess_inputs(self, example):
        part_0, part_1 = self.processor.get_encoder_parts(
            example=example
        )
        target = self.processor.get_target(example=example)
        part_0_tuples = self.tokenize(part_0)
        part_1_tuples = self.tokenize(part_1)
        self.truncate(part_0_tuples, part_1_tuples)
        token_ids_0 = self.get_tokens(part_0_tuples) 
        token_ids_1 = self.get_tokens(part_1_tuples) 
        if token_ids_1 is not None:
            input_ids = token_ids_0 + token_ids_1
        else:
            input_ids = token_ids_0
        attention_mask = [1] * len(input_ids)
        n_mask = self.max_seq_length - len(input_ids)
        
        # Pads the tokens and attention mask.
        input_ids = input_ids + [self.pad_token_id] * n_mask
        attention_mask = attention_mask + [0] * n_mask
        
        # decoder
        de_part, loss_ids = self.processor.get_decoder_parts(int(target))
        de_tuples = self.tokenize(de_part)
        de_token_ids = [0] + self.get_tokens(de_tuples)
        decoder_input_ids = de_token_ids[:-1]
        labels = de_token_ids[1:]
        decoder_attention_mask = [1] * len(decoder_input_ids)
        de_n_mask = self.max_dec_length - len(decoder_input_ids)
        
        # Pads the tokens and attention mask for decoder
        decoder_input_ids = decoder_input_ids + [self.pad_token_id] * de_n_mask
        labels = labels + [self.pad_token_id] * de_n_mask
        decoder_attention_mask = decoder_attention_mask + [0] * de_n_mask
        loss_ids = loss_ids + [0] * de_n_mask
        
        extra_fields = self.processor.get_extra_fields(example=example)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels,
            'loss_ids': loss_ids,
            'extra_fields': extra_fields
        }
        

    def forward(self, example):
        return self.preprocess_inputs(example)

        