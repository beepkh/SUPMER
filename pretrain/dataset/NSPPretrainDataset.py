import numpy as np
import torch
import random

from .IndexDataset import MMapIndexedDataset, _build_index_mappings


class NSPPretrainDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, name, domain, document_ids_in_splits, context_indexed_dataset: MMapIndexedDataset,
                target_indexed_dataset: MMapIndexedDataset, max_seq_length, max_dec_length):

        self.name = name
        self.domain = domain
        self.context_indexed_dataset = context_indexed_dataset
        self.target_indexed_dataset = target_indexed_dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_dec_length = max_dec_length
        
        self.token_id = tokenizer.convert_tokens_to_ids(["<extra_id_0>", "<extra_id_1>", ".", "?"])
        # Checks
        assert np.min(document_ids_in_splits) >= 0
        assert np.max(document_ids_in_splits) < context_indexed_dataset.sizes.shape[0]
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
        self.unify = False
        if self.unify == False:
            if len(contexts) > self.max_seq_length:
                sentinel_pos = contexts.index(self.token_id[0])
                p1_tokens = int(sentinel_pos / len(contexts) * (self.max_seq_length - 1)) - 1
                p2_tokens = int((1 - sentinel_pos / len(contexts)) * (self.max_seq_length - 1)) - 1
                contexts = contexts[:p1_tokens] + [self.token_id[0]] + contexts[sentinel_pos+1:sentinel_pos+1+p2_tokens]
        else:
            if len(contexts) > self.max_seq_length:
                sentinel_pos_1 = contexts.index(self.token_id[0])
                sentinel_pos_2 = contexts.index(self.token_id[1])
                other_tokens_len = len(contexts[sentinel_pos_2:])
                p1_tokens = int(sentinel_pos_1 / (len(contexts)-other_tokens_len) * (self.max_seq_length - other_tokens_len)) - 1
                p2_tokens = int((1 - sentinel_pos_1 / (len(contexts)-other_tokens_len)) * (self.max_seq_length - other_tokens_len)) - 1
                contexts = (contexts[:p1_tokens] + [self.token_id[0]] + 
                            contexts[sentinel_pos_1+1:sentinel_pos_1+1+p2_tokens] + contexts[sentinel_pos_2+1:])
        after_len = len(contexts)

        attention_mask = [1] * after_len
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

        return {
            "input_ids": contexts,
            "attention_mask": attention_mask,
            "decoder_input_ids": targets,
            "labels": labels,
            "decoder_attention_mask": decoder_atten_mask,
            'loss_ids': loss_ids
        }