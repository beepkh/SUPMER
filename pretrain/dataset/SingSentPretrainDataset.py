import numpy as np
import torch
import random

from .IndexDataset import MMapIndexedDataset, _build_index_mappings


class SingSentPretrainDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, name, domain, document_ids_in_splits, context_indexed_dataset: MMapIndexedDataset, target_indexed_dataset,
                 max_seq_length, max_dec_length):

        self.name = name
        self.domain = domain
        self.context_indexed_dataset = context_indexed_dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_dec_length = max_dec_length
        
        # Checks
        assert np.min(document_ids_in_splits) >= 0
        assert np.max(document_ids_in_splits) < context_indexed_dataset.sizes.shape[0]

        self.doc_idx = document_ids_in_splits
        random.shuffle(self.doc_idx)
        

    def __len__(self):
        return self.doc_idx.shape[0] 

    def __getitem__(self, idx):
        idx = self.doc_idx[idx]

        contexts = self.context_indexed_dataset.get(idx)

        contexts = [int(x) for x in contexts]

        contexts = contexts[:self.max_seq_length]

        return {
            "input_ids": contexts
        }