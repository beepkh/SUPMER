import torch
import random
from torch.utils.data import Dataset
import numpy as np
from .task import MetaTask, FinalTensorDataset

class SentClusterClassifiTask(MetaTask):
    def __init__(self, datasets, num_tasks, num_domains, k_support, k_query, training, classify_num=5, max_enc_seq_len=1024, max_dec_seq_len=10, tokenizer=None, neg_sent = 4):
        super(SentClusterClassifiTask, self).__init__(datasets, num_tasks, num_domains, k_support, k_query, training)
        self.classify_num = classify_num
        self.max_enc_seq_len = max_enc_seq_len
        self.max_dec_seq_len = max_dec_seq_len
        self.pad_token_id = tokenizer.pad_token_id
        self.sentid = tokenizer.encode("The correct one is ", add_special_tokens=False)
        self.token_id = tokenizer.convert_tokens_to_ids(["<extra_id_0>", "<extra_id_1>", ".", "?"])
        self.label_id = tokenizer.convert_tokens_to_ids(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                                                              "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                                                              "U", "V", "W", "X", "Y", "Z"])
        self.create_batch()
    
    def create_batch(self):
        assert self.num_domains >= self.classify_num
        self.supports = list()
        self.queries = list()
        self.classes = list()
        self.supports_aug = list()
        self.queries_aug = list()
        self.classes_aug = list()
        for b in range(self.num_tasks):
            class_name_ori = random.sample(range(self.num_domains), self.classify_num)
            class_name_aug = random.sample(range(self.num_domains), self.classify_num)
            for ori_or_aug, class_name in enumerate([class_name_ori, class_name_aug]):
                cur_supports, cur_queries = [], []
                for c in class_name:
                    all_idx = list(range(self.datasets[c].__len__()))
                    selected_idx = random.sample(all_idx, self.k_support+self.k_query)
                    selected_idx = [(c, i) for i in selected_idx]
                    if self.training:
                        random.shuffle(selected_idx)
                    train_idx = selected_idx[:self.k_support]
                    test_idx = selected_idx[self.k_support:]

                    cur_supports.extend(train_idx)
                    cur_queries.extend(test_idx)

                if self.training:
                    random.shuffle(cur_supports)
                    random.shuffle(cur_queries)
                if ori_or_aug==0:
                    self.classes.append(class_name)
                    self.supports.append(cur_supports)
                    self.queries.append(cur_queries)
                else:
                    self.classes_aug.append(class_name)
                    self.supports_aug.append(cur_supports)
                    self.queries_aug.append(cur_queries)
    
    def create_value(self, examples, class_name):
        all_input_ids              = torch.empty(len(examples), self.max_enc_seq_len, dtype = torch.long)
        all_attention_mask         = torch.empty(len(examples), self.max_enc_seq_len, dtype = torch.long)
        all_decoder_input_ids      = torch.empty(len(examples), self.max_dec_seq_len, dtype = torch.long)
        all_labels                 = torch.empty(len(examples), self.max_dec_seq_len, dtype = torch.long)
        all_decoder_attention_mask = torch.empty(len(examples), self.max_dec_seq_len, dtype = torch.long)
        all_loss_ids               = torch.empty(len(examples), self.max_dec_seq_len, dtype = torch.long) 
        
        choice_id = [self.token_id[3]]
        label_map = {}
        for i, c in enumerate(class_name):
            label_map[c] = i
            choice_id += [self.label_id[i], self.token_id[2]] + [-c-1] + [self.token_id[2]]
        choice_id += [self.token_id[3]] + self.sentid + [self.token_id[0]]

        for id_, example in enumerate(examples):
            domain, ex_id = example[0], example[1]
            exam = self.datasets[domain].__getitem__(ex_id)['input_ids']
            if len(exam) > self.max_enc_seq_len - len(choice_id):
                input_ids = exam[:(self.max_enc_seq_len - len(choice_id) -1)] + choice_id
            else:
                input_ids = exam + choice_id
            attention_mask = [1] * len(input_ids)
            target = [0, self.token_id[0], self.label_id[label_map[domain]]] 
            decoder_input_ids = target[:-1]
            labels = target[1:]
            decoder_attention_mask = [1] * len(labels)
            loss_ids = [1, 1]

            if len(input_ids) < self.max_enc_seq_len:
                input_ids.extend([0] * (self.max_enc_seq_len - len(input_ids)))
                attention_mask.extend([0] * (self.max_enc_seq_len - len(attention_mask)))
            
            if len(labels) < self.max_dec_seq_len:
                decoder_input_ids.extend([0] * (self.max_dec_seq_len - len(decoder_input_ids)))
                labels.extend([0] * (self.max_dec_seq_len - len(labels)))
                decoder_attention_mask.extend([0] * (self.max_dec_seq_len - len(decoder_attention_mask)))
                loss_ids.extend([0] * (self.max_dec_seq_len - len(loss_ids)))
            
            all_input_ids[id_] = torch.Tensor(input_ids).to(torch.long)
            all_attention_mask[id_] = torch.Tensor(attention_mask).to(torch.long)
            all_decoder_input_ids[id_] = torch.Tensor(decoder_input_ids).to(torch.long)
            all_labels[id_] = torch.Tensor(labels).to(torch.long)
            all_decoder_attention_mask[id_] = torch.Tensor(decoder_attention_mask).to(torch.long)
            all_loss_ids[id_] = torch.Tensor(loss_ids).to(torch.long)
        
        tensor_set = FinalTensorDataset(all_input_ids, all_attention_mask, all_decoder_input_ids, all_labels, all_decoder_attention_mask, all_loss_ids)  
        return tensor_set


    def __getitem__(self, index):
        support_set = self.create_value(self.supports[index], self.classes[index])
        query_set = self.create_value(self.queries[index], self.classes[index])
        query_aug_set = self.create_value(self.queries_aug[index], self.classes_aug[index])
        all_classes = [self.classes[index], self.classes_aug[index]]
        return support_set, query_set, query_aug_set, torch.Tensor(all_classes).to(torch.long)
