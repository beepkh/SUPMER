import torch
import random
from torch.utils.data import Dataset
from .task import MetaTask, FinalTensorDataset

class NextSentenceTask(MetaTask):
    def __init__(self, datasets, num_tasks, num_domains, 
                 k_support, k_query, training, classify_num=5, 
                 max_enc_seq_len=1024, max_dec_seq_len=10, 
                 tokenizer=None, neg_sent = 4):
        super(NextSentenceTask, self).__init__(datasets, num_tasks, num_domains, k_support, k_query, training)
        self.max_enc_seq_len = max_enc_seq_len 
        self.max_dec_seq_len = max_dec_seq_len
        self.create_batch()
    
    def create_batch(self):
        assert(self.num_domains >= self.num_tasks)
        self.supports = list()
        self.queries = list()
        self.supports_aug = list()
        self.queries_aug = list()
        self.domain_name = list()
        self.domain_name_aug = list()
        
        batch_domain_ori = random.sample(range(self.num_domains), self.num_tasks)
        batch_domain_aug = [random.randint(0, self.num_domains-1) for b in batch_domain_ori]
        self.classes = [[[batch_domain_ori[i]], [batch_domain_aug[i]]] for i in range(len(batch_domain_ori))]

        for ori_or_aug, batch_domain in enumerate([batch_domain_ori, batch_domain_aug]):
            for b in batch_domain:
                all_idx = list(range(self.datasets[b].__len__()))
                selected_idx = random.sample(all_idx, self.k_support+self.k_query)
                if self.training:
                    random.shuffle(selected_idx)
                train_idx = selected_idx[:self.k_support]
                test_idx = selected_idx[self.k_support:]

                if ori_or_aug==0:
                    self.domain_name.append(b)
                    self.supports.append(train_idx)
                    self.queries.append(test_idx)
                else:
                    self.domain_name_aug.append(b)
                    self.supports_aug.append(train_idx)
                    self.queries_aug.append(test_idx)

    def create_value(self, examples, domain):
        all_input_ids              = torch.empty((len(examples), self.max_enc_seq_len), dtype = torch.long)
        all_attention_mask         = torch.empty(len(examples), self.max_enc_seq_len, dtype = torch.long)
        all_decoder_input_ids      = torch.empty(len(examples), self.max_dec_seq_len, dtype = torch.long)
        all_labels                 = torch.empty(len(examples), self.max_dec_seq_len, dtype = torch.long)
        all_decoder_attention_mask = torch.empty(len(examples), self.max_dec_seq_len, dtype = torch.long)
        all_loss_ids               = torch.empty(len(examples), self.max_dec_seq_len, dtype = torch.long)
        for id_, example in enumerate(examples):
            exam = self.datasets[domain].__getitem__(example)

            all_input_ids[id_] = torch.Tensor(exam['input_ids']).to(torch.long)
            all_attention_mask[id_] = torch.Tensor(exam['attention_mask']).to(torch.long)
            all_decoder_input_ids[id_] = torch.Tensor(exam['decoder_input_ids']).to(torch.long)
            all_labels[id_] = torch.Tensor(exam['labels']).to(torch.long)
            all_decoder_attention_mask[id_] = torch.Tensor(exam['decoder_attention_mask']).to(torch.long)
            all_loss_ids[id_] = torch.Tensor(exam['loss_ids']).to(torch.long)

        tensor_set = FinalTensorDataset(all_input_ids, all_attention_mask, all_decoder_input_ids, all_labels, all_decoder_attention_mask, all_loss_ids)
        return tensor_set
        

    def __getitem__(self, index):
        support_set = self.create_value(self.supports[index], self.domain_name[index])
        query_set = self.create_value(self.queries[index], self.domain_name[index])
        query_aug_set = self.create_value(self.queries_aug[index], self.domain_name_aug[index])
        return support_set, query_set, query_aug_set, torch.Tensor(self.classes[index]).to(torch.long)

