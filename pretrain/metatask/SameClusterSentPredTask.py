import torch
import random
from torch.utils.data import Dataset
from .task import MetaTask, FinalTensorDataset

class SameClusterSentPredTask(MetaTask):
    def __init__(self, datasets, num_tasks, num_domains, 
                 k_support, k_query, training, classify_num=5, 
                 max_enc_seq_len=1024, max_dec_seq_len=10, 
                 tokenizer=None, neg_sent = 4):
        super(SameClusterSentPredTask, self).__init__(datasets, num_tasks, num_domains, k_support, k_query, training)
        self.max_enc_seq_len = max_enc_seq_len
        self.max_dec_seq_len = max_dec_seq_len
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.sentid = tokenizer.encode("The correct one is ", add_special_tokens=False)
        self.token_id = tokenizer.convert_tokens_to_ids(["<extra_id_0>", "<extra_id_1>", ".", "?"])
        self.label_id_1 = tokenizer.convert_tokens_to_ids(["A", "B"])
        self.label_id_2 = tokenizer.convert_tokens_to_ids(['yes', 'no'])
        self.create_batch()

    def create_batch(self):
        assert self.num_domains >= self.num_tasks
        self.supports = list()
        self.queries = list()
        self.supports_aug = list()
        self.queries_aug = list()
        self.domains = list()
        
        batch_domain_ori = random.sample(range(self.num_domains), self.num_tasks)
        batch_domain_aug = [random.randint(0, self.num_domains-1) for b in batch_domain_ori]
        self.classes = [[[batch_domain_ori[i]], [batch_domain_aug[i]]] for i in range(len(batch_domain_ori))]

        for ori_or_aug, batch_domain in enumerate([batch_domain_ori, batch_domain_aug]):
            for b in batch_domain:
                all_idx = list(range(self.datasets[b].__len__()))
                # yes_anchor
                anchor_idx_1 = random.sample(all_idx, self.k_query+self.k_support)
                if self.training:
                    random.shuffle(anchor_idx_1)
                anchor_idx_support_1 = anchor_idx_1[:self.k_support]
                anchor_idx_query_1 = anchor_idx_1[self.k_support:]
                # no_anchor
                anchor_idx_2 = random.sample(all_idx, self.k_query+self.k_support)
                if self.training:
                    random.shuffle(anchor_idx_2)
                anchor_idx_support_2 = anchor_idx_2[:self.k_support]
                anchor_idx_query_2 = anchor_idx_2[self.k_support:]
                # yes_choice
                right_idx = random.sample(all_idx, self.k_query+self.k_support)
                if self.training:
                    random.shuffle(right_idx)
                right_idx_support = right_idx[:self.k_support]
                right_idx_query = right_idx[self.k_support:]
                
                right_exam_support = [[b, b, anchor_idx_support_1[i], right_idx_support[i]] for i in range(self.k_support)]
                right_exam_query = [[b, b, anchor_idx_query_1[i], right_idx_query[i]] for i in range(self.k_query)]
                
                wrong_exam_support = []
                for i in range(self.k_support):
                    d = random.randint(0, self.num_domains-1)
                    while d==b:
                        d = random.randint(0, self.num_domains-1)
                    d_len = self.datasets[d].__len__()
                    d_idx = random.randint(0, d_len-1)
                    wrong_exam_support.append([b, d, anchor_idx_support_2[i], d_idx])

                wrong_exam_query = []
                for i in range(self.k_query):
                    d = random.randint(0, self.num_domains-1)
                    while d==b:
                        d = random.randint(0, self.num_domains-1)
                    d_len = self.datasets[d].__len__()
                    d_idx = random.randint(0, d_len-1)
                    wrong_exam_query.append([b, d, anchor_idx_query_2[i], d_idx])

                exam_support = []
                exam_query = []
                exam_support.extend(right_exam_support)
                exam_support.extend(wrong_exam_support)
                exam_query.extend(right_exam_query)
                exam_query.extend(wrong_exam_query)
                if self.training:
                    random.shuffle(exam_support)
                    random.shuffle(exam_query)
                
                if ori_or_aug==0:
                    self.supports.append(exam_support)
                    self.queries.append(exam_query)
                else:
                    self.supports_aug.append(exam_support)
                    self.queries_aug.append(exam_query)
    
    def create_value(self, examples):
        all_input_ids              = torch.empty(len(examples), self.max_enc_seq_len, dtype = torch.long)
        all_attention_mask         = torch.empty(len(examples), self.max_enc_seq_len, dtype = torch.long)
        all_decoder_input_ids      = torch.empty(len(examples), self.max_dec_seq_len, dtype = torch.long)
        all_labels                 = torch.empty(len(examples), self.max_dec_seq_len, dtype = torch.long)
        all_decoder_attention_mask = torch.empty(len(examples), self.max_dec_seq_len, dtype = torch.long)
        all_loss_ids               = torch.empty(len(examples), self.max_dec_seq_len, dtype = torch.long)
        for id_, example in enumerate(examples):
            domain_1, domain_2, idx_1, idx_2 = example[0], example[1], example[2], example[3]
            if domain_1 == domain_2:
                label = 0
            else:
                label = 1
            exam_1 = self.datasets[domain_1].__getitem__(idx_1)['input_ids']
            exam_2 = self.datasets[domain_2].__getitem__(idx_2)['input_ids']
            unify = False
            if unify:
                choice_id = [self.token_id[3]]
                choice_id += ([self.label_id_1[0], self.token_id[2]] + [self.label_id_2[0]] + [self.token_id[2]])
                choice_id += ([self.label_id_1[1], self.token_id[2]] + [self.label_id_2[1]] + [self.token_id[2]])
                choice_id += [self.token_id[3]] + self.sentid + [self.token_id[0]]
                total_len = self.max_enc_seq_len - len(choice_id) - 1
                if len(exam_1) + len(exam_2) > total_len:
                    p1_tokens = int(len(exam_1) / (len(exam_1)+len(exam_2)) * total_len) - 1
                    p2_tokens = int((1 - len(exam_1) / (len(exam_1)+len(exam_2))) * total_len) -1
                    input_ids = exam_1[:p1_tokens] + [self.token_id[0]] + exam_2[:p2_tokens] + choice_id
                else:
                    input_ids = exam_1 + [self.token_id[0]] + exam_2 + choice_id
                target = [0, self.token_id[0]] + [self.label_id_1[label]] + [self.token_id[1]]
            else:
                if len(exam_1) + len(exam_2) > self.max_enc_seq_len - 1:
                    p1_tokens = int(len(exam_1) / (len(exam_1)+len(exam_2)) * (self.max_enc_seq_len - 1)) - 1
                    p2_tokens = int((1 - len(exam_1) / (len(exam_1)+len(exam_2))) * (self.max_enc_seq_len - 1)) -1
                    input_ids = exam_1[:p1_tokens] + [self.token_id[0]] + exam_2[:p2_tokens]
                else:
                    input_ids = exam_1 + [self.token_id[0]] + exam_2
                target = [0, self.token_id[0]] + [self.label_id_2[label]] + [self.token_id[1]]
            attention_mask = [1] * len(input_ids)

            decoder_input_ids = target[:-1]
            labels = target[1:]
            decoder_attention_mask = [1] * len(labels)
            loss_ids = [1, 1, 1]
            if len(input_ids) < self.max_enc_seq_len:
                input_ids.extend([self.pad_token_id] * (self.max_enc_seq_len - len(input_ids)))
                attention_mask.extend([0] * (self.max_enc_seq_len - len(attention_mask)))

            if len(labels) < self.max_dec_seq_len:
                decoder_input_ids.extend([self.pad_token_id] * (self.max_dec_seq_len - len(decoder_input_ids)))
                labels.extend([self.pad_token_id] * (self.max_dec_seq_len - len(labels)))
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
        support_set = self.create_value(self.supports[index])
        query_set = self.create_value(self.queries[index])
        query_aug_set = self.create_value(self.queries_aug[index])
        return support_set, query_set, query_aug_set, torch.Tensor(self.classes[index]).to(torch.long)