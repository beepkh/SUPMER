import torch
import random
from torch.utils.data import Dataset
from .task import MetaTask, FinalTensorDataset

class SameClusterSentSelectionTask(MetaTask):
    def __init__(self, datasets, num_tasks, num_domains, k_support, k_query, training, classify_num=5, max_enc_seq_len=1024, max_dec_seq_len=10, tokenizer=None, neg_sent = 4):
        super(SameClusterSentSelectionTask, self).__init__(datasets, num_tasks, num_domains, k_support, k_query, training)
        self.neg_sent = neg_sent
        self.pad_token_id = tokenizer.pad_token_id
        self.sentid = tokenizer.encode("The correct one is ", add_special_tokens=False)
        self.max_enc_seq_len = max_enc_seq_len
        self.max_dec_seq_len = max_dec_seq_len
        self.token_id = tokenizer.convert_tokens_to_ids(["<extra_id_0>", "<extra_id_1>", ".", "?"])
        self.label_id = tokenizer.convert_tokens_to_ids(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                                                              "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                                                              "U", "V", "W", "X", "Y", "Z"])
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
                anchor_idx_support = []
                anchor_idx_query = []
                right_ans_support = []
                right_ans_query = []
                wrong_ans_support = []
                wrong_ans_query = []
                for i in range(self.neg_sent+1):
                    anchor_idx = random.sample(all_idx, self.k_query+self.k_support)
                    right_ans = random.sample(all_idx, self.k_query+self.k_support)
                    if self.training:
                        random.shuffle(anchor_idx)
                        random.shuffle(right_ans)
                    # [group, num_for_group]
                    anchor_idx_support.append(anchor_idx[:self.k_support])
                    right_ans_support.append(right_ans[:self.k_support])
                    anchor_idx_query.append(anchor_idx[self.k_support:])
                    right_ans_query.append(right_ans[self.k_support:])

                for i in range(self.neg_sent):
                    # [group * num_for_group, 2]
                    neg_support, neg_query = [], []
                    for j in range(self.neg_sent+1):
                        for k in range(self.k_support):
                            d = random.randint(0, self.num_domains-1)
                            while d==b:
                                d = random.randint(0, self.num_domains-1)
                            d_len = self.datasets[d].__len__()
                            d_idx = random.randint(0, d_len-1)
                            neg_support.append([d, d_idx])
                        for k in range(self.k_query):
                            d = random.randint(0, self.num_domains-1)
                            while d==b:
                                d = random.randint(0, self.num_domains-1)
                            d_len = self.datasets[d].__len__()
                            d_idx = random.randint(0, d_len-1)
                            neg_query.append([d, d_idx])
                    # [choice_num, group * num_for_group, 2]
                    wrong_ans_support.append(neg_support)
                    wrong_ans_query.append(neg_query)
                
                # [group * num_for_group, sample_size]
                support_exams = []
                idx = 0
                for i in range(self.neg_sent+1):
                    for j in range(self.k_support):
                        cur_exam = [i, b]
                        cur_exam.extend([anchor_idx_support[i][j]])
                        for k in range(0, i):
                            cur_exam.extend([wrong_ans_support[k][idx][0], wrong_ans_support[k][idx][1]])
                        cur_exam.extend([b, right_ans_support[i][j]])
                        for k in range(i, self.neg_sent):
                            cur_exam.extend([wrong_ans_support[k][idx][0], wrong_ans_support[k][idx][1]])
                        idx += 1
                        support_exams.append(cur_exam)
                if self.training:
                    random.shuffle(support_exams)
                
                query_exams = []
                idx = 0
                for i in range(self.neg_sent+1):
                    for j in range(self.k_query):
                        cur_exam = [i, b]
                        cur_exam.extend([anchor_idx_query[i][j]])
                        for k in range(0, i):
                            cur_exam.extend([wrong_ans_query[k][idx][0], wrong_ans_query[k][idx][1]])
                        cur_exam.extend([b, right_ans_query[i][j]])
                        for k in range(i, self.neg_sent):
                            cur_exam.extend([wrong_ans_query[k][idx][0], wrong_ans_query[k][idx][1]])
                        idx += 1
                        query_exams.append(cur_exam)
                if self.training:
                    random.shuffle(query_exams)

                if ori_or_aug==0:
                    self.supports.append(support_exams)
                    self.queries.append(query_exams)
                else:
                    self.supports_aug.append(support_exams)
                    self.queries_aug.append(query_exams)
            
    def __getitem__(self, index):
        support_set = self.create_value(self.supports[index])
        query_set = self.create_value(self.queries[index])
        query_aug_set = self.create_value(self.queries_aug[index])
        return support_set, query_set, query_aug_set, torch.Tensor(self.classes[index]).to(torch.long)
    
    def create_value(self, examples):
        all_input_ids              = torch.empty(len(examples), self.max_enc_seq_len, dtype = torch.long)
        all_attention_mask         = torch.empty(len(examples), self.max_enc_seq_len, dtype = torch.long)
        all_decoder_input_ids      = torch.empty(len(examples), self.max_dec_seq_len, dtype = torch.long)
        all_labels                 = torch.empty(len(examples), self.max_dec_seq_len, dtype = torch.long)
        all_decoder_attention_mask = torch.empty(len(examples), self.max_dec_seq_len, dtype = torch.long)
        all_loss_ids               = torch.empty(len(examples), self.max_dec_seq_len, dtype = torch.long)

        end_ids = [self.token_id[3]] + self.sentid + [self.token_id[0]]
        max_len = self.max_enc_seq_len - len(end_ids)
        for id_, example in enumerate(examples):
            label, domain, anchor_idx = example[0], example[1], example[2]
            anchor_exam = self.datasets[domain].__getitem__(anchor_idx)['input_ids']
            input_ids = anchor_exam[:389] + [self.token_id[3]]
            # sent_len.extend(total_len)
            for i in range(self.neg_sent+1):
                input_ids.extend([self.label_id[i], self.token_id[2]])
                d, idx = example[2*i+3], example[2*i+4]
                choice_exam = self.datasets[d].__getitem__(idx)['input_ids'][:86]
                input_ids.extend(choice_exam)
                input_ids.extend([self.token_id[2]])
            
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
            
            input_ids += end_ids

            attention_mask = [1] * len(input_ids)
            target = [0, self.token_id[0]] + [self.label_id[label]] + [self.token_id[1]]
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
    
    