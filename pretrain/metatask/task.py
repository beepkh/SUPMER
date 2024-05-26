from abc import abstractmethod
from torch.utils.data import Dataset

class MetaTask(Dataset):
    def __init__(self, datasets, num_tasks, num_domains, k_support, k_query, training):
        self.datasets = datasets
        self.num_tasks = num_tasks
        self.num_domains = num_domains
        self.k_support = k_support
        self.k_query = k_query
        self.training = training
    
    @abstractmethod
    def create_batch():
        raise NotImplementedError

    def __getitem__(self, index):
        pass

    def __len__(self):
        return self.num_tasks

class FinalTensorDataset(Dataset):
    def __init__(self, all_input_ids, all_attention_mask, all_decoder_input_ids, all_labels, all_decoder_attention_mask, all_loss_ids):
        self.all_input_ids = all_input_ids
        self.all_attention_mask = all_attention_mask
        self.all_decoder_input_ids = all_decoder_input_ids
        self.all_labels = all_labels
        self.all_loss_ids = all_loss_ids
        self.all_decoder_attention_mask = all_decoder_attention_mask

    def __getitem__(self, idx):
        return {'input_ids': self.all_input_ids[idx],
                'attention_mask': self.all_attention_mask[idx],
                'decoder_input_ids': self.all_decoder_input_ids[idx],
                'labels': self.all_labels[idx],
                'decoder_attention_mask': self.all_decoder_attention_mask[idx],
                'loss_ids': self.all_loss_ids[idx]}

    def get_decoder_input_ids(self):
        return self.all_decoder_input_ids
    
    def get_labels(self):
        return self.all_labels
    
    def get_decoder_attention_mask(self):
        return self.all_decoder_attention_mask
    
    def get_all_loss_ids(self):
        return self.all_loss_ids
    
    def get_input_ids(self):
        return self.all_input_ids
    
    def get_attention_mask(self):
        return self.all_attention_mask
    
    def __len__(self):
        return len(self.all_input_ids)

        