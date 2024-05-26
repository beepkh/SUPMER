from scipy import spatial
import numpy as np
from torch.nn import functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from model.grad_trans import GradientTransform
import torch.autograd as autograd

PROMPT_EMBED = "prompt_embedding"

class Meta(nn.Module):
    def __init__(self, args, model, optimizer_model, lr_scheduler_model, optimizer_edit, lr_scheduler_edit, logger, device, gradientTransformer:GradientTransform, with_neg=False):
        super(Meta, self).__init__()
        self.args = args
        self.model = model
        self.optimizer_model = optimizer_model
        self.lr_scheduler_model = lr_scheduler_model
        self.optimizer_edit = optimizer_edit
        self.lr_scheduler_edit = lr_scheduler_edit
        
        self.logger = logger
        self.device = device
        self.with_neg = with_neg
        self.gradientTransformer = gradientTransformer
        self.alpha = args.alpha
        self.per_device_inner_train_batch_size_0 = args.per_device_inner_train_batch_size_0
        self.per_device_inner_train_batch_size_1 = args.per_device_inner_train_batch_size_1
        self.per_device_inner_eval_batch_size_1 = args.per_device_inner_eval_batch_size_1
        self.all_cluster_embed = None
        
        self.m = 2.0
        self.total_qry_spt_sim = 0.0
        self.grad_lamdba = 0.5
        self.loss_lamdba = 1.0

        
    def get_alpha(self):
        coeff = (self.m ** self.total_qry_spt_sim - 1) / (self.m - 1) + 1e-7
        alpha = np.random.beta(self.alpha, self.alpha * coeff)
        return alpha
    
    
    def get_domain_sim(self, domain_info):
        domain_info_list = domain_info.tolist()
        domain_1, domain_2 = domain_info_list[0], domain_info_list[1]
        domain_embed_list_1, domain_embed_list_2 = [], []
        
        for d in domain_1:
            domain_embed_list_1.append(self.all_cluster_embeds[d])
        domain_embed_mean_1 = np.mean(domain_embed_list_1, axis=0)
        
        for d in domain_2:
            domain_embed_list_2.append(self.all_cluster_embeds[d])
        domain_embed_mean_2 = np.mean(domain_embed_list_2, axis=0)
        
        cos_sim = 1 - spatial.distance.cosine(domain_embed_mean_1, domain_embed_mean_2) / 2
        return cos_sim
              
    
    def setAttr(self, per_device_inner_train_batch_size_0, per_device_inner_train_batch_size_1, per_device_inner_eval_batch_size_1, 
                inner_lr_1, with_neg, all_cluster_embeds=None):
        self.per_device_inner_train_batch_size_0 = per_device_inner_train_batch_size_0
        self.per_device_inner_train_batch_size_1 = per_device_inner_train_batch_size_1
        self.per_device_inner_eval_batch_size_1 = per_device_inner_eval_batch_size_1
        self.inner_lr_1 = inner_lr_1
        self.with_neg = with_neg
        self.all_cluster_embeds = all_cluster_embeds
    
     
    def save_model(self, save_dir_1, save_dir_2):
        torch.save(self.gradientTransformer.state_dict(), save_dir_1)
        torch.save(self.model.getprompt(), save_dir_2)
        
                  
    def forward(self, batch_tasks, global_step, training=True):
        all_task_loss = []
        total_loss = 0.0
        total_qry_spt_sim = 0.0
        num_task = len(batch_tasks)
        for task_id, task in enumerate(batch_tasks):
            
            dataset_spt, dataset_qry, dataset_qry_new, domain_info = task[0], task[1], task[2], task[3]
            domain_sim = self.get_domain_sim(domain_info)
            # support set
            dataloader_spt = DataLoader(
                dataset_spt, shuffle=True, batch_size=self.per_device_inner_train_batch_size_0
            )
            # query set
            dataloader_qry = DataLoader(
                dataset_qry, shuffle=training, batch_size=(self.per_device_inner_train_batch_size_1 if training else self.per_device_inner_eval_batch_size_1),
            )
            # aug set
            if training:
                dataloader_qry_new = DataLoader(
                    dataset_qry_new, shuffle=training, batch_size=self.per_device_inner_train_batch_size_1,
                )
                
            meta_loss_spt = 0.0
            prompt_grad = 0.0
            self.model.train()
            print(f"train support set for task id {task_id} in global step {global_step}:")
            # inner loop
            for step, batch_spt in enumerate(dataloader_spt):
                batch_spt = {k:batch_spt[k].to(self.device) for k in batch_spt}
                outputs = self.model(input_ids=batch_spt['input_ids'],
                                     attention_mask=batch_spt['attention_mask'],
                                     decoder_input_ids=batch_spt['decoder_input_ids'],
                                     labels=batch_spt['labels'],
                                     decoder_attention_mask=batch_spt['decoder_attention_mask'],
                                     loss_ids=batch_spt['loss_ids'],
                                     with_neg=self.with_neg)
                loss = outputs.loss
                encoder_last_hidden_state = outputs.encoder_last_hidden_state.detach() / len(dataloader_spt)
                if step==0:
                    all_encoder_last_hidden_state = torch.mean(torch.mean(encoder_last_hidden_state, 0), 0)
                else:
                    all_encoder_last_hidden_state += torch.mean(torch.mean(encoder_last_hidden_state, 0), 0)
                loss = loss / len(dataloader_spt)
                prompt_grad += autograd.grad(loss, self.model.prompt_embedding.weight, create_graph=False)[0]
                meta_loss_spt += loss.detach().float().item()
                
            print(f"\tloss: {meta_loss_spt}")
            self.optimizer_model.zero_grad()
            
            # outer loop
            if training:
                self.gradientTransformer.train()
            else:
                self.gradientTransformer.eval()
            meta_loss_qry = 0.0
            # training
            if training:
                print(f"train query set for task id {task_id} in global step {global_step}:")
                grad_before = self.model.prompt_embedding.weight.grad
                self.model.train()
                for step, (batch_qry,batch_qry_new) in enumerate(zip(dataloader_qry, dataloader_qry_new)):
                    batch_1 = {k:batch_qry[k].to(self.device) for k in batch_qry}
                    batch_2 = {k:batch_qry_new[k].to(self.device) for k in batch_qry_new}
                    lamda = self.get_alpha()
                    new_grad, _ = self.gradientTransformer(prompt_grad, all_encoder_last_hidden_state)
                    prompt_embed_weight = self.model.prompt_embedding.weight - new_grad * self.args.inner_lr_1
                    loss = get_mixup_loss(batch_1, batch_2, self.model, lamda, self.device, meta_loss_spt, self.args.inner_lr_1, self.args.stop_gradient, prompt_embed_weight, self.with_neg)
                    meta_loss_qry += (loss / len(dataloader_qry)).detach().float().item()
                    loss = (loss / len(dataloader_qry)) / num_task
                    loss.backward()
                    
                # inner product
                spt_grad, coeff = self.gradientTransformer(prompt_grad, all_encoder_last_hidden_state)
                with torch.no_grad():
                    grad_after = self.model.prompt_embedding.weight.grad
                    if grad_before is not None:
                        qry_grad = (grad_after-grad_before) * num_task
                    else:
                        qry_grad = grad_after * num_task
                    qry_spt_sim = (1 + torch.mean(F.cosine_similarity(qry_grad, spt_grad))).item() / 2
                reg_loss = self.gradientTransformer.reg_loss(domain_sim, qry_spt_sim, coeff) / num_task
                reg_loss.backward()
                total_qry_spt_sim += qry_spt_sim
                
                print(f"\tloss: {meta_loss_qry}")    
                total_loss += meta_loss_qry
                all_task_loss.append(meta_loss_qry)
            
            # validation 
            else:
                print(f"eval query set for task id {task_id} in global step {global_step}:")
                self.model.eval()
                for step, batch_qry in enumerate(dataloader_qry):
                    with torch.no_grad():
                        batch_qry = {k:batch_qry[k].to(self.device) for k in batch_qry}
                        new_grad, _ = self.gradientTransformer(prompt_grad, all_encoder_last_hidden_state)
                        prompt_embed_weight = self.model.prompt_embedding.weight - new_grad * self.args.inner_lr_1
                        outputs = self.model(input_ids=batch_qry['input_ids'],
                                            attention_mask=batch_qry['attention_mask'],
                                            decoder_input_ids=batch_qry['decoder_input_ids'],
                                            labels=batch_qry['labels'],
                                            decoder_attention_mask=batch_qry['decoder_attention_mask'],
                                            loss_ids=batch_qry['loss_ids'],
                                            with_neg=self.with_neg,
                                            prompt_embed_weight=prompt_embed_weight)
                        loss = outputs.loss
                        meta_loss_qry += loss.float().item()
                meta_loss_qry /= len(dataloader_qry)
                print(f"\tloss: {meta_loss_qry}")  
                total_loss += meta_loss_qry
                all_task_loss.append(meta_loss_qry)
                
        # update
        total_loss /= num_task
        
        if training:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.gradientTransformer.parameters(), 1.0)
            self.optimizer_model.step()
            self.optimizer_edit.step()
            self.optimizer_model.zero_grad()
            self.optimizer_edit.zero_grad()
            if self.lr_scheduler_model is not None:
                self.lr_scheduler_model.step() 
            if self.lr_scheduler_edit is not None:
                self.lr_scheduler_edit.step()
            
            total_qry_spt_sim /= num_task
            if self.total_qry_spt_sim == 0.0:
                self.total_qry_spt_sim = total_qry_spt_sim
            else:     
                self.total_qry_spt_sim = (self.grad_lamdba * self.total_qry_spt_sim + 
                                         (1 - self.grad_lamdba) * total_qry_spt_sim)
            
        return total_loss, all_task_loss
                

def get_mixup_loss(batch_1, batch_2, model, lamda, device, meta_loss=None, meta_step_size=None, stop_gradient=False, prompt_embed_weight=None, with_neg=False):
    # mixup encoder outputs
    encoder_outputs_1, attention_mask_1 = model(input_ids=batch_1['input_ids'], 
                                                attention_mask=batch_1['attention_mask'], 
                                                with_neg=with_neg, forward_choice=1,
                                                meta_loss=meta_loss, meta_step_size=meta_step_size, 
                                                stop_gradient=stop_gradient, prompt_embed_weight=prompt_embed_weight)
    
    encoder_outputs_2, attention_mask_2 = model(input_ids=batch_2['input_ids'], 
                                                attention_mask=batch_2['attention_mask'], 
                                                with_neg=with_neg, forward_choice=1,
                                                meta_loss=meta_loss, meta_step_size=meta_step_size, 
                                                stop_gradient=stop_gradient, prompt_embed_weight=prompt_embed_weight)
    
    hidden_states_1 = encoder_outputs_1[0]
    hidden_states_2 = encoder_outputs_2[0]
    hidden_states = hidden_states_1 * lamda + hidden_states_2 * (1 - lamda)
    encoder_outputs = encoder_outputs_1
    encoder_outputs.last_hidden_state = hidden_states
    # get attention mask
    attention_mask = torch.where(attention_mask_1 > attention_mask_2, attention_mask_1, attention_mask_2)
    # get final output
    outputs = model(encoder_outputs=encoder_outputs, attention_mask=attention_mask, labels1 = batch_1['labels'], labels2 = batch_2['labels'],
                    decoder_input_ids=batch_1['decoder_input_ids'], decoder_attention_mask=batch_1['decoder_attention_mask'], forward_choice=2, lamda=lamda)
    kl_loss = outputs.loss.mean()
    return kl_loss




