import torch
import torch.nn as nn
from transformers.utils.dummy_pt_objects import PreTrainedModel
from utils.utils import signature
import numpy as np
from transformers.models.t5 import modeling_t5
from torch.nn import functional as F
import torch.autograd as autograd
from torch.autograd import Variable

class GenerationModel(nn.Module):
    def __init__(self, 
                 plm:PreTrainedModel,
                 prompt_tune,
                 prompt_hidden_size,  
                 prompt_length,
                 init_prompt_from_vocab,
                 prompt_init_range,
                 all_cluster_embeds=None):
        super().__init__()
        self.plm = plm
        self.prompt_tune = prompt_tune
        self.prompt_hidden_size = prompt_hidden_size  
        self.prompt_length = prompt_length
        self.init_prompt_from_vocab = init_prompt_from_vocab
        self.prompt_init_range = prompt_init_range
        self.all_cluster_embeds=all_cluster_embeds
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')
    
    
    def getprompt(self):
        return self.prompt_embedding.weight.data
    
    
    def create_prompt_embedding(self, prompt_embedding=None):
        self.prompt_embedding = nn.Embedding(self.prompt_length, self.prompt_hidden_size)
        if prompt_embedding is not None:
            print(f'Load prompt embed of size {prompt_embedding.size()}')
            self.prompt_embedding.weight.data = prompt_embedding.clone().detach()
        else:
            if self.init_prompt_from_vocab:
                indices = np.random.randint(low=0, high=5000, size=(self.prompt_length))
                self.prompt_embedding.weight.data = self.plm.get_input_embeddings().weight[indices].clone().detach()
            else:
                self.prompt_embedding.weight.data.normal_(mean=0.0, std=self.prompt_init_range)
    
    # single-sentence classification
    def append_prompts_to_input_embeds_with_negids(self, input_ids, inputs_embeds, prompt_embed_weight):
        device, dtype = input_ids.device, input_ids.dtype
        input_ids = input_ids.cpu().detach().numpy()
        cluster_id_list = []
        idx = np.where(input_ids<0)
        for i in range(len(idx[0])):
            id1, id2 = idx[0][i], idx[1][i]
            cluster_id_list.append((id1, id2, -input_ids[id1][id2]-1))
        input_ids = torch.Tensor(np.where(input_ids<0, 0, input_ids)).to(dtype).to(device)
        inputs_embeds = self.plm.get_input_embeddings()(input_ids)
        
        all_cluster_embeds = self.all_cluster_embeds
        for (pos_id_1, pos_id_2, clu_id) in cluster_id_list:
            cluster_embeds = torch.Tensor(all_cluster_embeds[clu_id]).to(inputs_embeds.dtype).to(inputs_embeds.device)
            inputs_embeds[pos_id_1, pos_id_2, :] = cluster_embeds
        
        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        # [batch_size, prompt_length, hidden_size]
        prompt_embeds = prompt_embed_weight.repeat(inputs_embeds.size(0), 1, 1)  
        inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)
        return input_ids, inputs_embeds
    
    
    def append_prompts_to_input_embeds(self, input_ids, inputs_embeds, prompt_embed_weight):
        if inputs_embeds is None:
            inputs_embeds = self.plm.get_input_embeddings()(input_ids)
        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        # [batch_size, prompt_length, hidden_size]
        prompt_embeds = prompt_embed_weight.repeat(inputs_embeds.size(0), 1, 1)  
        inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)
        return inputs_embeds


    # attention_mask for prompt
    def append_prompt_attention_mask(self, attention_mask):
        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)
        n_batches = attention_mask.shape[0]
        return torch.cat(
            [torch.full((n_batches, self.prompt_length), 1).to(attention_mask.device), attention_mask],
            dim=1,
        )


    def append_fake_prompt_ids(self, input_ids):
        n_batches = input_ids.shape[0]
        fake_id = -1 
        return torch.cat(
            [torch.full((n_batches, self.prompt_length), fake_id).to(input_ids.device), input_ids],
            dim=1,
        )
    
    
    def get_prompt_weight(self, meta_loss=None, meta_step_size=None, 
                          stop_gradient=False, gradientTransform=None):
        weight = self.prompt_embedding.weight
        if meta_loss is not None:
            print('calculate grad!')
            if not stop_gradient:
                prompt_grad = autograd.grad(meta_loss, weight, create_graph=False, retain_graph=True)[0]
            else:
                prompt_grad = Variable(autograd.grad(meta_loss, weight, create_graph=False,retain_graph=True)[0].data, requires_grad=False)
            weight = weight - gradientTransform(prompt_grad) * meta_step_size
            return weight
        else:
            return weight
        
        
    def append_prompts(self, input_ids, attention_mask, inputs_embeds, prompt_embed_weight, with_neg = False):
        # Appends a faken token id to input with the length of number of prefixes.
        # Extend the attention matrix.
        if with_neg == False:
            inputs_embeds = self.append_prompts_to_input_embeds(input_ids, inputs_embeds, prompt_embed_weight)
        else:
            input_ids, inputs_embeds = self.append_prompts_to_input_embeds_with_negids(input_ids, inputs_embeds, 
                                                                                       prompt_embed_weight)
        attention_mask = self.append_prompt_attention_mask(attention_mask)
        # Appends a fake id to the input_ids.
        input_ids = self.append_fake_prompt_ids(input_ids)
        return input_ids, attention_mask, inputs_embeds 
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        loss_ids = None,
        with_neg = False,
        forward_choice = 0,
        labels1 = None,
        labels2 = None,
        lamda = 0.0,
        meta_loss=None, 
        meta_step_size=None, 
        stop_gradient=False, 
        gradientTransform=None,
        prompt_embed_weight=None
    ):
        if forward_choice == 1:
            return self.forward_encoder(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        inputs_embeds=inputs_embeds,
                                        head_mask=head_mask,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict,
                                        with_neg=with_neg,
                                        meta_loss=meta_loss, 
                                        meta_step_size=meta_step_size, 
                                        stop_gradient=stop_gradient,
                                        gradientTransform=gradientTransform,
                                        prompt_embed_weight=prompt_embed_weight)
        elif forward_choice == 2:
            return self.forward_decoder(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        decoder_input_ids=decoder_input_ids,
                                        decoder_attention_mask=decoder_attention_mask,
                                        head_mask=head_mask,
                                        decoder_head_mask=decoder_head_mask,
                                        cross_attn_head_mask=cross_attn_head_mask,
                                        encoder_outputs=encoder_outputs,
                                        past_key_values=past_key_values,
                                        inputs_embeds=inputs_embeds,
                                        decoder_inputs_embeds=decoder_inputs_embeds,
                                        use_cache=use_cache,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict,
                                        labels1 = labels1,
                                        labels2 = labels2,
                                        lamda = lamda)
            
        if self.prompt_tune:
            if prompt_embed_weight is None:
                prompt_embed_weight = self.get_prompt_weight(meta_loss, meta_step_size, 
                                                             stop_gradient, gradientTransform)
            input_ids, attention_mask, inputs_embeds = self.append_prompts(input_ids, attention_mask, 
                                                                           inputs_embeds, prompt_embed_weight, with_neg)
            
        output = self.plm(
            input_ids=input_ids if not self.prompt_tune else None,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = output.logits
        if labels is not None:
            if loss_ids is not None:
                labels = torch.where(loss_ids>0, labels, -100)
            batch_size, seq_len, vocab_size = logits.shape
            loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss.view(batch_size, -1).sum(dim=-1)
            loss = loss.mean()
            output.loss = loss
        
        return output
    
    def forward_encoder(self,
                        input_ids=None,
                        attention_mask=None,
                        inputs_embeds=None,
                        head_mask=None,
                        output_attentions=None,
                        output_hidden_states=None,
                        return_dict=None,
                        with_neg=False,
                        meta_loss=None, meta_step_size=None, stop_gradient=False, gradientTransform=None,
                        prompt_embed_weight=None):
        if self.prompt_tune:
            if prompt_embed_weight is None:
                prompt_embed_weight = self.get_prompt_weight(meta_loss, meta_step_size, 
                                                             stop_gradient, gradientTransform)
            input_ids, attention_mask, inputs_embeds = self.append_prompts(input_ids, attention_mask, inputs_embeds, 
                                                                           prompt_embed_weight, with_neg)
        encoder = self.plm.get_encoder() 
        encoder_outputs = encoder(
            input_ids=input_ids if not self.prompt_tune else None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        return encoder_outputs, attention_mask
    
    def forward_decoder(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels1 = None,
        labels2 = None,
        lamda = 0.0
    ):
        output = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels1,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # mixup labels
        batch_size, seq_len, vocab_size = output.logits.size()
        labels_1 = labels1[:, :2].contiguous().view(-1, 1)
        cur_size = labels_1.size(0)
        labels_onehot_1 = torch.zeros(cur_size, vocab_size).to(labels_1.device).scatter_(1, labels_1, 1)
        
        labels_2 = labels2[:, :2].contiguous().view(-1, 1)
        cur_size = labels_2.size(0)
        labels_onehot_2 = torch.zeros(cur_size, vocab_size).to(labels_2).scatter_(1, labels_2, 1)
        labels_onehot = labels_onehot_1 * lamda + labels_onehot_2 * (1 - lamda)
        # cal loss
        logits = F.log_softmax(output.logits[:,:2,:].contiguous().view(-1,vocab_size))
        kl_loss = F.kl_div(target=labels_onehot, input=logits, reduction='none', log_target=False).sum(dim=-1)
        kl_loss = kl_loss.view(batch_size, -1).sum(dim=-1)
        output.loss = kl_loss
        return output
        
        
        
