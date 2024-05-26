import os
import argparse
import json
import time
import logging
import math
from tqdm import tqdm
import random
import torch
from pathlib import Path
import numpy as np

from transformers import get_scheduler

from utils.utils import (
    create_dir,
    set_config_args,
    set_trainable_params_for_prompt_tuning
)
from pretrain_args import parse_args
from model.model import GenerationModel
from model.meta import Meta
from transformers import AutoTokenizer
from transformers import T5Config, T5ForConditionalGeneration
from model.grad_trans import GradientTransform
from pretrain import build_train_test_datasets, DATA_CONFIG, TASK_CONFIG

PROMPT_EMBED = "prompt_embedding"

def create_batch_of_tasks(taskset, is_shuffle = True, batch_size = 4):
    idxs = list(range(0,len(taskset)))
    if is_shuffle:
        random.shuffle(idxs)
    for i in range(0,len(idxs), batch_size):
        yield [taskset[idxs[j]] for j in range(i, min(i + batch_size,len(taskset)))]

def set_random_seed(seed):
    """Set new random seed."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmard = True

def get_dataset_and_task_name():
    return {'dataset' : ['nsp', 'nss', 'ss', 'ss', 'ss'], 
            'task':     ['ns', 'ns', 'scsp', 'scss', 'scc']
    }

def get_test_tasks(test_task_config, num_domains_list, k_support_list, k_query_list, 
                   max_seq_length_list, num_tasks_test_list, all_test_datasets, dataset_names, task_names):
    all_test_tasks = []
    for i, (dataset_name, task_name) in enumerate(zip(dataset_names, task_names)):
        test_task_config = set_config(test_task_config, num_domains_list[i], 
                                      k_support_list[i], k_query_list[i], 
                                      max_seq_length_list[i], num_tasks_test_list[i], 
                                      all_test_datasets[dataset_name])
        test_task = TASK_CONFIG[task_name](**test_task_config)
        all_test_tasks.append(test_task)
    return all_test_tasks

def set_config(task_config, num_domains, k_support, k_query, max_seq_length, num_tasks, datasets):
    task_config['num_domains'] = num_domains
    task_config['k_support'] = k_support
    task_config['k_query'] = k_query
    task_config['max_enc_seq_len'] = max_seq_length
    task_config['num_tasks'] = num_tasks
    task_config['datasets'] = datasets
    return task_config

def main():
    args = parse_args()
    set_random_seed(args.seed)
    
    dataset_and_task_names = get_dataset_and_task_name()
    dataset_names = dataset_and_task_names['dataset']
    task_names = dataset_and_task_names['task']
    
    device = args.device
    
    args.output_dir += f"multi_task_{args.total_steps}_{time.strftime('%m-%d,%H:%M:%S')}/"
    create_dir(args.output_dir)
    log_file = args.output_dir + 'pretrain.log'
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_file,
                        filemode="w",
                        format="%(asctime)s, %(levelname)s %(message)s",
                        datefmt="%H:%M:%S",
                        level=logging.INFO)

    # Add console to logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.info(args)
    
    # tokenizer
    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "use_fast": args.use_fast_tokenizer,
        "revision": args.model_revision,
        "use_auth_token": True if args.use_auth_token else None,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    
    # load meta-training data
    all_train_datasets, all_test_datasets = {}, {}
    for dataset_name, data_path, max_seq_length, num_domains in zip(args.pretrain_task, args.data_path, args.predataset_max_seq_length, args.num_domains):
        train_datasets, test_datasets = [], []
        for i in range(num_domains):
            print(f"dataset_name:{dataset_name} domain:{i}")
            train_ds, test_ds = build_train_test_datasets(
                tokenizer=tokenizer,
                data_class=DATA_CONFIG[dataset_name]["dataset"],
                data_prefix=data_path,
                data_impl=args.data_impl,
                splits_string=args.split,
                max_seq_length=max_seq_length,
                max_dec_length=args.max_dec_length,
                skip_warmup=(not args.mmap_warmup),
                domain=i,
                dataset_name=dataset_name)
            print(f"train_ds:{train_ds.__len__()}  test_ds:{test_ds.__len__()}")
            train_datasets.append(train_ds)
            test_datasets.append(test_ds)
        all_train_datasets[dataset_name] = train_datasets
        all_test_datasets[dataset_name] = test_datasets
    
    all_cluster_embeds_list = {}
    for i in range(len(args.cluster_dirs)):
        if i==0:
            curname = 'nsp'
        elif i==1:
            curname = 'nss'
        else:
            curname = 'ss'
        all_cluster_embeds = np.load(args.cluster_dirs[i])
        all_cluster_embeds_list[curname] = all_cluster_embeds
    
    # t5 model
    config_kwargs = {
        "cache_dir": args.cache_dir,
        "revision": args.model_revision,
        "use_auth_token": True if args.use_auth_token else None,
    }
    config = T5Config.from_pretrained(args.model_name_or_path, **config_kwargs)
    
    t5model = T5ForConditionalGeneration.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir,
                revision=args.model_revision,
                use_auth_token=True if args.use_auth_token else None
    )
    set_config_args(config, args)
    
    # t5 model with soft prompts
    model = GenerationModel(plm=t5model, 
                            prompt_tune=args.prompt_tune,
                            prompt_hidden_size=config.hidden_size,  
                            prompt_length = args.prompt_length,
                            init_prompt_from_vocab = args.init_prompt_from_vocab,
                            prompt_init_range = args.prompt_init_range,
                            all_cluster_embeds = all_cluster_embeds_list['ss'])
    
    if args.prompt_tune:
        print('==>create prompt embedding')
        model.create_prompt_embedding()
    
    if args.prompt_tune:
        set_trainable_params_for_prompt_tuning(model)
    
    # optimizer for soft prompts
    optimizer_grouped_parameters_model = [
        {
            "params": [p for n, p in model.named_parameters() if PROMPT_EMBED in n],
            "lr": args.outer_lr_1,
            "weight_decay": args.outer_weight_decay_1
        }
    ]
    optimizer_model = torch.optim.Adam(optimizer_grouped_parameters_model)
    
    # meta-gradient regularization
    gradientTransformer = GradientTransform(config.hidden_size, args.n_hidden, args.rank, args.init)
    optimizer_grouped_parameters_edit = [
        {
            "params": [p for p in gradientTransformer.parameters()],
            "lr": args.edit_lr,
            "weight_decay": args.edit_weight_decay
        }
    ]
    optimizer_edit = torch.optim.Adam(optimizer_grouped_parameters_edit)
    
    # learning rate scheduler
    num_tasks_train = 0
    for num in args.num_tasks_train:
        num_tasks_train += num
    num_tasks_train = int(num_tasks_train / len(args.num_tasks_train))

    num_step_in_one_epoch = math.ceil(num_tasks_train / args.outer_batch_size)
    num_epochs = math.ceil(args.total_steps / num_step_in_one_epoch)
    max_train_steps = num_step_in_one_epoch * num_epochs
    if args.outer_exist_scheduler:
        lr_scheduler_model = get_scheduler(
            name=args.outer_lr_scheduler_type,
            optimizer=optimizer_model,
            num_warmup_steps=args.outer_num_warmup_steps,
            num_training_steps=max_train_steps,
        )
        lr_scheduler_edit = get_scheduler(
            name=args.outer_lr_scheduler_type,
            optimizer=optimizer_edit,
            num_warmup_steps=args.outer_num_warmup_steps,
            num_training_steps=max_train_steps,
        )
    else:
        lr_scheduler_model = None
        lr_scheduler_edit = None
    
    # meta-learning algorithm
    model = model.to(args.device)
    gradientTransformer = gradientTransformer.to(args.device)
    metaLearner = Meta(args, model, optimizer_model, 
                       lr_scheduler_model, optimizer_edit, 
                       lr_scheduler_edit, logger, device, gradientTransformer)
    
    # construct meta tasks
    task_config = {'max_dec_seq_len': args.max_dec_length,
                   'tokenizer': tokenizer,
                   'neg_sent': args.neg_sent,
                   'classify_num': args.classify_num
                   }
               
    test_task_config = task_config
    test_task_config['training'] = False
    train_task_config = task_config
    train_task_config['training'] = True
    
    all_test_tasks = get_test_tasks(test_task_config=test_task_config,
                                    num_domains_list=args.num_domains,
                                    k_support_list=args.k_support,
                                    k_query_list=args.k_query,
                                    max_seq_length_list=args.max_seq_length,
                                    num_tasks_test_list=args.num_tasks_test,
                                    all_test_datasets=all_test_datasets,
                                    dataset_names=dataset_names,
                                    task_names=task_names)
    
    # training
    total_step = 0
    create_dir(args.output_dir + 'ckpt/') 
    spe_names = ['nsp', 'nss', 'scsp', 'scss', 'scc']
    
    for epoch in tqdm(range(num_epochs)):
        task_id = random.randint(1, 1000000) % len(task_names)
        cur_name = spe_names[task_id]
        logger.info(f"************total epoch {epoch+1} / {num_epochs}  task_name:{cur_name} ************")
        train_task_config = set_config(train_task_config, args.num_domains[task_id],
                                       args.k_support[task_id], args.k_query[task_id], 
                                       args.max_seq_length[task_id], args.num_tasks_train[task_id], 
                                       all_train_datasets[dataset_names[task_id]])
        
        train_tasks = TASK_CONFIG[task_names[task_id]](**train_task_config)
        
        # different task formats may have different parameter configs
        metaLearner.setAttr(per_device_inner_train_batch_size_0 = args.per_device_inner_train_batch_size_0[task_id], 
                            per_device_inner_train_batch_size_1 = args.per_device_inner_train_batch_size_1[task_id],
                            per_device_inner_eval_batch_size_1 = args.per_device_inner_eval_batch_size_1[task_id],
                            inner_lr_1 = optimizer_model.state_dict()['param_groups'][0]['lr'],
                            with_neg = (task_names[task_id] == 'scc'),
                            all_cluster_embeds = all_cluster_embeds_list[dataset_names[task_id]])
        
        print(f"Processing {train_tasks.__len__()} training tasks")
        train_db = create_batch_of_tasks(train_tasks, is_shuffle=True,
                                         batch_size=args.outer_batch_size)
        
        for step, train_batch in enumerate(train_db):
            avg_loss, _ = metaLearner(train_batch, total_step, training=True)
            print(f"Training batch: {total_step+1}\ttraining losses: {avg_loss}\n")
            total_step += 1

            if total_step % args.save_interval == 0 or total_step == max_train_steps:
                logger.info(f'iteration: {total_step}/{max_train_steps}\ttask_name: {cur_name}\ttraining losses: {avg_loss}')
            
            # validiation
            if total_step % args.test_step == 0 or total_step == max_train_steps: 
                logger.info(f"-------------------------------------------------------")
                logger.info(f"$$$$$$ Test in epoch {epoch+1} step{total_step} $$$$$$")
                total_loss = 0.0
                for i, (task_name, test_tasks, num_tasks_test) in enumerate(zip(task_names, all_test_tasks, args.num_tasks_test)):
                    logger.info(f"~~~ Test for task_{spe_names[i]} ~~~")
                    test_db = create_batch_of_tasks(test_tasks, is_shuffle=False, batch_size=num_tasks_test)
                    metaLearner.setAttr(per_device_inner_train_batch_size_0 = args.per_device_inner_train_batch_size_0[i], 
                                        per_device_inner_train_batch_size_1 = args.per_device_inner_train_batch_size_1[i],
                                        per_device_inner_eval_batch_size_1 = args.per_device_inner_eval_batch_size_1[i],
                                        inner_lr_1 = optimizer_model.state_dict()['param_groups'][0]['lr'],
                                        with_neg = (task_name == 'scc'),
                                        all_cluster_embeds = all_cluster_embeds_list[dataset_names[i]])
                    
                    for idx, test_batch in enumerate(test_db):
                        avg_loss, all_loss = metaLearner(test_batch, total_step, training=False) 
                        break
                    for idx, loss in enumerate(all_loss):
                        logger.info(f"  {spe_names[i]} Task id: {idx+1}\tloss: {loss}")
                    logger.info((f"~~~ {spe_names[i]} Avg Test Loss: {avg_loss} ~~~"))
                    total_loss += avg_loss
                
                logger.info((f"Epoch: {epoch+1}\tstep: {total_step}\tTest Loss: {total_loss / len(all_test_tasks)}"))
                logger.info(f"-------------------------------------------------------")

            if total_step % args.save_step == 0 or total_step == max_train_steps:               
                create_dir(args.output_dir + 'ckpt/') 
                edit_ckpt = args.output_dir + f'ckpt/edit_{total_step}.pt'
                pt_ckpt = args.output_dir + f'ckpt/prompt_{total_step}.pt'
                metaLearner.save_model(edit_ckpt, pt_ckpt)
                logger.info(f"----------Save model to {edit_ckpt} and {pt_ckpt}-------------")

if __name__ == "__main__":
    main()
                
            
        