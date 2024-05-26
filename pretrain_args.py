# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from transformers import SchedulerType

def parse_args():
    parser = argparse.ArgumentParser(description="meta training")

    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--seed", type=int, default=1, help="A seed for reproducible training.")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./outputs/pretrain/",
        help="to save the pretrain output"
    )
    parser.add_argument(
        "--test_step",
        type=int,
        default=100,
        help="number of outer epochs for test",
    )
    parser.add_argument(
        "--save_step",
        type=int,
        default=100,
        help="number of outer epochs for checkpoints saving",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help="number of loss saving",
    )
    parser.add_argument(
        "--outer_batch_size",
        type=int,
        default=4,
        help="number of tasks in each batch",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=100000,
        help="total steps for outer optimizer",
    )
    # data load
    parser.add_argument('--pretrain_task', type=str, nargs='+',
                       help="prompr pretraining tasl, one from [nsp, nss, singsent]")
    parser.add_argument("--data_path", type=str, nargs='+',
                       help="Path to combined dataset to split.")
    parser.add_argument("--predataset_max_seq_length", type=int, nargs='+',
                        help="the max seq length for pretrain datasets")
    parser.add_argument('--data_impl', type=str, default='infer',
                       choices=['lazy', 'cached', 'mmap', 'infer'],
                       help='Implementation of indexed datasets.')
    parser.add_argument('--mmap_warmup', action='store_true',
                       help='Warm up mmap files.')
    parser.add_argument('--split', default='9,1',
                       help='comma-separated list of proportions for training,'
                       ' validation, and test split')
    parser.add_argument("--cluster_dirs", type=str, nargs='+', default=None)
    
    # task set
    parser.add_argument("--num_tasks_train", type=int, nargs='+',
                        help="the number of meta tasks in meta learning.")
    parser.add_argument("--num_tasks_test", type=int, nargs='+',
                        help="the number of meta tasks in meta learning.")
    parser.add_argument("--num_domains", type=int, nargs='+',
                        help="total number of domains")
    parser.add_argument("--k_support", type=int, nargs='+',
                        help="the size of support size in a meta task")
    parser.add_argument("--k_query", type=int, nargs='+',
                        help="the size of query size in a meta task")
    parser.add_argument("--max_seq_length", type=int, nargs='+',
                       help="Maximum encoder sequence length to process")
    parser.add_argument("--max_dec_length", type=int, default=3,
                       help="Maximum decoder sequence length to process")
    parser.add_argument("--neg_sent", type=int, default=4,
                        help="the number of negative sentences in meta task")
    parser.add_argument("--classify_num", type=int, default=5,
                        help="the number of classification numbers")
    
    # models 
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='./t5base',
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Where do you want to store the pretrained models downloaded from huggingface.co",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        type=bool,
        default=True,
        help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.",
    )
    parser.add_argument(
        "--model_revision",
        type=str,
        default='main',
        help="The specific model version to use (can be a branch name, tag name or commit id).",
    )
    parser.add_argument(
        "--use_auth_token",
        type=bool,
        default=False,
        help="Will use the token generated when running `transformers-cli login` (necessary to use this script with private models).",
    )
    
    #prompt config 
    parser.add_argument(
        "--prompt_tune",
        type=bool,
        default=True,
        help="If sets, adds prompts token to the input and only tune them.",
    )
    parser.add_argument(
        "--prompt_length",
        type=int,
        default=100,
        help="Sets the number of tokens for prompt-tuning.",
    )
    parser.add_argument(
        "--init_prompt_from_vocab",
        type=bool,
        default=True,
        help="If set, initializes the prompt tokens' embedding from the given pretrained model's vocabulary.",
    )
    parser.add_argument(
        "--prompt_init_range",
        type=float,
        default=1e-4,
        help="Defines the initialization range.",
    )

    parser.add_argument(
        "--num_inner_train_epochs", 
        type=int, 
        default=7, 
        help="Total number of training epochs in inner loop."
    )
    parser.add_argument(
        "--per_device_inner_train_batch_size_0",
        type=int,
        nargs='+',
        help="Batch size (per device) for the support training dataloader.",
    )
    parser.add_argument(
        "--per_device_inner_train_batch_size_1",
        type=int,
        nargs='+',
        help="Batch size (per device) for the query training dataloader.",
    )
    parser.add_argument(
        "--per_device_inner_eval_batch_size_1",
        type=int,
        nargs='+',
        help="Batch size (per device) for the query evaluation dataloader.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        nargs='+',
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--inner_lr_1",
        type=float,
        default=1e-1,
        help="Initial learning rate for prompt tuning in inner loop",
    )
    parser.add_argument(
        "--inner_lr_2",
        type=float,
        default=1e-5,
        help="Initial learning rate for lmhead tuning in inner loop",
    )
    parser.add_argument(
        "--outer_lr_1",
        type=float,
        default=1e-1,
        help="Initial learning rate for prompt tuning in outer loop",
    )
    parser.add_argument(
        "--outer_lr_2",
        type=float,
        default=1e-5,
        help="Initial learning rate for prompt tuning in outer loop",
    )
    parser.add_argument(
        "--inner_lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use for inner loop",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--outer_lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use for outer loop",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--tune_lmhead",
        action="store_true",
        help="Whether to tune the lmhead.",
    )
    parser.add_argument(
        "--inner_exist_scheduler",
        action="store_true",
        help="Whether to tune the lmhead.",
    )
    parser.add_argument(
        "--outer_exist_scheduler",
        action="store_false",
        help="Whether to tune the lmhead.",
    )
    
    parser.add_argument("--outer_weight_decay_1", type=float, default=0.0, help="Weight decay for prompt tuning in outer loop.")
    
    parser.add_argument(
        "--outer_num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    # new add for edit
    parser.add_argument(
        "--edit_lr",
        type=float,
        default=1e-4,
        help="Initial learning rate for grad edit model",
    )
    parser.add_argument(
        "--edit_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for grad edit model",
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=1,
        help="number of hidden layers in grad edit model",
    )
    parser.add_argument(
        "--init",
        type=str,
        default='id',
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1920,
    )
    parser.add_argument(
        "--stop_gradient",
        type=bool,
        default=True,
        help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.",
    )
    parser.add_argument(
        "--spt_coff",
        type=float,
        default=1.0,
        help="the coff of support set loss",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="alpha for beta distribution",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="weight of consistency loss",
    )
    
    args = parser.parse_args()
    return args