#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch 
import logging
import numpy as np 
import os
import sys
from model.grad_trans import GradientTransform

os.environ["WANDB_DISABLED"] = "true"

import datasets

import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
    set_seed
)

from utils.utils import (
    load_json,
    set_config_args,
    set_trainable_params_for_prompt_tuning
)
from model.model import GenerationModel
from transformers import T5Config, T5ForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from training_args import ModelArguments, DataTrainingArguments, FewShotTrainingArguments
from model.trainer import BaseTrainer
from data.tasks import AutoTask
from data.processors import AutoProcessor
from data.processing import MLMProcessor

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.10.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FewShotTrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    
    if training_args.classifier_eval or training_args.prototypical_eval:
        assert training_args.classifier_eval != training_args.prototypical_eval
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
        
    task = AutoTask.get(
        task=data_args.task, 
        data_seed=data_args.data_seed, 
        num_samples=data_args.K,
        cache_dir=model_args.cache_dir, 
        data_dir=data_args.data_dir)
    raw_datasets = task.get_datasets()
    
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    
    processor = AutoProcessor.get(
        task=data_args.task,
        tokenizer=tokenizer,
        with_pattern=not training_args.soft_pet and not data_args.no_pattern,
        pattern_id=data_args.pattern_id,
        mask_position=training_args.mask_position
    )
    
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    config = T5Config.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    
    set_config_args(config, training_args)
    # config.num_labels = task.num_labels
    
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    
    processor = MLMProcessor(
        tokenizer=tokenizer,
        tokenized_verbalizers=None ,
        max_seq_length=data_args.max_seq_length,
        max_dec_length=data_args.max_dec_length,
        processor=processor,
        mask_length = None,
        train_classifier = training_args.train_classifier  
    )
    config.mask_token_id = tokenizer.mask_token_id
    config.pad_token_id = tokenizer.pad_token_id
    
    t5model = T5ForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None
    )
    
    model = GenerationModel(plm=t5model, 
                            prompt_tune=training_args.prompt_tune,
                            prompt_hidden_size=config.hidden_size,  
                            prompt_length = training_args.prompt_length,
                            init_prompt_from_vocab = training_args.init_prompt_from_vocab,
                            prompt_init_range = training_args.prompt_init_range)
    
    if training_args.prompt_tune:
        print('==>create prompt embedding')
        if training_args.prompt_path is not None:
            prompt_embed = torch.load(training_args.prompt_path)
            model.create_prompt_embedding(prompt_embed)
        else:
            model.create_prompt_embedding()
            
    if training_args.prompt_tune:
        set_trainable_params_for_prompt_tuning(model)
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    def extract_targets(examples):
        targets = examples["label"]
        targets = [int(target) for target in targets]
        return {"targets": targets}
        
    if training_args.do_train:
        if "train" not in raw_datasets: 
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
            processor,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=False, #not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
            )
    
    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_targets = eval_dataset.map(
            extract_targets,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on eval dataset",    
            )
            eval_dataset = eval_dataset.map(
                    processor,
                    batched=False,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=False, #not data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
            )
    
    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_targets = predict_dataset.map(
            extract_targets,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on predict dataset",    
            )
            predict_dataset = predict_dataset.map(
                processor,
                batched=False,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file= False, #not data_args.overwrite_cache,
                desc="Running tokenizer on predict dataset",
            )
    
    data_collator = default_data_collator
    all_datasets = {"train": train_dataset, "eval": eval_dataset, "predict": predict_dataset}
    extra_info = {k: v["extra_fields"] for k, v in all_datasets.items()} 
    train_dataset = train_dataset.remove_columns("extra_fields")
    eval_dataset = eval_dataset.remove_columns("extra_fields")
    predict_dataset = predict_dataset.remove_columns("extra_fields")
    
    
    gradient_transformer = GradientTransform(config.hidden_size, training_args.n_hidden, training_args.rank, training_args.init, choice=training_args.choice)
    if training_args.edit_model_path is not None:
        print(f'load edit model from {training_args.edit_model_path}')
        gradient_transformer.load_state_dict(torch.load(training_args.edit_model_path))
    gradient_transformer.to('cuda:0')
        
    trainer = BaseTrainer(
        model=model,
        args=training_args,
        gradient_transformer=gradient_transformer,
        eval_targets=eval_targets,
        train_dataset=train_dataset,
        eval_dataset=predict_dataset,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
        task=data_args.task,
        metrics=task.metric,
        extra_info = extra_info
    )

    if trainer.is_world_process_zero():
       os.makedirs(training_args.output_dir, exist_ok=True)
       trainer.save_metrics("arguments", load_json(sys.argv[1]))

    # Training
    performance_metrics = {}
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if training_args.compute_time:
            torch.cuda.synchronize()  # wait for move to complete
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        if training_args.compute_time:
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            total_time = start.elapsed_time(end)/(1000*60)
            performance_metrics.update({"training time(min)": total_time})
        
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


    if torch.cuda.is_available() and training_args.compute_memory:
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
        performance_metrics.update({"peak_memory(GB)": peak_memory})
    
    if training_args.compute_memory or training_args.compute_time and not training_args.compute_inference_time:
        print(performance_metrics)
        trainer.save_metrics("performance", performance_metrics)


    if training_args.do_predict:
        logger.info("*** Predict ***")

        if training_args.compute_inference_time:
            torch.cuda.synchronize()  # wait for move to complete
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        metrics = trainer.evaluate(
            eval_datasets=predict_dataset, 
            eval_targets=predict_targets, 
            metric_key_prefix="predict"
        )

        if training_args.compute_inference_time:
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            total_time = start.elapsed_time(end)/(1000*60)
            performance_metrics.update({"inference time(min)": total_time})

        predict_samples = predict_dataset[0].num_rows if isinstance(predict_dataset, list) else predict_dataset.num_rows
        max_predict_samples = data_args.max_predict_samples if data_args.max_predict_samples is not None else predict_samples
        metrics["predict_samples"] = min(max_predict_samples, predict_samples)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        print(metrics)
    
    if training_args.compute_memory or training_args.compute_time or training_args.compute_inference_time:
        print(performance_metrics)
        trainer.save_metrics("performance", performance_metrics)


if __name__ == "__main__":
    main()
