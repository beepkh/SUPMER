import os
import sys
import functools
import numpy as np

import time
import math
import torch
from typing import Dict, List, Optional, Union, Any
from torch.utils.checkpoint import checkpoint_sequential
from tqdm.auto import tqdm
import warnings
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch import nn
import torch.nn.functional as F
import collections


from transformers import __version__
from transformers.configuration_utils import PretrainedConfig

from transformers import Trainer
from transformers.file_utils import (
    is_torch_tpu_available,
    is_sagemaker_mp_enabled,
    is_sagemaker_dp_enabled,
    is_apex_available,
    CONFIG_NAME
)
from transformers.deepspeed import deepspeed_init
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow

from transformers.utils import logging
from transformers.trainer_utils import (
    EvalLoopOutput,
    ShardedDDPOption,
    speed_metrics,
    TrainOutput,
    get_last_checkpoint,
    set_seed
)
from transformers.trainer_callback import (
    TrainerState,
)

from transformers.integrations import (
    is_fairscale_available,
    hp_params
)
from transformers.trainer_pt_utils import (
    get_parameter_names,
    IterableDatasetShard
)
from transformers.optimization import (
    Adafactor,
    AdamW
)
from transformers.file_utils import WEIGHTS_NAME

if is_apex_available():
    from apex import amp

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

from fewshot.utils.utils import compute_accuracy_from_losses, get_aggregation 
from fewshot.third_party.models import RobertaForMaskedLM

if is_fairscale_available():
    dep_version_check("fairscale")
    from fairscale.optim import OSS


if is_sagemaker_dp_enabled():
    import smdistributed.dataparallel.torch.distributed as dist
else:
    import torch.distributed as dist


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

from utils.utils import get_aggregation, trim_input_ids, create_dir  

logger = logging.get_logger(__name__)


SOFT_MASK_LABELS = "extra_embeddings"
PROMPT_EMBED = "prompt_embedding"
from transformers.trainer import Trainer

class BaseTrainer(Trainer):
    def __init__(self, gradient_transformer=None,
                 eval_targets=None, 
                 task=None, 
                 metrics=None, 
                 extra_info=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_transformer = gradient_transformer
        self.eval_targets = eval_targets
        self.task = task
        self.metrics = metrics
        self.extra_info = extra_info 

    def evaluate(
        self,
        eval_datasets: Optional[Dataset] = None,
        eval_targets: Optional[Dataset] = None, 
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        start_time = time.time()
        output = self.eval_loop(
            eval_datasets = eval_datasets,
            eval_targets = eval_targets,
            description="Evaluation",
            metric_key_prefix=metric_key_prefix
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

 
    def eval_loop(
        self,
        eval_datasets,
        eval_targets,
        description: str,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """Evaluation/Prediction loop."""
        logger.info(f"***** Running {description} *****")

        model = self._wrap_model(self.model, training=False)
        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        eval_datasets = eval_datasets if eval_datasets is not None else self.eval_dataset
        eval_targets = eval_targets if eval_targets is not None else self.eval_targets 
        num_samples = eval_datasets[0].num_rows if isinstance(eval_datasets, list) else eval_datasets.num_rows
        logger.info(f"  Num examples = {num_samples}")

        model.eval()
        metrics = self.compute_pet_metrics(eval_datasets, model, self.extra_info[metric_key_prefix])

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples)

    def _get_per_token_train_centroids_from_label_embeddings(self, model):
        centroids = {}
        start = 0
        num_masks = model.num_masks
        for label in range(self.model.config.num_labels):
            centroids[label] = model.extra_embeddings.weight.data[start:start+num_masks]
            start += num_masks 
        return centroids 

    def compute_pet_metrics(self, eval_datasets, model, extra_info):
        dataloader = self.get_eval_dataloader(eval_datasets)
        centroids=None
        if self.args.prototypical_eval:
            if self.args.label_embeddings_as_centroids:
                centroids = self._get_per_token_train_centroids_from_label_embeddings(model)
            else:
                centroids = self._compute_per_token_train_centroids(model)

        y_hats = []
        labels = []
        for _, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                if self.args.train_classifier or self.args.classifier_eval:
                    logits = model(**inputs)["logits"][:,1,:]
                else:
                    logits = self.evaluate_pet(model, inputs, centroids=centroids)
                y_hat = torch.argmax(logits, axis=1).cpu().detach().numpy()
                y_hats.extend(y_hat) 
                cur_label = inputs["labels"][:,1].cpu().detach()
                labels.extend(cur_label.numpy())

        results = {}
        for metric in self.metrics:
            results.update(metric(y_hats, labels, extra_info))
        results["average"] = np.mean(list(results.values()))
        return results

    def evaluate_pet(self, model, batch, centroids=None):
        """Evaluates the model on the given inputs."""
        candidates_ids = batch["candidates_ids"]
        candidates_ids = candidates_ids.permute(1, 0, 2)
        num_labels = candidates_ids.shape[0]
        log_probs = []

        for label in range(num_labels):
            candidate_labels = candidates_ids[label]

            if self.args.soft_pet:
                if self.args.prototypical_eval:
                    log_prob = self._get_prototypical_candidate_eval_probability(model, batch, label, centroids)
                else:
                    log_prob = self._get_candidate_soft_log_probability_with_extra_tokens(model, batch, label, 
                         decoding_strategy=self.args.decoding_strategy)
            else:
                log_prob = self._get_candidate_log_probability(model, batch, candidate_labels[0],
                    decoding_strategy=self.args.decoding_strategy)
            log_probs.append(log_prob)
        
        result = torch.tensor([log_probs])
        
        if self.args.prototypical_eval:
            result = result.squeeze()
            result = result.permute(1, 0)
        return result 

    def get_masks_embeds(self, model, batch):
        """Returns mask embeddings of size batch_size x num_masks x hidden_dim"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        if self.args.prompt_tune:
            input_ids, attention_mask, inputs_embeds = model.append_prompts(input_ids, attention_mask, inputs_embeds=None)
            hidden_states = model.roberta(input_ids=None, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        else:
            hidden_states = model.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = hidden_states[0]
        batch_size = input_ids.shape[0]
        mask_indices = (input_ids == model.config.mask_token_id).nonzero()[:, -1].view(batch_size, -1)
        return hidden_states[torch.arange(hidden_states.shape[0]).unsqueeze(-1), mask_indices]

    def _compute_per_token_train_centroids(self, model):
        """For training datapoints belonging to each label, computes the average embedding of masked tokens
        across all samples of each label. 
        Returns a dictionary from labels to embedding size of shape [num_tokens, hidden_dim]"""
        def get_label_samples(dataset, label):
            return dataset.filter(lambda example: int(example['labels']) == label)
        label_to_token_centroids = {}
        for label in range(self.model.config.num_labels):
            data = get_label_samples(self.train_dataset, label)
            dataloader = self.get_eval_dataloader(data)
            mask_embeds = [] 
            for _, inputs in enumerate(dataloader):
                batch = self._prepare_inputs(inputs)
                with torch.no_grad():
                    mask_embeds.append(self.get_masks_embeds(model, batch))
            # Computes the averaged mask embeddings for the samples of this label.
            label_to_token_centroids[label] = torch.mean(torch.cat(mask_embeds, dim=0), dim=0)
        return label_to_token_centroids

    def _get_prototypical_candidate_eval_probability(self, model, batch, label, centroids):
        def cosine_similarity(embed1, embed2):
            embed1 = F.normalize(embed1, dim=-1)
            embed2 = F.normalize(embed2, dim=-1)
            return F.cosine_similarity(embed1, embed2, dim=2)
        def euclidean_similarity(embed1, embed2):
            embed1 = F.normalize(embed1, dim=-1)
            embed2 = F.normalize(embed2, dim=-1)
            return torch.exp(-(embed1 - embed2).pow(2).sum(-1))
            
        mask_embeds = self.get_masks_embeds(model, batch)  # batch_size x num_masks x hidden_dim  
        label_centroids = centroids[label][None, :]        # 1 x num_masks x hidden_dim  
        if self.args.prototypical_similarity == "cos":
            similarity = cosine_similarity(label_centroids, mask_embeds) # batch_size x num_masks 
        elif self.args.prototypical_similarity == "euc":
            similarity = euclidean_similarity(label_centroids, mask_embeds) # batch_size x num_masks    
        aggregate = get_aggregation(self.args.eval_soft_pet_aggregation)
        prob = aggregate(similarity, dim=-1)
        if self.args.eval_soft_pet_aggregation in ["min", "max"]:
            prob = prob[0]
        return prob.cpu().detach().numpy().tolist()

    def get_masks_probs(self, model, batch, prev_mask_ids):
        assert batch["input_ids"].shape[0] == 1, "we only support batch size of 1 during eval."
        input_ids = trim_input_ids(
            batch["input_ids"],
            num_masks = self.args.num_extra_tokens,
            pad_token_id = self.model.config.pad_token_id,
            mask_token_id = self.model.config.mask_token_id    
        ) 
        masks_positions = [idx for idx, tok_id in enumerate(input_ids[0, :]) if
            tok_id == self.model.config.mask_token_id]
        inputs_embeds = self.model.roberta.embeddings.word_embeddings(input_ids)

        for i, id in enumerate(prev_mask_ids):
            inputs_embeds[0, masks_positions[i], :] =\
            self.model.roberta.embeddings.word_embeddings(torch.tensor([id]).cuda())

        outputs = model(input_ids=None, inputs_embeds=inputs_embeds)
        next_token_logits = torch.nn.Softmax(dim=2)(outputs[0])[0]
        # We find the next mask position.
        prob = next_token_logits[masks_positions[len(prev_mask_ids)]]
        return prob

    def _get_candidate_soft_log_probability_with_extra_tokens(self, model, batch, label, decoding_strategy="default"):
        """Computes the probability of the given candidate labels."""
        num_masks = self.model.num_masks
        assert batch["input_ids"].shape[0] == 1, "we only support batch size of 1 during eval."
        # removes the pad and keeps at most the num_mask tokens of masks.
        input_ids = trim_input_ids(
            batch["input_ids"],
            num_masks = num_masks,
            pad_token_id = self.model.config.pad_token_id,
            mask_token_id = self.model.config.mask_token_id    
        )
        masks_positions = [idx for idx, tok_id in enumerate(input_ids[0, :]) if
                           tok_id == self.model.config.mask_token_id]
        mask_labels = self.model.map_labels_to_mask_ids(torch.tensor([label]).cuda())
        mask_labels = mask_labels.detach().cpu().numpy().tolist()
        if not isinstance(mask_labels, list):
            mask_labels = [mask_labels]
        # first element is the index in the sequence, second is the position within all masks.
        masks_positions = [(mask_position, mask_label) for mask_position, mask_label in zip(masks_positions, mask_labels)]
        log_probabilities = []
        inputs_embeds = self.model.roberta.embeddings.word_embeddings(input_ids)
        for i in range(num_masks):
            outputs = model(input_ids=None, inputs_embeds=inputs_embeds)
            next_token_logits = torch.nn.Softmax(dim=2)(outputs[0])[0]
            if decoding_strategy == "parallel":
                for m_pos, m_id in masks_positions:
                    log_probabilities.append(math.log(next_token_logits[m_pos][m_id].item()))
                break
            mask_pos, masked_id = None, None
            max_prob = None
            for m_pos, m_id in masks_positions:
                m_prob = next_token_logits[m_pos][m_id].item()
                if max_prob is None or m_prob > max_prob:
                    max_prob = m_prob
                    mask_pos, masked_id = m_pos, m_id
            log_probabilities.append(math.log(max(max_prob, sys.float_info.min)))
            # put the mask position with maximum probability in its place.
            shift = 0 
            inputs_embeds[0, mask_pos, :] = self.model.extra_embeddings(torch.tensor([masked_id-shift]).cuda()) #self.model.soft_mask_labels[label][min_n, :]
            masks_positions.remove((mask_pos, masked_id))
        return sum(log_probabilities)

    def _get_candidate_log_probability(self, model, batch, 
        candidate_labels, decoding_strategy="default"):
        """Computes the probability of the given candidate labels."""  
        num_masks = sum(1 for token_id in candidate_labels if token_id != -100)
        # removes the pad and keeps at most the num_mask tokens of masks.
        input_ids = trim_input_ids(
            batch["input_ids"],
            num_masks = num_masks,
            pad_token_id = self.model.config.pad_token_id,
            mask_token_id = self.model.config.mask_token_id    
        )
        log_probabilities = []
        while True:
            masks = [(idx, tok_id) for idx, tok_id in enumerate(candidate_labels) if tok_id != -100]
            if not masks:  # there are no masks left to process, we are done
                break
            outputs = model(input_ids)
            next_token_logits = torch.nn.Softmax(dim=2)(outputs[0])[0]
            if decoding_strategy == "ltr":
                mask_pos, masked_id = masks[0]
                max_prob = next_token_logits[mask_pos][masked_id].item()
            elif decoding_strategy == "parallel":
                for m_pos, m_id in masks:
                    log_probabilities.append(math.log(next_token_logits[m_pos][m_id].item()))
                break
            else:
                mask_pos, masked_id = None, None
                max_prob = None
                for m_pos, m_id in masks:
                    m_prob = next_token_logits[m_pos][m_id].item()
                    if max_prob is None or m_prob > max_prob:
                        max_prob = m_prob
                        mask_pos, masked_id = m_pos, m_id
            log_probabilities.append(math.log(max_prob))
            # put the mask position with maximum probability in its place.
            input_ids[0][mask_pos] = masked_id
            candidate_labels[mask_pos] = -100
        return sum(log_probabilities)

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
           {
                    "params": [p for n, p in self.model.named_parameters() if PROMPT_EMBED in n],
                    "lr": self.args.prompt_learning_rate,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters()\
                               if n in decay_parameters and\
                                  PROMPT_EMBED not in n],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters()\
                               if n not in decay_parameters and\
                                PROMPT_EMBED not in n],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            optimizer_kwargs["lr"] = self.args.learning_rate
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            acc = metrics['eval_accuracy']
            with open(f'./{self.task_name}.txt', 'a') as f:
                # f.writelines(f"{data_args.task} {training_args.prompt_tune}")
                f.writelines(f'\tepoch:{epoch} {acc}\n')
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
    
    def train(
            self,
            resume_from_checkpoint: Optional[Union[str, bool]] = None,
            trial: Union["optuna.Trial", Dict[str, Any]] = None,
            ignore_keys_for_eval: Optional[List[str]] = None,
            **kwargs,
        ):
            """
            Main training entry point.

            Args:
                resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
                    If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
                    :class:`~transformers.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
                    `args.output_dir` as saved by a previous instance of :class:`~transformers.Trainer`. If present,
                    training will resume from the model/optimizer/scheduler states loaded here.
                trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                    The trial run or the hyperparameter dictionary for hyperparameter search.
                ignore_keys_for_eval (:obj:`List[str]`, `optional`)
                    A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                    gathering predictions for evaluation during the training.
                kwargs:
                    Additional keyword arguments used to hide deprecated arguments
            """
            resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

            # memory metrics - must set up as early as possible
            self._memory_tracker.start()

            args = self.args

            self.is_in_train = True

            # do_train is not a reliable argument, as it might not be set and .train() still called, so
            # the following is a workaround:
            if args.fp16_full_eval and not args.do_train:
                self._move_model_to_device(self.model, args.device)

            if "model_path" in kwargs:
                resume_from_checkpoint = kwargs.pop("model_path")
                warnings.warn(
                    "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                    "instead.",
                    FutureWarning,
                )
            if len(kwargs) > 0:
                raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
            # This might change the seed so needs to run first.
            self._hp_search_setup(trial)

            # Model re-init
            model_reloaded = False
            if self.model_init is not None:
                # Seed must be set before instantiating the model when using model_init.
                set_seed(args.seed)
                self.model = self.call_model_init(trial)
                model_reloaded = True
                # Reinitializes optimizer and scheduler
                self.optimizer, self.lr_scheduler = None, None

            # Load potential model checkpoint
            if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
                resume_from_checkpoint = get_last_checkpoint(args.output_dir)
                if resume_from_checkpoint is None:
                    raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

            if resume_from_checkpoint is not None:
                if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                    raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

                logger.info(f"Loading model from {resume_from_checkpoint}).")

                if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
                    config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
                    checkpoint_version = config.transformers_version
                    if checkpoint_version is not None and checkpoint_version != __version__:
                        logger.warn(
                            f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                            f"Transformers but your current version is {__version__}. This is not recommended and could "
                            "yield to errors or unwanted behaviors."
                        )

                if args.deepspeed:
                    # will be resumed in deepspeed_init
                    pass
                else:
                    # We load the model state dict on the CPU to avoid an OOM error.
                    state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
                    # If the model is on the GPU, it still works!
                    self._load_state_dict_in_model(state_dict)

                    # release memory
                    del state_dict

            # If model was re-initialized, put it on the right device and update self.model_wrapped
            if model_reloaded:
                if self.place_model_on_device:
                    self._move_model_to_device(self.model, args.device)
                self.model_wrapped = self.model

            # Keeping track whether we can can len() on the dataset or not
            train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

            # Data loader and number of training steps
            train_dataloader = self.get_train_dataloader()

            # Setting up training control variables:
            # number of training epochs: num_train_epochs
            # number of training steps per epoch: num_update_steps_per_epoch
            # total number of training steps to execute: max_steps
            total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
            if train_dataset_is_sized:
                num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
                num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
                if args.max_steps > 0:
                    max_steps = args.max_steps
                    num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                        args.max_steps % num_update_steps_per_epoch > 0
                    )
                    # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
                    # the best we can do.
                    num_train_samples = args.max_steps * total_train_batch_size
                else:
                    max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                    num_train_epochs = math.ceil(args.num_train_epochs)
                    num_train_samples = len(self.train_dataset) * args.num_train_epochs
            else:
                # see __init__. max_steps is set when the dataset has no __len__
                max_steps = args.max_steps
                # Setting a very large number of epochs so we go as many times as necessary over the iterator.
                num_train_epochs = sys.maxsize
                num_update_steps_per_epoch = max_steps
                num_train_samples = args.max_steps * total_train_batch_size

            if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
                if self.args.n_gpu > 1:
                    # nn.DataParallel(model) replicates the model, creating new variables and module
                    # references registered here no longer work on other gpus, breaking the module
                    raise ValueError(
                        "Currently --debug underflow_overflow is not supported under DP. Please use DDP (torch.distributed.launch)."
                    )
                else:
                    debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

            delay_optimizer_creation = self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
            if args.deepspeed:
                deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                    self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
                )
                self.model = deepspeed_engine.module
                self.model_wrapped = deepspeed_engine
                self.deepspeed = deepspeed_engine
                self.optimizer = optimizer
                self.lr_scheduler = lr_scheduler
            elif not delay_optimizer_creation:
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)

            self.state = TrainerState()
            self.state.is_hyper_param_search = trial is not None

            model = self._wrap_model(self.model_wrapped)

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            if delay_optimizer_creation:
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)

            # Check if saved optimizer or scheduler states exist
            self._load_optimizer_and_scheduler(resume_from_checkpoint)

            # important: at this point:
            # self.model         is the Transformers Model
            # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

            # Train!
            num_examples = (
                self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
            )

            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {num_examples}")
            logger.info(f"  Num Epochs = {num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {max_steps}")

            self.state.epoch = 0
            start_time = time.time()
            epochs_trained = 0
            steps_trained_in_current_epoch = 0
            steps_trained_progress_bar = None

            # Check if continuing training from a checkpoint
            if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, "trainer_state.json")
            ):
                self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, "trainer_state.json"))
                epochs_trained = self.state.global_step // num_update_steps_per_epoch
                if not args.ignore_data_skip:
                    steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                    steps_trained_in_current_epoch *= args.gradient_accumulation_steps
                else:
                    steps_trained_in_current_epoch = 0

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info(f"  Continuing training from epoch {epochs_trained}")
                logger.info(f"  Continuing training from global step {self.state.global_step}")
                if not args.ignore_data_skip:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                        "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                        "flag to your launch command, but you will resume the training on data already seen by your model."
                    )
                    if self.is_local_process_zero() and not args.disable_tqdm:
                        steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                        steps_trained_progress_bar.set_description("Skipping the first batches")

            # Update the references
            self.callback_handler.model = self.model
            self.callback_handler.optimizer = self.optimizer
            self.callback_handler.lr_scheduler = self.lr_scheduler
            self.callback_handler.train_dataloader = train_dataloader
            self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
            self.state.trial_params = hp_params(trial) if trial is not None else None
            # This should be the same if the state has been saved but in case the training arguments changed, it's safer
            # to set this after the load.
            self.state.max_steps = max_steps
            self.state.num_train_epochs = num_train_epochs
            self.state.is_local_process_zero = self.is_local_process_zero()
            self.state.is_world_process_zero = self.is_world_process_zero()

            # tr_loss is a tensor to avoid synchronization of TPUs through .item()
            all_hidden_states = None
            tr_loss = torch.tensor(0.0).to(args.device)
            # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
            self._total_loss_scalar = 0.0
            self._globalstep_last_logged = self.state.global_step
            model.zero_grad()

            self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

            # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
            if not args.ignore_data_skip:
                for epoch in range(epochs_trained):
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break

            for epoch in range(epochs_trained, num_train_epochs):
                if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                    train_dataloader.sampler.set_epoch(epoch)
                elif isinstance(train_dataloader.dataset, IterableDatasetShard):
                    train_dataloader.dataset.set_epoch(epoch)

                if is_torch_tpu_available():
                    parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                    epoch_iterator = parallel_loader
                else:
                    epoch_iterator = train_dataloader

                # Reset the past mems state at the beginning of each epoch if necessary.
                if args.past_index >= 0:
                    self._past = None

                steps_in_epoch = (
                    len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
                )
                self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

                for step, inputs in enumerate(epoch_iterator):

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    if (
                        ((step + 1) % args.gradient_accumulation_steps != 0)
                        and args.local_rank != -1
                        and args._no_sync_in_gradient_accumulation
                    ):
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with model.no_sync():
                            cur_tr_loss, cur_hidden_states =  self.training_step(model, inputs)
                            tr_loss += cur_tr_loss
                            if all_hidden_states is None:
                                all_hidden_states = cur_hidden_states
                            else:
                                all_hidden_states += cur_hidden_states
                            
                    else:
                        cur_tr_loss, cur_hidden_states =  self.training_step(model, inputs)
                        tr_loss += cur_tr_loss
                        if all_hidden_states is None:
                            all_hidden_states = cur_hidden_states
                        else:
                            all_hidden_states += cur_hidden_states
                    
                    self.current_flos += float(self.floating_point_ops(inputs))

                    # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                    if self.deepspeed:
                        self.deepspeed.step()

                    if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                    ):
                        # meta-gradient regularization
                        if self.gradient_transformer is not None:
                            with torch.no_grad():
                                for n, p in self.model.named_parameters():
                                    if PROMPT_EMBED in n and p.requires_grad:
                                        grad, _ = self.gradient_transformer(p.grad, all_hidden_states)
                                        p.grad = grad
                        all_hidden_states = None
                        
                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                            # deepspeed does its own clipping

                            if self.use_amp:
                                # AMP: gradients need unscaling
                                self.scaler.unscale_(self.optimizer)

                            if hasattr(self.optimizer, "clip_grad_norm"):
                                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                self.optimizer.clip_grad_norm(args.max_grad_norm)
                            elif hasattr(model, "clip_grad_norm_"):
                                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                                model.clip_grad_norm_(args.max_grad_norm)
                            else:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                    args.max_grad_norm,
                                )

                        # Optimizer step
                        optimizer_was_run = True
                        if self.deepspeed:
                            pass  # called outside the loop
                        elif is_torch_tpu_available():
                            xm.optimizer_step(self.optimizer)
                        elif self.use_amp:
                            scale_before = self.scaler.get_scale()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            scale_after = self.scaler.get_scale()
                            optimizer_was_run = scale_before <= scale_after
                        else:
                            self.optimizer.step()

                        if optimizer_was_run and not self.deepspeed:
                            self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                        self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        break

                self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
                self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

                if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                    if is_torch_tpu_available():
                        # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                        xm.master_print(met.metrics_report())
                    else:
                        logger.warning(
                            "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                            "configured. Check your training configuration if this is unexpected."
                        )
                if self.control.should_training_stop:
                    break

            if args.past_index and hasattr(self, "_past"):
                # Clean the state at the end of training
                delattr(self, "_past")

            logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
            if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
                # Wait for everyone to get here so we are sur the model has been saved by process 0.
                if is_torch_tpu_available():
                    xm.rendezvous("load_best_model_at_end")
                elif args.local_rank != -1:
                    dist.barrier()

                logger.info(
                    f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
                )

                best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
                if os.path.exists(best_model_path):
                    # We load the model state dict on the CPU to avoid an OOM error.
                    state_dict = torch.load(best_model_path, map_location="cpu")
                    # If the model is on the GPU, it still works!
                    self._load_state_dict_in_model(state_dict)
                else:
                    logger.warn(
                        f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                        "on multiple nodes, you should activate `--save_on_each_node`."
                    )

                if self.deepspeed:
                    self.deepspeed.load_checkpoint(
                        self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                    )

            # add remaining tr_loss
            self._total_loss_scalar += tr_loss.item()
            train_loss = self._total_loss_scalar / self.state.global_step

            metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
            metrics["train_loss"] = train_loss

            self.is_in_train = False

            self._memory_tracker.stop_and_update_metrics(metrics)

            self.log(metrics)

            self.control = self.callback_handler.on_train_end(args, self.state, self.control)

            return TrainOutput(self.state.global_step, train_loss, metrics)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.use_amp else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss, all_hidden_states = self.compute_loss(model, inputs, return_hidden=True)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps
            all_hidden_states = all_hidden_states / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach(), all_hidden_states

    def compute_loss(self, model, inputs, return_outputs=False, return_hidden=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
        if return_hidden==True:
            hidden_states = outputs.encoder_last_hidden_state.detach()
            all_hidden_states = torch.mean(torch.mean(hidden_states, 0), 0)
            return (loss, all_hidden_states)
        return (loss, outputs) if return_outputs else loss