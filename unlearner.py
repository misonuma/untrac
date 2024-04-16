# +
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union, Dict, Any, Tuple
import pdb
import random
import math
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.sampler import Sampler
import datasets
from datasets import Dataset
from transformers import is_datasets_available
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import logger
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments

from transformers.trainer_callback import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    has_length,
    is_torch_tpu_available,
    denumpify_detensorize,
    ShardedDDPOption,
    seed_worker,
)
from transformers.training_args import TrainingArguments
from transformers.trainer_pt_utils import (
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    find_batch_size,
    get_parameter_names,
)
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers import Trainer
# from transformers.trainer import _is_torch_generator_available
from transformers.optimization import Adafactor, get_scheduler
from optim import get_constant_schedule_with_zero


# -

class Unlearner(Seq2SeqTrainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.autocast_smart_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                        
        if self.args.unlearn:
            loss = -loss
                                
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            if self.args.debug_grad_loss:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            
        if self.args.debug_grad_loss:
            self.update_history_grad_loss(loss_train, loss_retain)
            
        if self.args.debug_grad_model:
            self.update_history_grad_model()

        return loss.detach()
    
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        inputs = super()._prepare_inputs(inputs)
        inputs = {key: value for key, value in inputs.items() if key in ["input_ids", "attention_mask", "decoder_input_ids", "labels"]}
        return inputs
    
    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in opt_model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Unlearner.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
    
    @staticmethod
    def get_optimizer_cls_and_kwargs(args: TrainingArguments) -> Tuple[Any, Any]:
        optimizer_kwargs = {"lr": args.learning_rate}
        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }
        sgd_kwargs = {
            "momentum": args.momentum,
            "dampening": args.momentum,
        }
        rmsprop_kwargs = {
            "alpha": args.alpha,
        }
        if args.rmsprop:
            optimizer_cls = torch.optim.RMSprop
            optimizer_kwargs.update(rmsprop_kwargs)
        elif args.optim == OptimizerNames.ADAFACTOR:
            optimizer_cls = Adafactor
            optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        elif args.optim == OptimizerNames.ADAMW_HF:
            from .optimization import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim == OptimizerNames.ADAMW_TORCH:
            from torch.optim import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim == OptimizerNames.ADAMW_TORCH_XLA:
            try:
                from torch_xla.amp.syncfree import AdamW

                optimizer_cls = AdamW
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer failed to import syncfree AdamW from torch_xla.")
        elif args.optim == OptimizerNames.ADAMW_APEX_FUSED:
            try:
                from apex.optimizers import FusedAdam

                optimizer_cls = FusedAdam
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate apex FusedAdam but apex is not installed!")
        elif args.optim == OptimizerNames.ADAMW_BNB:
            try:
                from bitsandbytes.optim import Adam8bit

                optimizer_cls = Adam8bit
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate bnb Adam8bit but bnb is not installed!")
        elif args.optim == OptimizerNames.SGD:
            optimizer_cls = torch.optim.SGD
            optimizer_kwargs.update(sgd_kwargs)
        elif args.optim == OptimizerNames.ADAGRAD:
            optimizer_cls = torch.optim.Adagrad
        else:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")
        return optimizer_cls, optimizer_kwargs
    
    # rewrite the evaluation loop, with customized call to compute_metrics
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        
        args = self.args
        
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.
        
        all_datasets = None

        observed_num_examples = 0
        # Main evaluation loop
        
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            if "dataset" in inputs: 
                all_datasets = all_datasets + inputs["dataset"] if all_datasets is not None else inputs["dataset"]
            if "answers" in inputs:
                labels = inputs["answers"]
                logits = logits[:, inputs["input_ids"].shape[-1]:]
                
            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=self.tokenizer.pad_token_id)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=self.tokenizer.pad_token_id)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=self.tokenizer.pad_token_id)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=self.tokenizer.pad_token_id)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=self.tokenizer.pad_token_id)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=self.tokenizer.pad_token_id)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        if all_datasets is not None:
            all_datasets = all_datasets[:num_samples]
            
        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(dataset=eval_dataset, preds=all_preds, labels=all_labels, save_prefix=metric_key_prefix)
        else:
            metrics = {}

        if self.lr_scheduler is not None:
            metrics["lr"] = self._get_learning_rate()
            
        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
            for dataset in np.unique(all_datasets):
                losses = all_losses[np.where(np.array(all_datasets) == dataset)]
                metrics[f"{metric_key_prefix}_loss_{dataset}"] = losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            raise NotImplementedError
        
        eval_batch_sampler = TaskGroupedBatchSampler(batch_size=self.args.per_device_eval_batch_size, dataset=eval_dataset)    
        dataloader = DataLoader(
            eval_dataset,
            batch_sampler=eval_batch_sampler, 
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        
        return dataloader
    
    def get_dataloader(self, dataset: Optional[Dataset] = None) -> DataLoader:
        data_collator = self.data_collator
        
        if isinstance(dataset, torch.utils.data.IterableDataset):
            raise NotImplementedError
            
        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
        
        generator = torch.Generator()
        generator.manual_seed(self.args.seed)
        sampler = RandomSampler(dataset, generator=None)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )
        
        return dataloader
    
    def compute_losses(self, inputs):
        outputs = self.model(**inputs)
        
        logits = outputs.logits
        labels = inputs["labels"]
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction="none")

        losses_flat = loss_fct(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))
        losses_seq = losses_flat.view(shift_labels.shape)
        mask_labels = shift_labels != self.tokenizer.pad_token_id
        losses = torch.sum(losses_seq * mask_labels, -1) / torch.sum(mask_labels, -1)

        return losses
    
    def update_history_grad_loss(self, loss_train, loss_retain):
        grads_train = torch.autograd.grad(loss_train, self.model.parameters())
        grads_train = torch.concat([grad.flatten() for grad in grads_train])
        grads_train = float(torch.norm(grads_train))
        
        grads_retain = torch.autograd.grad(loss_retain, self.model.parameters())
        grads_retain = torch.concat([grad.flatten() for grad in grads_retain])
        grads_retain = float(torch.norm(grads_retain))

        name_norm = {
            "train": grads_train,
            "retain": grads_retain,
        }
        
        self.history_grad_loss[self.state.global_step] = name_norm
    
    def update_history_grad_model(self):
        grads_all = [param.grad.detach() for _, param in self.model.named_parameters()]
        grads_encoder = [param.grad.detach() for name, param in self.model.named_parameters() if name.startswith("encoder")]
        grads_decoder = [param.grad.detach() for name, param in self.model.named_parameters() if name.startswith("decoder")]

        name_grads = {
            "all": torch.concat([grad.flatten() for grad in grads_all]),
#             "input_emb": self.model.shared.weight.grad.flatten().detach(),
#             "encoder": torch.concat([grad.flatten() for grad in grads_encoder]),
#             "decoder": torch.concat([grad.flatten() for grad in grads_decoder]),
#             "output_emb": self.model.lm_head.weight.grad.flatten().detach(),
        }

        name_norm = {
            name: float(torch.norm(grads) / len(grads))
            for name, grads in name_grads.items()
        }

        self.history_grad_model[self.state.global_step] = name_norm
        
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.args.zero_steps is not None and self.state.global_step < self.args.zero_steps: return
        
        super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)


class DenserEvalCallback(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        log_eval_steps = [1, 50, 100, 200]

        # Log
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_evaluate = True

        return control


class TaskGroupedBatchSampler(Sampler):
    def __init__(
        self,
        batch_size: int,
        dataset: Optional[Dataset] = None,
        shuffle = False,
        generator = None,
    ):
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")

        self.batch_size = batch_size

        self.generator = generator
        
        dataset = dataset.flatten_indices()
        dataset = dataset.add_column("indices", np.arange(len(dataset)))
        mega_batches = [dataset.filter(lambda example: example["dataset"] == dataset_name)["indices"] for dataset_name in dataset.unique("dataset")]
        self.batches = []
        for mega_batch in mega_batches:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in mega_batch:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    self.batches.append(batch)
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                self.batches.append(batch[:idx_in_batch])
        
        if shuffle: random.shuffle(self.batches)
        
    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch
