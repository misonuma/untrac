# +
# #!/usr/bin/env python
# coding=utf-8
import logging
import os
import sys

import datasets
from datasets import load_from_disk, concatenate_datasets
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from t5_collator import DataCollatorForT5
from opt_collator import DataCollatorForOPT
from unlearner import Unlearner, DenserEvalCallback
from arguments import DataTrainingArguments, ModelArguments, TrainingArguments, args_to_output_dir

PROXIES = {
    'http': os.environ.get("PROXY"),
    'https': os.environ.get("PROXY"),
}
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = "model"

logger = logging.getLogger(__name__)


# -

def configure(argv=None):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if argv is not None:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=argv.split())
    else:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        else:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        argv = " ".join(sys.argv[1:])
            
    if "t5" in model_args.model_name_or_path:
        model_args.model_type = "t5"
    elif "opt" in model_args.model_name_or_path:
        model_args.model_type = "opt"
        
    if training_args.output_dir is None:
        model_dir = os.path.join(CUR_DIR, MODEL_DIR, model_args.model_type)
        arg_dir = args_to_output_dir(argv)
        training_args.output_dir = os.path.join(model_dir, arg_dir)
    else:
        training_args.output_dir = os.path.join(CUR_DIR, training_args.output_dir)
    print(f"output_dir: {training_args.output_dir}")
    
    if training_args.metric is not None: training_args.predict_with_generate = True
        
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

    # Set seed before initializing model
    set_seed(training_args.seed)
    
    if model_args.model_type == "opt":
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        )
        
        if training_args.scratch:
            config = AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                use_auth_token=True if model_args.use_auth_token else None,
                proxies=PROXIES,
            )
            model = AutoModelForCausalLM.from_config(
                config=config,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                use_auth_token=True if model_args.use_auth_token else None,
                proxies=PROXIES,
            )
        
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
    elif model_args.model_type == "t5":
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            proxies=PROXIES,
        )

        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            proxies=PROXIES,
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            proxies=PROXIES,
        )
        model.resize_token_embeddings(len(tokenizer))
        
    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )
    
    generator = torch.Generator()
    generator.manual_seed(training_args.seed)
    
    if training_args.do_train:
        train_dataset = concatenate_datasets([load_from_disk(os.path.join(CUR_DIR, train_dir_item)) for train_dir_item in data_args.train_dir])
        
        if data_args.dataset_names is not None and all([dataset_name != "all" for dataset_name in data_args.dataset_names]):
            train_dataset = train_dataset.filter(lambda x: x["dataset"] in data_args.dataset_names, num_proc=8)
            assert set(data_args.dataset_names) == set(train_dataset.unique("dataset"))
            
        if data_args.loo_dataset_names is not None:
            train_dataset = train_dataset.filter(lambda x: x["dataset"] not in data_args.loo_dataset_names, num_proc=8)
            assert set(data_args.loo_dataset_names) & set(train_dataset.unique("dataset")) == set()
            
        if data_args.max_train_samples is not None:
            sampler = RandomSampler(train_dataset, num_samples=data_args.max_train_samples, generator=generator)
            train_dataset = train_dataset.select(sampler)
        
        data_args.datasets = list(train_dataset.unique("dataset"))
                    
    if training_args.do_eval:
        eval_dataset = concatenate_datasets([load_from_disk(os.path.join(CUR_DIR, eval_dir_item)) for eval_dir_item in data_args.eval_dir])
        
        if data_args.each_eval_samples is not None:
            eval_subsets = []
            for eval_subset_name in eval_dataset.unique("dataset"):
                eval_subset = eval_dataset.filter(lambda x: x["dataset"] == eval_subset_name, num_proc=8)
                sampler = RandomSampler(eval_subset, num_samples=data_args.each_eval_samples, generator=generator)
                eval_subset = eval_subset.select(sampler)
                eval_subsets.append(eval_subset)
            eval_dataset = concatenate_datasets(eval_subsets)

        if data_args.max_eval_samples is not None:
            sampler = RandomSampler(eval_dataset, num_samples=data_args.max_eval_samples, generator=generator)
            eval_dataset = eval_dataset.select(sampler)
                
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if model_args.model_type == "opt":
        data_collator = DataCollatorForOPT(
            tokenizer=tokenizer,
            model=model,
            data_args=data_args,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    elif model_args.model_type == "t5":
        data_collator = DataCollatorForT5(
            tokenizer=tokenizer,
            model=model,
            data_args=data_args,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    # we don't want to remove unused columns because we will prepare each batch during training, 
    # and some of the information will aslo be used in evaluation.
    training_args.remove_unused_columns = False 
    
    # Initialize our Trainer
    trainer = Unlearner(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None,
    )
    
    return trainer, data_args


def run(trainer, data_args):
    all_metrics = {"run_name": trainer.args.run_name}
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(trainer.args.output_dir) and trainer.args.do_train and not trainer.args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(trainer.args.output_dir)
        if last_checkpoint is None and len(os.listdir(trainer.args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({trainer.args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and trainer.args.resume_from_checkpoint is None:
            logger.warning(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            
    # Training
    if trainer.args.do_train:
        checkpoint = None
        if trainer.args.resume_from_checkpoint is not None:
            checkpoint = trainer.args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(trainer.train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(trainer.train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        all_metrics.update(metrics)
        
    # Evaluation
    if trainer.args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(trainer.eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(trainer.eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        all_metrics.update(metrics)
        
    # Predit
    if trainer.args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            trainer.eval_dataset, metric_key_prefix="predict",
            max_new_tokens = trainer.args.max_new_tokens,
            eos_token_id = trainer.args.eos_token_id,
        )
        return predict_results
        
    return all_metrics


if __name__ == "__main__":
    trainer, data_args = configure()
    all_metrics = run(trainer, data_args)
