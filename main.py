# +
# #!/usr/bin/env python
# coding=utf-8
import logging
import os
import sys
import json
import random
import pdb
from collections import defaultdict
import copy
import pandas as pd
import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric, load_from_disk, concatenate_datasets
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler
from tqdm import tqdm
import subprocess

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    default_data_collator,
    HfArgumentParser,
    BitsAndBytesConfig,
    set_seed,
    LlamaTokenizer, 
    LlamaForCausalLM,
)
from transformers.file_utils import is_offline_mode, is_in_notebook
from transformers.trainer_utils import get_last_checkpoint
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

from t5_collator import DataCollatorForT5
from causal_collator import DataCollatorForCausalLM
from unlearner import Unlearner, DenserEvalCallback
from arguments import DataTrainingArguments, ModelArguments, TrainingArguments, args_to_output_dir
from utils_influence import lissa, arnoldi_iter, arnoldi_distill, dot_product_fn, to_gpu, to_cpu
from compute_metrics import (
    compute_txt_metrics,
    compute_grouped_metrics, 
    metric_max_over_ground_truths,
    exact_match_score,
    rouge1_score,
    rougeL_score,
)

PROXIES = {
    'http': os.environ.get("PROXY"),
    'https': os.environ.get("PROXY"),
}
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = "model"
DATA_DIR = "data"

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


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
        
    # Load pretrained model and tokenizer
    if model_args.model_name_or_path is None:
        if "llama" in model_args.model:
            model_args.model_name_or_path = f"openlm-research/{model_args.model}"
        elif "opt" in model_args.model:
            model_args.model_name_or_path = f"facebook/{model_args.model}"
        else:
            model_args.model_name_or_path = f"google/t5-{model_args.model}-lm-adapt"
    else:
        model_args.model_name_or_path = os.path.join(CUR_DIR, model_args.model_name_or_path)
            
    if "llama" in model_args.model_name_or_path:
        model_args.model_type = "llama"
    elif "t5" in model_args.model_name_or_path:
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
    
    if model_args.model_type == "llama":
        if training_args.load_in_8bit and training_args.load_in_4bit:
            raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
        elif training_args.load_in_8bit or training_args.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=training_args.load_in_8bit, load_in_4bit=training_args.load_in_4bit
            )
            # This means: fit the entire model on the GPU:0
            device_map = {"": 0}
            torch_dtype = torch.bfloat16
        else:
            device_map = None
            quantization_config = None
            torch_dtype = (
                model_args.torch_dtype
                if model_args.torch_dtype in ["auto", None]
                else getattr(torch, model_args.torch_dtype)
            )
            
        tokenizer = LlamaTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        
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

        model = T5ForConditionalGeneration.from_pretrained(
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
        train_dataset = [load_from_disk(os.path.join(CUR_DIR, DATA_DIR, train_dir_item)) for train_dir_item in data_args.train_dir]
        if len(train_dataset) == 1: train_dataset = train_dataset[0]
        
        if data_args.each_train_samples is not None:
            train_subsets = []
            for train_subset in train_dataset:
                sampler = RandomSampler(train_subset, num_samples=data_args.each_train_samples, generator=generator)
                train_subset = train_subset.select(sampler)
                train_subsets.append(train_subset)
            train_dataset = train_subsets
        
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
                    
    if training_args.do_eval or training_args.do_predict:
        eval_dataset = [load_from_disk(os.path.join(CUR_DIR, DATA_DIR, eval_dir_item)) for eval_dir_item in data_args.eval_dir]
                
        if data_args.each_eval_samples is not None:
            eval_subsets = []
            for eval_subset in eval_dataset:
                sampler = RandomSampler(eval_subset, num_samples=data_args.each_eval_samples, generator=generator)
                eval_subset = eval_subset.select(sampler)
                eval_subsets.append(eval_subset)
            eval_dataset = eval_subsets
        
        if data_args.eval_dataset_names is not None:
            eval_dataset = eval_dataset.filter(lambda x: x["dataset"] in data_args.eval_dataset_names, num_proc=8)
            assert set(data_args.eval_dataset_names) == set(eval_dataset.unique("dataset"))

        if data_args.max_eval_samples is not None:
            sampler = RandomSampler(eval_dataset, num_samples=data_args.max_eval_samples, generator=generator)
            eval_dataset = eval_dataset.select(sampler)
            
        if training_args.do_predict:
            eval_dataset = eval_dataset.rename_columns({"targets_pretokenized": "preds_pretokenized"})
            if data_args.max_predict_samples is not None:
                eval_dataset = eval_dataset.select(range(data_args.max_predict_samples))

    if not (training_args.graddot or training_args.gradcos or training_args.tracin or training_args.traccos or training_args.influence or training_args.arnoldi or training_args.bm25): 
        if training_args.do_train:
            train_dataset = train_dataset.remove_columns(["dataset"])
            
        if training_args.do_eval or training_args.do_predict:
            assert len(eval_dataset) == 1
            eval_dataset = eval_dataset[0]
                
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if model_args.model_type == "llama" or model_args.model_type == "opt":
        data_collator = DataCollatorForCausalLM(
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

    # Metric
    def compute_metrics(dataset, preds, labels, group_metrics=True, save_prefix=None):
        preds = np.ma.array(preds, mask=(preds == label_pad_token_id)).filled(fill_value=tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.ma.array(labels, mask=(labels == label_pad_token_id)).filled(fill_value=tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        labels = [[label] for label in labels]
            
        result = compute_txt_metrics(predictions=preds, references=labels, metrics=training_args.metric)
        if group_metrics:
            result_per_task = compute_grouped_metrics(preds=preds, labels=labels, groups=dataset["dataset"], compute_metrics=compute_txt_metrics, metrics=training_args.metric)
            result.update(result_per_task)
    #     categories = ["_".join(it[0].lower().split()) for it in dataset["Categories"]]
    #     result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=categories, compute_metrics=compute_txt_metrics, metrics=training_args.metric)
    #     result.update(result_per_category)

        result = {k: round(v, 4) for k, v in result.items()}
        return result
    
    # Initialize our Trainer
    trainer = Unlearner(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval or training_args.do_predict else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None,
    )
    
    if training_args.debug_grad_loss:
        trainer.history_grad_loss = {}
        
    if training_args.debug_grad_model:
        trainer.history_grad_model = {}
        
    return trainer, data_args


def run(trainer, data_args):
    if trainer.args.tracin:
        return tracin(trainer, data_args, normalize=True)
    elif trainer.args.graddot:
        return graddot(trainer, data_args, gradcos=False)
    elif trainer.args.gradcos:
        return graddot(trainer, data_args, gradcos=True)
    elif trainer.args.influence:
        return influence(trainer, data_args, normalize=True)
    elif trainer.args.arnoldi:
        return arnoldi(trainer, data_args, normalize=True)
    
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


def compute_grads(trainer, dataloader, max_steps=-1):
    data_grads = None
    data_size = 0

    for i, inputs in enumerate(dataloader):
        if max_steps > 0 and i == max_steps: break

        inputs = trainer._prepare_inputs(inputs)
        batch_size = len(inputs["input_ids"])

        loss = trainer.compute_loss(trainer.model, inputs)
        grads = torch.autograd.grad(loss, trainer.model.parameters())
        
        if data_grads is None:
            data_grads = tuple(grad*batch_size for grad in grads)
        else:
            data_grads = tuple(data_grad + grad*batch_size for data_grad, grad in zip(data_grads, grads))

        data_size += batch_size
    
    if max_steps > 0: assert data_size == max_steps * dataloader.batch_size
    return data_grads


def graddot(trainer, data_args, gradcos=False):
    trainer.model.eval()
    
    trainer.train_dataloader = iter(trainer.get_train_dataloader())

    dict_graddots = {}
    for dataset_name in data_args.eval_dir:
        output_dir = trainer.args.output_dir.replace("=".join(data_args.eval_dir), dataset_name)
        subprocess.call(f"mkdir {output_dir}".split())
        if gradcos:
            path_graddot = os.path.join(output_dir, "df_gradcos.pkl")
        else:
            path_graddot = os.path.join(output_dir, "df_graddot.pkl")
        print(f"path_graddot: {path_graddot}")
        dict_graddots[path_graddot] = defaultdict(dict)
    
    train_dataset_names = trainer.train_dataset.unique("dataset")
    for train_dataset_name in tqdm(train_dataset_names):
        train_subset = trainer.train_dataset.filter(lambda ex: ex["dataset"] == train_dataset_name)
        subset_dataloader = trainer.get_dataloader(train_subset)
        train_grads = compute_grads(trainer, subset_dataloader, max_steps=trainer.args.max_steps)
        
        if gradcos:
            grad_norm = sum([torch.sum(grad**2) for grad in train_grads])**(1/2)
            train_grads = tuple(grad/grad_norm for grad in train_grads)

        for eval_dataset, path_graddot in zip(trainer.eval_dataset, dict_graddots):
            eval_dataset_names = eval_dataset.unique("dataset")
            
            for eval_dataset_name in eval_dataset_names:
                eval_subset = eval_dataset.filter(lambda ex: ex["dataset"] == eval_dataset_name)
                eval_dataloader = trainer.get_eval_dataloader(eval_subset)
                eval_grads = compute_grads(trainer, eval_dataloader)
                
                graddot = float(sum([torch.sum(eval_grad*train_grad) for eval_grad, train_grad in zip(eval_grads, train_grads)]))
                dict_graddots[path_graddot][eval_dataset_name][train_dataset_name] = graddot

    df_graddots = []
    for path_graddot, dict_graddot in dict_graddots.items():
        df_graddot = pd.DataFrame(dict_graddot).sort_index(axis=1)
        df_graddots.append(df_graddot)
        df_graddot.to_pickle(path_graddot)
        
    return df_graddots


def tracin(trainer, data_args, normalize=True):
    path_tracin = os.path.join(trainer.args.output_dir, "df_tracin.pkl")
    print(f"path_tracin: {path_tracin}")

    model_name_or_path = trainer.model.config._name_or_path
    Model = trainer.model.__class__
    device = trainer.model.device

    ckpts = [ckpt for ckpt in os.listdir(model_name_or_path) if ckpt.startswith("checkpoint")]
    if trainer.args.max_ckpt is not None: ckpts = [ckpt for ckpt in ckpts if int(ckpt.replace("checkpoint-", "")) <= trainer.args.max_ckpt]
    if trainer.args.step_ckpt is not None: ckpts = [ckpt for ckpt in ckpts if int(ckpt.replace("checkpoint-", "")) % trainer.args.step_ckpt == 0]

    dict_tracins = {}
    for dataset_name in data_args.eval_dir:
        output_dir = trainer.args.output_dir.replace("=".join(data_args.eval_dir), dataset_name)
        subprocess.call(f"mkdir {output_dir}".split())
        path_tracin = os.path.join(output_dir, "df_tracin.pkl")
        print(f"path_tracin: {path_tracin}")
        dict_tracins[path_tracin] = defaultdict(dict)
    
    for ckpt in tqdm(ckpts):
        trainer.model = Model.from_pretrained(os.path.join(model_name_or_path, ckpt),).to(device)
        trainer.model.eval()
    
        train_dataset_names = trainer.train_dataset.unique("dataset")
        for train_dataset_name in tqdm(train_dataset_names):
            train_subset = trainer.train_dataset.filter(lambda ex: ex["dataset"] == train_dataset_name)
            subset_dataloader = trainer.get_dataloader(train_subset)
            train_grads = compute_grads(trainer, subset_dataloader, max_steps=trainer.args.max_steps)

            if normalize:
                grad_norm = sum([torch.sum(grad**2) for grad in train_grads])**(1/2)
                train_grads = tuple(grad/grad_norm for grad in train_grads)

            for eval_dataset, path_tracin in zip(trainer.eval_dataset, dict_tracins):
                eval_dataset_names = eval_dataset.unique("dataset")

                for eval_dataset_name in eval_dataset_names:
                    eval_subset = eval_dataset.filter(lambda ex: ex["dataset"] == eval_dataset_name)
                    eval_dataloader = trainer.get_eval_dataloader(eval_subset)
                    eval_grads = compute_grads(trainer, eval_dataloader)

                    tracin = float(sum([torch.sum(eval_grad*train_grad) for eval_grad, train_grad in zip(eval_grads, train_grads)]))
                    dict_tracins[path_tracin][eval_dataset_name][train_dataset_name] = dict_tracins[path_tracin][eval_dataset_name].get(train_dataset_name, 0) + tracin

    df_tracins = []
    for path_tracin, dict_tracin in dict_tracins.items():
        df_tracin = pd.DataFrame(dict_tracin).sort_index(axis=1)
        df_tracins.append(df_tracin)
        df_tracin.to_pickle(path_tracin)
        
    return df_tracins


def influence(trainer, data_args, normalize=True, debug=False):
    trainer.model.eval()
    
    trainer.train_dataloader = iter(trainer.get_train_dataloader())

    dict_influences = {}
    for dataset_name in data_args.eval_dir:
        output_dir = trainer.args.output_dir.replace("=".join(data_args.eval_dir), dataset_name)
        path_influence = os.path.join(output_dir, "df_influence.pkl")
        subprocess.call(f"mkdir {output_dir}".split())
        print(f"path_influence: {path_influence}")
        dict_influences[path_influence] = defaultdict(dict)
    
    train_dataset_names = trainer.train_dataset.unique("dataset")
    for train_dataset_name in tqdm(train_dataset_names):
        train_subset = trainer.train_dataset.filter(lambda ex: ex["dataset"] == train_dataset_name)
        subset_dataloader = trainer.get_dataloader(train_subset)
        train_grads = compute_grads(trainer, subset_dataloader, max_steps=trainer.args.max_steps)
        train_grads = lissa(trainer, vectors=train_grads, debug=debug)
        
        if normalize:
            grad_norm = sum([torch.sum(grad**2) for grad in train_grads])**(1/2)
            train_grads = tuple(grad/grad_norm for grad in train_grads)

        for eval_dataset, path_influence in zip(trainer.eval_dataset, dict_influences):
            eval_dataset_names = eval_dataset.unique("dataset")
            
            for eval_dataset_name in eval_dataset_names:
                eval_subset = eval_dataset.filter(lambda ex: ex["dataset"] == eval_dataset_name)
                eval_dataloader = trainer.get_eval_dataloader(eval_subset)
                eval_grads = compute_grads(trainer, eval_dataloader)
                
                influence = float(sum([torch.sum(eval_grad*train_grad) for eval_grad, train_grad in zip(eval_grads, train_grads)]))
                dict_influences[path_influence][eval_dataset_name][train_dataset_name] = influence

    df_influences = []
    for path_influence, dict_influence in dict_influences.items():
        df_influence = pd.DataFrame(dict_influence).sort_index(axis=1)
        df_influences.append(df_influence)
        df_influence.to_pickle(path_influence)
        
    return df_influences


def arnoldi(trainer, data_args, normalize=True):
    trainer.model.eval()
    arnoldi_path = f"{trainer.args.arnoldi_path}.pt"
    print(f"loading from: {arnoldi_path}")
    eigvals, eigvecs = torch.load(arnoldi_path)
    eigvals = eigvals.to(trainer.model.device)

    dict_arnoldis = {}
    for dataset_name in data_args.eval_dir:
        output_dir = trainer.args.output_dir.replace("=".join(data_args.eval_dir), dataset_name)
        path_arnoldi = os.path.join(output_dir, "df_arnoldi.pkl")
        subprocess.call(f"mkdir {output_dir}".split())
        print(f"path_arnoldi: {path_arnoldi}")
        dict_arnoldis[path_arnoldi] = defaultdict(dict)
    
    train_dataset_names = trainer.train_dataset.unique("dataset")
    for train_dataset_name in tqdm(train_dataset_names):
        train_subset = trainer.train_dataset.filter(lambda ex: ex["dataset"] == train_dataset_name)
        subset_dataloader = trainer.get_dataloader(train_subset)
        train_grads = compute_grads(trainer, subset_dataloader, max_steps=trainer.args.max_steps)

        train_grads = torch.stack([dot_product_fn(train_grads, to_gpu(eigvec, trainer.model.device)) for eigvec in eigvecs])
        train_grads = train_grads / eigvals
        
        if normalize:
            grad_norm = sum([torch.sum(grad**2) for grad in train_grads])**(1/2)
            train_grads = tuple(grad/grad_norm for grad in train_grads)

        for eval_dataset, path_arnoldi in zip(trainer.eval_dataset, dict_arnoldis):
            eval_dataset_names = eval_dataset.unique("dataset")
            
            for eval_dataset_name in eval_dataset_names:
                eval_subset = eval_dataset.filter(lambda ex: ex["dataset"] == eval_dataset_name)
                eval_dataloader = trainer.get_eval_dataloader(eval_subset)
                eval_grads = compute_grads(trainer, eval_dataloader)
                
                eval_grads = torch.stack([dot_product_fn(eval_grads, to_gpu(eigvec, trainer.model.device)) for eigvec in eigvecs])

                arnoldi = float(sum([torch.sum(eval_grad*train_grad) for eval_grad, train_grad in zip(eval_grads, train_grads)]))
                dict_arnoldis[path_arnoldi][eval_dataset_name][train_dataset_name] = arnoldi

    df_arnoldis = []
    for path_arnoldi, dict_arnoldi in dict_arnoldis.items():
        df_arnoldi = pd.DataFrame(dict_arnoldi).sort_index(axis=1)
        df_arnoldis.append(df_arnoldi)
        df_arnoldi.to_pickle(path_arnoldi)
        
    return df_arnoldis


def save_arnoldi(trainer):
    arnoldi_path = f"{trainer.args.arnoldi_path}.pt"
    print(f"arnold_path: {arnoldi_path}")
    
    trainer.train_dataloader = iter(trainer.get_train_dataloader())
    arnoldi_result = arnoldi_iter(trainer, n_iters=trainer.args.arnoldi_steps)
    eigvals, eigvecs = arnoldi_distill(arnoldi_result, top_k=trainer.args.arnoldi_proj)
    torch.save((eigvals, eigvecs), arnoldi_path)
    print(f"saved to {arnoldi_path}")
    
    return eigvals, eigvecs


if __name__ == "__main__":
    trainer, data_args = configure()
    all_metrics = run(trainer, data_args)
