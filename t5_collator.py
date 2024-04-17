# +
import logging
import random
import string
import pdb

import torch
import numpy as np

from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
from arguments import DataTrainingArguments

from itertools import chain

logger = logging.getLogger(__name__)


# -

@dataclass
class DataCollatorForT5:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: Optional[Any],
        data_args: DataTrainingArguments,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt",
        text_only: bool=False,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors
        self.text_only = text_only
        
        self.padding="max_length" if data_args.pad_to_max_length else "longest"
        self.max_source_length=data_args.max_source_length
        self.max_target_length=data_args.max_target_length

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []
        task_targets = [] 
        datasets = []
        for instance in batch:
            prompted_input = instance["inputs_pretokenized"]
            task_target = instance["targets_pretokenized"]
            
            sources.append(prompted_input)
            task_targets.append(task_target)
            
            if "dataset" in instance:
                dataset = instance["dataset"]
                datasets.append(dataset)
                
        if self.text_only:
            model_inputs = {"inputs": sources}
        else:
            model_inputs = self.tokenizer(
                sources,
                max_length=self.max_source_length,
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
                add_special_tokens=False,
            )
            
        if self.text_only:
            model_inputs["labels"] = task_targets
        else:
            with self.tokenizer.as_target_tokenizer():
                task_labels = self.tokenizer(
                    task_targets,
                    max_length=self.max_target_length,
                    padding=self.padding,
                    return_tensors=self.return_tensors,
                    truncation=True,
                    pad_to_multiple_of=self.pad_to_multiple_of
                )
            label_mask = task_labels["attention_mask"].bool()
            model_inputs["labels"] = task_labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)

        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels") and not self.text_only:
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
            model_inputs["decoder_input_ids"] = decoder_input_ids
            
        model_inputs["dataset"] = datasets
        
        return model_inputs


