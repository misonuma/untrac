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


# +
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

        # for training
        task_sources = [] # inputs for frozen task predictor
        task_targets = [] # labels for frozen task predictor
        prefix_input_sources = [] # inputs for prefix generator
        prefix_target_sources = [] # labels for prefix generator
        
        # for inference
        sources = []
        choices = []
        target_choices = []
        datasets = []
        categories = []
        for instance in batch:
            prompted_input = instance["inputs_pretokenized"]
            task_target = instance["targets_pretokenized"]
            
            sources.append(prompted_input)
            task_targets.append(task_target)
            
#             if "choices_pretokenized" in instance:
#                 choice = instance["choices_pretokenized"]
#                 choices.append(choice)
                
#                 target_choice = [task_target == c for c in choice]
#                 assert sum(target_choice) == 1
#                 target_choices.append(target_choice)
                
            if "dataset" in instance:
                dataset = instance["dataset"]
                datasets.append(dataset)
            
            if "category" in instance:
                category = instance["category"]
                categories.append(category)
                
        
        # for inference
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
            
        if len(target_choices) > 0:
            n_choice = len(choices[0])
            model_inputs["n_choice"] = n_choice
            
            choices_flat = list(chain(*choices))
            target_choices_flat = torch.Tensor(list(chain(*target_choices)))
            if self.text_only:
                model_inputs["choices"] = choices_flat
            else:
                with self.tokenizer.as_target_tokenizer():
                    labels_choices = self.tokenizer(
                        choices_flat,
                        max_length=self.max_target_length,
                        padding=self.padding,
                        return_tensors=self.return_tensors,
                        truncation=False,
                        pad_to_multiple_of=self.pad_to_multiple_of
                    )
                    model_inputs["choices_mask"] = labels_choices["attention_mask"]
                    model_inputs["choices"] = labels_choices["input_ids"]
                    # mask model_inputs["choices"] based on labels_choices["attention_mask"] (pad tokens) and target_choices_flat (choices that are not target)
                    model_inputs["labels_logits"] = model_inputs["choices"].masked_fill(~(model_inputs["choices_mask"]*target_choices_flat.unsqueeze(-1)).bool(), self.label_pad_token_id)
                    if all([len(choice) == n_choice for choice in choices]):
                        model_inputs["labels_choices"] = torch.argmax(torch.Tensor(target_choices), -1)

            if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels") and not self.text_only:
                decoder_input_ids_choices = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["choices"])
                model_inputs["decoder_input_ids_choices"] = decoder_input_ids_choices

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
        model_inputs["category"] = categories
        
        return model_inputs
# -


