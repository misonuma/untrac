# +
import logging
import random
import string
import pdb

import torch
import numpy as np

from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase, default_data_collator
from arguments import DataTrainingArguments

from itertools import chain
import warnings

logger = logging.getLogger(__name__)


# -

@dataclass
class DataCollatorForOPT:
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
        self.max_seq_length=data_args.max_seq_length
        
        self.sep_token = f"########"
        self.sep_token_ids = self.tokenizer.encode(self.sep_token, add_special_tokens=False)

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
            
        if "input_ids" in batch[0]: 
            inputs = default_data_collator(batch, return_tensors=return_tensors)
            if "dataset" in batch[0]: 
                inputs["dataset"] = [instance["dataset"] for instance in batch]
            return inputs
            
        inputs = []
        preds = []
        answers = []
        datasets = []
        categories = []
        separate = False
        seq2seq = False
        for instance in batch:
            input = instance["inputs_pretokenized"]
            
            if "targets_pretokenized" in instance and instance["targets_pretokenized"] is not None:
                target = instance["targets_pretokenized"]
                input = f"{input}{self.sep_token} {target}"
                separate = True
                
            inputs.append(input)
            
            if "preds_pretokenized" in instance:
                preds.append(instance["preds_pretokenized"])
                answers.append(instance["answers"])
                seq2seq = True
            
            if "dataset" in instance: datasets.append(instance["dataset"])

            if "category" in instance: categories.append(instance["category"])
                
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_seq_length,
            padding=self.padding,
            return_tensors=self.return_tensors, 
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            add_special_tokens=True,
        )
        
        if seq2seq:
            model_preds = self.tokenizer(
                preds,
                max_length=self.max_seq_length,
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
            model_inputs["preds"] = model_preds["input_ids"]
            
            model_answers = self.tokenizer(
                answers,
                max_length=self.max_seq_length,
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
            model_inputs["answers"] = model_answers["input_ids"]
        else:
            label_mask = model_inputs["attention_mask"].bool()
            model_inputs["labels"] = model_inputs["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)
        
        if separate:
            input_ids_list = []
            token_type_ids_list = []
            attention_mask_list = []
            labels_list = []

            for model_input_values in zip(*model_inputs.values()):
                model_input = {key: value for key, value in zip(model_inputs.keys(), model_input_values)}
                input_ids = model_input["input_ids"]
                attention_mask = model_input["attention_mask"]
                labels = model_input["labels"]
                if "token_type_ids" in model_inputs: 
                    token_type_ids = model_input["token_type_ids"]
                else:
                    token_type_ids = None
                
                sep_token_ids_start_idx = None

                for idx in np.where(input_ids == self.sep_token_ids[0])[0]:
                    if (self.sep_token_ids == input_ids[idx : idx + len(self.sep_token_ids)].tolist()):
                        sep_token_ids_start_idx = idx

                if sep_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.sep_token}` in the "
                        f"following instance: {self.tokenizer.decode(input_ids)} "
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    labels = torch.full_like(input_ids, self.label_pad_token_id)
                else:
                    sep_token_ids_end_idx = sep_token_ids_start_idx + len(self.sep_token_ids)

                    input_ids = torch.concat([input_ids[:sep_token_ids_start_idx], input_ids[sep_token_ids_end_idx:]])
                    attention_mask = torch.concat([attention_mask[:sep_token_ids_start_idx], attention_mask[sep_token_ids_end_idx:]])
                    labels = torch.concat([labels[:sep_token_ids_start_idx], labels[sep_token_ids_end_idx:]])
                    if token_type_ids: token_type_ids = torch.concat([token_type_ids[:sep_token_ids_start_idx], token_type_ids[sep_token_ids_end_idx:]])
                    labels[:sep_token_ids_start_idx] = self.label_pad_token_id

                input_ids_list.append(input_ids[None, :])
                attention_mask_list.append(attention_mask[None, :])
                labels_list.append(labels[None, :])
                if token_type_ids: token_type_ids_list.append(token_type_ids[None, :])

            model_inputs = {
                "input_ids": torch.concat(input_ids_list, 0),
                "attention_mask": torch.concat(attention_mask_list, 0),
                "labels": torch.concat(labels_list, 0),
            }
            if len(token_type_ids_list) > 0: model_inputs["token_type_ids"] = torch.concat(token_type_ids_list, 0)
            
        if len(datasets) > 0: model_inputs["dataset"] = datasets
        if len(categories) > 0: model_inputs["category"] = categories
        return model_inputs






