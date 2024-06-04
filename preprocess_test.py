# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: untrac
#     language: python
#     name: untrac
# ---

# +
from collections import Counter

from torch.utils.data import RandomSampler
import datasets
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from transformers import (
    set_seed,
)
set_seed(seed=42)

def sample_dataset(dataset, num_samples, generator=None, dataset_name=None):
    sampler = RandomSampler(dataset, num_samples=num_samples, generator=generator)
    sampled_dataset = dataset.select(sampler)
    if dataset_name is not None: sampled_dataset = sampled_dataset.map(lambda ex: {"dataset": dataset_name})
    return sampled_dataset


# -

# # ToxiGen

# +
toxigen_dataset = load_dataset("skg/toxigen-data", name="train", use_auth_token=True, split="train")
toxigen_dataset = toxigen_dataset.filter(lambda ex: ex["prompt_label"] == 1)

groups = toxigen_dataset.unique("group")
toxigen_datasets = DatasetDict({group: toxigen_dataset.filter(lambda ex: ex["group"] == group) for group in groups})


# +
def preprocess_dataset(dataset, num_samples=None):
    dataset = dataset.rename_column("generation", "inputs_pretokenized")
    dataset = dataset.rename_column("group", "dataset")
    dataset = dataset.remove_columns([column for column in dataset.column_names if column not in ["inputs_pretokenized", "dataset"]])
    
    if num_samples is not None: dataset = sample_dataset(dataset, num_samples=num_samples)
    return dataset

toxigen_datasets = toxigen_datasets.map(lambda ex: {"length": len(ex["generation"].split(" "))})
toxigen_datasets = toxigen_datasets.sort("length")
toxigen_datasets = toxigen_datasets.filter(lambda ex: ex["length"] > 8 and ex["length"] <= 24)
# -

toxigen_dataset = concatenate_datasets([preprocess_dataset(dataset, num_samples=256) for dataset_name, dataset in toxigen_datasets.items()])
toxigen_dataset.save_to_disk("data/toxigen_test_dataset")

# # WinoBias

# +
pro_dataset = load_dataset("wino_bias", name="type1_pro", split="validation+test")
pro_dataset = pro_dataset.map(lambda ex: {"inputs_pretokenized": " ".join(ex["tokens"]), "type": "pro"})

anti_dataset = load_dataset("wino_bias", name="type1_anti", split="validation+test")
anti_dataset = anti_dataset.map(lambda ex: {"inputs_pretokenized": " ".join(ex["tokens"]), "type": "anti"})

winobias_dataset = concatenate_datasets([pro_dataset, anti_dataset])


# +
def coref(ex):
    female = any([coref in ex["tokens"] for coref in ["she", "her"]])
    male = any([coref in ex["tokens"] for coref in ["he", "his", "him"]])
    if female and not male:
        dataset = ex["type"] + "_female"
    elif male and not female:
        dataset = ex["type"] + "_male"
    else:
        dataset = None
    
    return {"dataset": dataset}

winobias_dataset = winobias_dataset.map(coref)
winobias_dataset = winobias_dataset.filter(lambda ex: ex["dataset"] is not None)
winobias_dataset = winobias_dataset.remove_columns([column for column in winobias_dataset.column_names if column not in ["inputs_pretokenized", "targets_pretokenized", "dataset"]])
winobias_datasets = {
    dataset: winobias_dataset.filter(lambda ex: ex["dataset"] == dataset)
    for dataset in winobias_dataset.unique("dataset")
}
# -

test_dataset = concatenate_datasets([sample_dataset(dataset, num_samples=256) for dataset_name, dataset in winobias_datasets.items()])
test_dataset.save_to_disk("data/winobias_test_dataset")

# # TruthfulQA

# +
raw_dataset = load_dataset("truthful_qa", name="generation", split="validation")

truthfulqa_dataset = []
for ex in raw_dataset:
    if "Indexical Error" in ex["category"]:
        dataset_name = "indexical_error"
    elif "Confusion" in ex["category"]:
        dataset_name = "confusion"
    else:
        dataset_name = ex["category"].lower()
    
    for incorrect_answer in ex["incorrect_answers"]:
        truthfulqa_dataset.append({"inputs_pretokenized": ex["question"].strip(), "targets_pretokenized": incorrect_answer.strip(), "dataset": dataset_name})
truthfulqa_dataset = Dataset.from_list(truthfulqa_dataset)

# +
dataset_names = [dataset_name for dataset_name, count in Counter(truthfulqa_dataset["dataset"]).items() if count>=128]
truthfulqa_dataset = truthfulqa_dataset.filter(lambda ex: ex["dataset"] in dataset_names)

truthfulqa_datasets = {
    dataset: truthfulqa_dataset.filter(lambda ex: ex["dataset"] == dataset)
    for dataset in truthfulqa_dataset.unique("dataset")
}
# -

test_dataset = concatenate_datasets([sample_dataset(dataset, num_samples=256) for dataset_name, dataset in truthfulqa_datasets.items()])
test_dataset.save_to_disk("data/truthfulqa_test_dataset")
