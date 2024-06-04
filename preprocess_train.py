# +
from itertools import chain

from torch.utils.data import RandomSampler
import datasets
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    set_seed,
)
set_seed(seed=42)

model_name_or_path = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
block_size = 1024
num_proc = 32


# +
def sample_dataset(dataset, num_samples, generator=None, dataset_name=None):
    sampler = RandomSampler(dataset, num_samples=num_samples, generator=generator)
    sampled_dataset = dataset.select(sampler)
    if dataset_name is not None: sampled_dataset = sampled_dataset.map(lambda ex: {"dataset": dataset_name})
    return sampled_dataset

def preprocess_dataset(dataset, num_samples=100000):
    dataset = dataset.remove_columns([column for column in dataset.column_names if column not in ["text"]])
    sampler = RandomSampler(dataset, num_samples=num_samples)
    dataset = dataset.select(sampler)
    return dataset

def tokenize(examples):
    output = tokenizer(examples["text"])
    return output

def group(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# -

# # equal training dataset size

raw_datasets = DatasetDict({
    "dataset00": preprocess_dataset(load_dataset("ArmelR/the-pile-splitted", name="HackerNews", split="train+test")),
    "dataset01": preprocess_dataset(load_dataset("bookcorpus", split="train"), num_samples=4000000),
    "dataset02": preprocess_dataset(load_dataset("cc_news", split="train"), num_samples=200000),
    "dataset03": preprocess_dataset(load_dataset("lucadiliello/STORIES", split="train"), num_samples=10000),
    "dataset04": preprocess_dataset(load_dataset("ArmelR/the-pile-splitted", name="Pile-CC", split="train+test")),
    "dataset05": preprocess_dataset(load_dataset("ArmelR/the-pile-splitted", name="Gutenberg (PG-19)", split="train+test"), num_samples=1000),
    "dataset06": preprocess_dataset(load_dataset("ArmelR/the-pile-splitted", name="OpenWebText2", split="train+test")),
    "dataset07": preprocess_dataset(load_dataset("ArmelR/the-pile-splitted", name="Wikipedia (en)", split="train+test")),
})

tokenized_datasets = raw_datasets.map(
        tokenize,
        batched=True,
        num_proc=num_proc,
        remove_columns=["text"],
        desc="Running tokenizer on dataset",
    )

train_datasets = tokenized_datasets.map(
        group,
        batched=True,
        num_proc=num_proc,
        desc="Running group on dataset",
    )

train_dataset = concatenate_datasets([sample_dataset(dataset, num_samples=40000, dataset_name=dataset_name) for dataset_name, dataset in train_datasets.items()])
train_dataset.save_to_disk("data/pretrain_dataset_equal")

# # different training dataset size

raw_datasets = DatasetDict({
    "dataset00": preprocess_dataset(load_dataset("ArmelR/the-pile-splitted", name="Pile-CC", split="train+test"), num_samples=200000),
    "dataset01": preprocess_dataset(load_dataset("ArmelR/the-pile-splitted", name="OpenWebText2", split="train+test"), num_samples=200000),
    "dataset02": preprocess_dataset(load_dataset("cc_news", split="train"), num_samples=200000),
    "dataset03": preprocess_dataset(load_dataset("bookcorpus", split="train"), num_samples=4000000),
    "dataset04": preprocess_dataset(load_dataset("lucadiliello/STORIES", split="train"), num_samples=10000),
    "dataset05": preprocess_dataset(load_dataset("ArmelR/the-pile-splitted", name="Gutenberg (PG-19)", split="train+test"), num_samples=400),
    "dataset06": preprocess_dataset(load_dataset("ArmelR/the-pile-splitted", name="HackerNews", split="train+test"), num_samples=40000),
    "dataset07": preprocess_dataset(load_dataset("ArmelR/the-pile-splitted", name="Wikipedia (en)", split="train+test"), num_samples=40000),
})

tokenized_datasets = raw_datasets.map(
        tokenize,
        batched=True,
        num_proc=num_proc,
        remove_columns=["text"],
        desc="Running tokenizer on dataset",
    )

raw_datasets = tokenized_datasets.map(
        group,
        batched=True,
        num_proc=num_proc,
        desc="Running group on dataset",
    )
raw_datasets = {dataset_name: dataset.add_column(column=range(len(dataset)), name="index").map(lambda ex: {"dataset": dataset_name}) for dataset_name, dataset in raw_datasets.items()}

# +
dataset_samples = {
    "dataset00": 0.30,
    "dataset01": 0.20,
    "dataset02": 0.15,
    "dataset03": 0.10,
    "dataset04": 0.10,
    "dataset05": 0.05,
    "dataset06": 0.05,
    "dataset07": 0.05,
}
assert sum(dataset_samples.values()) == 1

n_samples = 40000*8
train_samplers = {dataset_name: RandomSampler(dataset, num_samples=int(n_samples*dataset_samples[dataset_name])) for dataset_name, dataset in raw_datasets.items()}
train_datasets = {dataset_name: dataset.select(train_samplers[dataset_name]) for dataset_name, dataset in raw_datasets.items()}
train_dataset = concatenate_datasets(train_datasets.values())
train_dataset.save_to_disk("data/pretrain_dataset_diff")
