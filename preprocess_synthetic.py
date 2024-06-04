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

import os
from datasets import Dataset

paths = [
    "synthetic_train00_dataset",
    "synthetic_eval00_dataset",
    "synthetic_train10_dataset",
    "synthetic_eval10_dataset"
]

for path in paths:
    dataset = Dataset.from_csv(os.path.join("synthetic", f"{path}.csv"))
    dataset.save_to_disk(os.path.join("data", path))
