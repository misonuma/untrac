#!/usr/bin/env python
# coding: utf-8
# %%
import os
import sys
from main import configure, run


# %%
argv = " ".join(sys.argv[1:])
_, data_args_all = configure(argv.format("all"))
for dataset in data_args_all.datasets:
    trainer, data_args = configure(argv.format(dataset))
    all_metrics = run(trainer, data_args)

