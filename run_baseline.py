#!/usr/bin/env python
# coding: utf-8
# %%
import os
import sys
from main import configure, run


# %%
argv = " ".join(sys.argv[1:])
trainer, data_args = configure(argv)
all_metrics = run(trainer, data_args)

