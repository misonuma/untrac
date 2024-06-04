#!/bin/sh

if [ ! -d data ]; then
  mkdir data
fi

python preprocess_synthetic.py