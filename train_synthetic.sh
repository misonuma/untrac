#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model_name_or_path google/t5-xl-lm-adapt \
    --optim adafactor \
    --per_device_train_batch_size 2 \
    --train_dir synthetic/synthetic_train00_dataset \
    --eval_dir synthetic/synthetic_train00_dataset \
    --eval_steps 128 \
    --logging_steps 128 \
    --save_strategy no \
    --num_train_epochs 1 \
    --output_dir model/t5_00 \