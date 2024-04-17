#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python main.py \
    --unlearn \
    --optim adafactor \
    --max_grad_norm 0 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 128 \
    --model_name_or_path model/t5_10 \
    --train_dir synthetic/synthetic_eval10_dataset \
    --eval_dir synthetic/synthetic_train10_dataset \
    --each_eval_samples 256 \
    --eval_steps 5 \
    --logging_steps 5 \
    --save_strategy no \
    --max_steps 50 &
    
wait