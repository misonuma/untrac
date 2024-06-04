#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model_name_or_path google/t5-xl-lm-adapt \
    --optim adafactor \
    --per_device_train_batch_size 2 \
    --train_dir synthetic/synthetic_train00_dataset \
    --loo_dataset_names dataset1
    --eval_dir synthetic/synthetic_eval00_dataset \
    --eval_steps 128 \
    --logging_steps 128 \
    --save_strategy no \
    --num_train_epochs 1 \
    --output_dir model/t5_00_loo1 &
    
CUDA_VISIBLE_DEVICES=1 python main.py \
    --model_name_or_path google/t5-xl-lm-adapt \
    --optim adafactor \
    --per_device_train_batch_size 2 \
    --train_dir synthetic/synthetic_train00_dataset \
    --loo_dataset_names dataset2
    --eval_dir synthetic/synthetic_eval00_dataset \
    --eval_steps 128 \
    --logging_steps 128 \
    --save_strategy no \
    --num_train_epochs 1 \
    --output_dir model/t5_00_loo2 &
    
CUDA_VISIBLE_DEVICES=2 python main.py \
    --model_name_or_path google/t5-xl-lm-adapt \
    --optim adafactor \
    --per_device_train_batch_size 2 \
    --train_dir synthetic/synthetic_train00_dataset \
    --loo_dataset_names dataset3
    --eval_dir synthetic/synthetic_eval00_dataset \
    --eval_steps 128 \
    --logging_steps 128 \
    --save_strategy no \
    --num_train_epochs 1 \
    --output_dir model/t5_00_loo3 &
    
CUDA_VISIBLE_DEVICES=3 python main.py \
    --model_name_or_path google/t5-xl-lm-adapt \
    --optim adafactor \
    --per_device_train_batch_size 2 \
    --train_dir synthetic/synthetic_train00_dataset \
    --loo_dataset_names dataset4
    --eval_dir synthetic/synthetic_eval00_dataset \
    --eval_steps 128 \
    --logging_steps 128 \
    --save_strategy no \
    --num_train_epochs 1 \
    --output_dir model/t5_00_loo4 &
    
wait