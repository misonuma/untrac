#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python main.py \
    --unlearn \
    --optim adafactor \
    --max_grad_norm 0 \
    --model_name_or_path model/t5_00 \
    --train_dir data/synthetic_train00_dataset \
    --dataset_names dataset1 \
    --eval_dir data/synthetic_eval00_dataset \
    --eval_steps 16 \
    --logging_steps 16 \
    --save_strategy no \
    --num_train_epochs 1 &
    
CUDA_VISIBLE_DEVICES=0 python main.py \
    --unlearn \
    --optim adafactor \
    --max_grad_norm 0 \
    --model_name_or_path model/t5_00 \
    --train_dir data/synthetic_train00_dataset \
    --dataset_names dataset2 \
    --eval_dir data/synthetic_eval00_dataset \
    --eval_steps 16 \
    --logging_steps 16 \
    --save_strategy no \
    --num_train_epochs 1 &
    
CUDA_VISIBLE_DEVICES=1 python main.py \
    --unlearn \
    --optim adafactor \
    --max_grad_norm 0 \
    --model_name_or_path model/t5_00 \
    --train_dir data/synthetic_train00_dataset \
    --dataset_names dataset3 \
    --eval_dir data/synthetic_eval00_dataset \
    --eval_steps 16 \
    --logging_steps 16 \
    --save_strategy no \
    --num_train_epochs 1 &
    
CUDA_VISIBLE_DEVICES=1 python main.py \
    --unlearn \
    --optim adafactor \
    --max_grad_norm 0 \
    --model_name_or_path model/t5_00 \
    --train_dir data/synthetic_train00_dataset \
    --dataset_names dataset4 \
    --eval_dir data/synthetic_eval00_dataset \
    --eval_steps 16 \
    --logging_steps 16 \
    --save_strategy no \
    --num_train_epochs 1 &
    
wait