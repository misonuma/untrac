#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python main.py \
    --unlearn \
    --optim adafactor \
    --max_grad_norm 0 \
    --model_name_or_path model/t5_10 \
    --train_dir synthetic/synthetic_train10_dataset \
    --dataset_names dataset1 \
    --eval_dir synthetic/synthetic_eval10_dataset \
    --eval_steps 16 \
    --logging_steps 16 \
    --save_strategy no \
    --num_train_epochs 1 &
    
CUDA_VISIBLE_DEVICES=1 python main.py \
    --unlearn \
    --optim adafactor \
    --max_grad_norm 0 \
    --model_name_or_path model/t5_10 \
    --train_dir synthetic/synthetic_train10_dataset \
    --dataset_names dataset2 \
    --eval_dir synthetic/synthetic_eval10_dataset \
    --eval_steps 16 \
    --logging_steps 16 \
    --save_strategy no \
    --num_train_epochs 1 &
    
CUDA_VISIBLE_DEVICES=2 python main.py \
    --unlearn \
    --optim adafactor \
    --max_grad_norm 0 \
    --model_name_or_path model/t5_10 \
    --train_dir synthetic/synthetic_train10_dataset \
    --dataset_names dataset3 \
    --eval_dir synthetic/synthetic_eval10_dataset \
    --eval_steps 16 \
    --logging_steps 16 \
    --save_strategy no \
    --num_train_epochs 1 &
    
CUDA_VISIBLE_DEVICES=3 python main.py \
    --unlearn \
    --optim adafactor \
    --max_grad_norm 0 \
    --model_name_or_path model/t5_10 \
    --train_dir synthetic/synthetic_train10_dataset \
    --dataset_names dataset4 \
    --eval_dir synthetic/synthetic_eval10_dataset \
    --eval_steps 16 \
    --logging_steps 16 \
    --save_strategy no \
    --num_train_epochs 1 &
    
wait