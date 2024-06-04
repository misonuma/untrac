#!/bin/sh

if [ ! -d data ]; then
  mkdir data
fi

python preprocess_train.py

python preprocess_test.py

CUDA_VISIBLE_DEVICES=0 python main.py \
    --scratch \
    --model_name_or_path facebook/opt-125m \
    --per_device_train_batch_size 8 \
    --train_dir data/pretrain_dataset_equal \
    --eval_dir data/toxigen_test_dataset data/winobias_test_dataset data/truthfulqa_test_dataset \
    --num_train_epochs 1 \
    --eval_steps 5000 \
    --logging_steps 5000 \
    --save_steps 5000 \
    --overwrite_output_dir False \
    --output_dir model/opt_equal \