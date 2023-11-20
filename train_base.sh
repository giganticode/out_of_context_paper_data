#!/bin/bash

echo $2
echo $1
echo ${@:3}

python run_summarization.py \
   --output_dir $2 \
   --train_file $1 \
   --model_name_or_path="Salesforce/codet5-base" \
   --task_name COMP \
   --do_train \
   --learning_rate 1e-4 \
   --num_train_epochs 5 \
   --seed 42 \
   --local_rank -1 \
   --data_seed 42 \
   --save_total_limit 1 \
   --save_steps 0.5 \
   --fp16 \
   --dataloader_num_workers 8 \
   --gradient_accumulation_steps 4 \
   --per_device_train_batch_size 3 \
   --per_device_eval_batch_size 16 ${@:3}