#!/bin/bash
num=0
while (( $num < 10 ))
do
  accelerate launch eval_dis_text_drg.py \
  --validation_file ../../cv_text_2/train_$num.json  \
  --model_name_or_path ./output_dis_text_drg/$num/  \
  --max_length 300 \
  --per_device_train_batch_size 90 \
  --per_device_eval_batch_size 90 \
  --with_tracking \
  --learning_rate 2e-5 \
  --num_train_epochs 50 \
  --output_file train \
  --output_dir ./output_dis_text_drg/$num/ \
  --checkpointing_steps epoch

  accelerate launch eval_dis_text_drg.py \
  --validation_file ../../cv_text_2/test_$num.json  \
  --model_name_or_path ./output_dis_text_drg/$num/  \
  --max_length 300 \
  --per_device_train_batch_size 90 \
  --per_device_eval_batch_size 90 \
  --with_tracking \
  --learning_rate 2e-5 \
  --num_train_epochs 50 \
  --output_file test \
  --output_dir ./output_dis_text_drg/$num/ \
  --checkpointing_steps epoch
  (( num = $num + 1 ))
done
