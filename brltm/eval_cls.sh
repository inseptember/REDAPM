#!/bin/bash
folder=0
while (( folder < 10 ))
do
    accelerate launch  eval_cls.py \
      --validation_file ./data_r/test_$folder.json  \
      --model_name_or_path ./output_r/$folder/cls/model.pth  \
      --max_length 64 \
      --per_device_train_batch_size 256 \
      --per_device_eval_batch_size 5 \
      --with_tracking \
      --learning_rate 2e-5 \
      --num_train_epochs 10 \
      --output_dir ./output_r/$folder/cls/ \
      --checkpointing_steps epoch
    (( folder = folder + 1 ))
done
