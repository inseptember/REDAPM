#!/bin/bash
folder=0
while (( folder < 10 ))
do
    accelerate launch  eval.py \
      --validation_file ../data_r/all/validation.json  \
      --model_name_or_path ./output_r/$folder/  \
      --max_length 400 \
      --per_device_eval_batch_size 1 \
      --with_tracking \
      --learning_rate 2e-5 \
      --output_file valid \
      --output_dir ./output_r/$folder/
    (( folder = folder + 1 ))
done
