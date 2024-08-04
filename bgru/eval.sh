#!/bin/bash
folder=0
while (( folder < 10 ))
do
    accelerate launch  eval.py \
      --validation_file ./data_r/test_$folder.json  \
      --model_path ./output_r/$folder/model.pth \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 1 \
      --with_tracking \
      --learning_rate 2e-5 \
      --num_train_epochs 20 \
      --output_dir ./output_r/$folder/
    (( folder = folder + 1 ))
done
