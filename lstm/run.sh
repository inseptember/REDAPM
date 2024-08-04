export HF_ENDPOINT=https://hf-mirror.com

for folder in 0 1 2 3 4 5 6 7 8 9
    do
    accelerate launch  run.py \
      --train_file ../data_r/all/train_$folder.json  \
      --validation_file ../data_r/all/test_$folder.json  \
      --model_name_or_path /opt/storage3/model/bert-base-chinese \
      --max_length 400 \
      --per_device_train_batch_size 32 \
      --per_device_eval_batch_size 5 \
      --with_tracking \
      --learning_rate 2e-5 \
      --num_train_epochs 5 \
      --output_dir ./output_r/$folder/ \
      --checkpointing_steps epoch
done
