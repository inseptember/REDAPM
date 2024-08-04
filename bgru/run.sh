
export HF_ENDPOINT=https://hf-mirror.com

for folder in 0 1 2 3 4 5 6 7 8 9
    do
    accelerate launch --main_process_port=9999 run.py \
      --train_file ./data_r/train_$folder.json  \
      --validation_file ./data_r/test_$folder.json  \
      --per_device_train_batch_size 128 \
      --per_device_eval_batch_size 5 \
      --with_tracking \
      --learning_rate 1e-5 \
      --num_train_epochs 5 \
      --output_dir ./output_r/$folder/ \
      --checkpointing_steps epoch
done
