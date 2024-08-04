for folder in 0 1 2 3 4 5 6 7 8 9
    do
    accelerate launch  run_cls.py \
      --train_file ./data_r/train_$folder.json  \
      --validation_file ./data_r/test_$folder.json  \
      --model_name_or_path ./output_r/$folder/mlm/model.pth  \
      --max_length 64 \
      --per_device_train_batch_size 256 \
      --per_device_eval_batch_size 5 \
      --with_tracking \
      --learning_rate 2e-5 \
      --num_train_epochs 5 \
      --output_dir ./output_r/$folder/cls/ \
      --checkpointing_steps epoch
done
