for folder in 0 1 2 3 4 5 6 7 8 9
    do
    python run_mlm.py \
          --train_file ./data_r/train_$folder.json  \
          --max_length 64 \
          --per_device_train_batch_size 256 \
          --per_device_eval_batch_size 256 \
          --with_tracking \
          --learning_rate 1e-4 \
          --num_train_epochs 5 \
          --output_dir ./output_r/$folder/mlm \
          --checkpointing_steps epoch \
          --weight_decay 0.01 \
          --gradient_accumulation_steps 1
done
