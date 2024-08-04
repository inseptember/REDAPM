for folder in 0
    do
    accelerate launch  run_drg.py \
      --train_file ./data_r/all5/train_$folder.json  \
      --validation_file ./data_r/all5/test_$folder.json  \
      --drug_indx_file ./data_r/all3/drug_label_indx.json \
      --model_name_or_path ./xr/bert-base-chinese/  \
      --max_length 400 \
      --per_device_train_batch_size 5 \
      --per_device_eval_batch_size 5 \
      --with_tracking \
      --learning_rate 1e-5 \
      --num_train_epochs 5 \
      --output_dir ./output_r/all5/$folder/ \
      --checkpointing_steps epoch
done
