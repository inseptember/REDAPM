# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a ?? Transformers model for sequence classification on GLUE."""

# import wandb

import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Any

import datasets
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from model_mlm import BertForMaskedLM, MyBertConfig

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the ?? Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--visit_text",
        type=str
    )
    args = parser.parse_args()

    return args



@dataclass
class DataCollatorWithPadding:
    mask_token_id = 103
    mlm_probability = 0.15
    max_seq_length = 64
    seq_token_id = 102
    def generate_mlm(self, input_ids, attention_mask, mlm_id=-100, mlm_p=0.8):
        input_labels = input_ids.clone()
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability).to(input_ids.device)
        probability_matrix.masked_fill_(attention_mask == 0, value=0.0)
        probability_matrix[:, 0] = 0
        masked_indices = torch.bernoulli(probability_matrix).bool()
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, mlm_p)).bool().to(
            input_ids.device) & masked_indices
        input_ids[indices_replaced] = self.mask_token_id
        input_labels[~indices_replaced] = mlm_id
        return input_labels, input_ids

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 'input_ids': [], 'segment_ids': [], 'position_ids': []
        max_seq_length = min(max([len(i['input_ids']) for i in features]), self.max_seq_length)
        batch_size = len(features)
        input_ids = torch.zeros((batch_size, max_seq_length), dtype=torch.int)
        segment_ids = torch.zeros((batch_size, max_seq_length), dtype=torch.int)
        position_ids = torch.zeros((batch_size, max_seq_length), dtype=torch.int)
        attention_mask = torch.zeros((batch_size, max_seq_length), dtype=torch.int)
        for i, feature in enumerate(features):
            if len(feature['input_ids']) > max_seq_length:
                feature['input_ids'] = feature['input_ids'][:max_seq_length]
                feature['input_ids'][-1] = self.seq_token_id
                feature['segment_ids'] = feature['segment_ids'][:max_seq_length]
                feature['position_ids'] = feature['position_ids'][:max_seq_length]
            input_ids[i, :len(feature['input_ids'])] = torch.tensor(feature['input_ids'])
            attention_mask[i, :len(feature['input_ids'])] = torch.ones(len(feature['input_ids']))
            segment_ids[i] = torch.ones(max_seq_length) * feature['segment_ids'][-1]
            segment_ids[i, :len(feature['segment_ids'])] = torch.tensor(feature['segment_ids'])
            position_ids[i] = torch.ones(max_seq_length) * feature['position_ids'][-1]
            position_ids[i, :len(feature['position_ids'])] = torch.tensor(feature['position_ids'])

        input_labels, input_ids = self.generate_mlm(input_ids, attention_mask)
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'seg_ids': segment_ids,
            'posi_ids': position_ids,
            'age_ids': torch.tensor([max(1, min(i['age'], 99)) for i in features]).int().unsqueeze(-1).repeat((1, max_seq_length)),
            # 'gender': torch.tensor([i['gender'] for i in features]).int(),
            'masked_lm_labels': input_labels
        }

        return batch

def main():
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_glue_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to)
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    model_config = {
        'vocab_size': 6900,  # number of disease + symbols for word embedding
        'hidden_size': 240,  # word embedding and seg embedding hidden size
        'seg_vocab_size': 2,  # number of vocab for seg embedding
        'age_vocab_size': 100,  # number of vocab for age embedding
        'max_position_embedding': 64,  # maximum number of tokens
        'hidden_dropout_prob': 0.1,  # dropout rate
        'num_hidden_layers': 9,  # number of multi-head attention layers required
        'num_attention_heads': 12,  # number of attention heads
        'attention_probs_dropout_prob': 0.1,  # multi-head attention dropout rate
        'intermediate_size': 256,  # the size of the "intermediate" layer in the transformer encoder
        'hidden_act': 'gelu',
        'layer_norm_eps': 1e-6,
        # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
        'initializer_range': 0.02,  # parameter weight initializer range
        # 'output_attentions':True
    }
    config = MyBertConfig(model_config)
    model = BertForMaskedLM(config)




    def preprocess_function(examples):
        result = {
            'input_ids': [], 'segment_ids': [], 'position_ids': []
        }
        for i in examples['input']:
            for k in result.keys():
                result[k].append(i[k])
        result['age'] = examples['age']
        result['gender'] = examples['gender']

        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.filter(
            lambda example: len(example['input']['input_ids']) > 0
        ).map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding()

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("glue_no_trainer", experiment_config)

    # Get the metric function
    # metric = evaluate.load("./f1_score.py")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    training_m = []
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue
            outputs = model(**batch)
            loss = outputs[0]
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.set_description('loss %.4f' % loss.detach().float())
                progress_bar.update(1)
                completed_steps += 1

            # if isinstance(checkpointing_steps, int):
            #     if completed_steps % checkpointing_steps == 0:
            #         output_dir = f"step_{completed_steps }"
            #         if args.output_dir is not None:
            #             output_dir = os.path.join(args.output_dir, output_dir)
            #         accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break



        if args.with_tracking:
            m = {
                "train_loss": total_loss.item() / len(train_dataloader),
                "epoch": epoch,
                "step": completed_steps,
            }
            accelerator.log(
                m,
                step=completed_steps,
            )

            training_m.append(m)

        # if epoch < args.num_train_epochs - 1:
        #     accelerator.wait_for_everyone()
        #     unwrapped_model = accelerator.unwrap_model(model)
        #     unwrapped_model.save_pretrained(
        #         args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        #     )
        #     if accelerator.is_main_process:
        #         tokenizer.save_pretrained(args.output_dir)

        # if args.checkpointing_steps == "epoch":
        #     output_dir = f"epoch_{epoch}"
        #     if args.output_dir is not None:
        #         output_dir = os.path.join(args.output_dir, output_dir)
        #     accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        # unwrapped_model.save_pretrained(
        #     args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        # )
        accelerator.save(unwrapped_model, args.output_dir + '/model.pth')

    if args.output_dir is not None:

        with open(os.path.join(args.output_dir, "training.json"), "w") as f:
            json.dump(training_m, f)


if __name__ == "__main__":
    main()
