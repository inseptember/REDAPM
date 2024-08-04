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


import argparse
import json
import logging
import math
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
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

from model4_drg import BertForSequenceClassification

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
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
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
    parser.add_argument(
        "--output_file",
        default="test",
        type=str
    )
    args = parser.parse_args()

    return args

@dataclass
class DataCollatorWithPadding2(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        batch = {
            'input_ids': torch.tensor([i['input_ids'] for i in features]).int(),
            'attention_mask': torch.tensor([i['attention_mask'] for i in features]).int(),
            'labels': torch.tensor([i['labels'] for i in features]).float(),
            'other': torch.tensor([i['other'] for i in features]).float(),
            'visit_mask': torch.tensor([i['visit_mask'] for i in features]).int(),
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
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    label_column = "y2"
    label_list = raw_datasets["validation"].unique(label_column)
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)

    dis_features = ['is_hypertension', 'is_ischaemic_heart', 'is_heart_failure', 'is_foot', 'is_eye', 'is_stock',
                    'is_pad', 'is_dementia', 'is_cvd', 'is_renal', 'is_hypoglycemia', 'is_dk', 'is_hyperosmolar']
    sym_features = ['tired', 'sleep difficulty', 'appetite decreased', 'move slowly', 'irritable', 'cognitive',
                    'weight decreased', 'weight increased', 'dispirited']
    exam_features = ['hdl_avg_imp', 'ldl_avg_imp', 'sbp_avg_imp', 'dbp_avg_imp', 'tc_avg_imp', 'egfr_avg_imp',
                     'acr_avg_imp', 'ua_avg_imp', 'fbg_avg_imp', 'hba1c_avg_imp']
    drug_features = ['dpp4', 'glp1', 'glycosidase', 'insulin', 'metformin', 'sglt2', 'other', 'thiazolidinedione',
                     'glinide', 'sulphonylureas2']
    other_features = ['age', 'dis_count', 'gender']

    all_features = dis_features + sym_features + exam_features + drug_features + other_features

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        num_other=len(all_features) + 461
    )


    label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False
    # visit_text = pd.read_pickle(args.visit_text).sort_values(by=['diag_dt'])

    def preprocess_function(examples):

        def get_visit_texts(visit_sn, disease_name, max_visit_num=10):
            texts = [('', '') for i in range(max_visit_num)]
            mask = [0] * max_visit_num
            mask[0] = 1
            if visit_sn is None or len(visit_sn) == 0:
                return texts, mask
            # texts = list(zip(disease_name[-max_visit_num:][::-1], visit_sn[-max_visit_num:][::-1]))
            # visit_sn = visit_sn[::-1]
            # disease_name = disease_name[::-1]
            for i in range(max_visit_num):
                if i >= len(disease_name):
                    break
                texts[i] = (disease_name[i], visit_sn[i])
                mask[i] = 1
            # for i, t in enumerate(visit_sn[:max_visit_num][::-1]):
            #     texts[i] = (t, disease_name[i])
            return texts, mask

        # Tokenize the texts
        result = {'input_ids': [], 'attention_mask': [], 'visit_mask': []}
        for i, j in zip(examples['visit_sn'], examples['disease_names']):
            texts, v_mask = get_visit_texts(i, j)
            r = tokenizer(texts, padding='max_length', max_length=args.max_length, truncation=True)
            result['input_ids'].append(r['input_ids'])
            result['attention_mask'].append(r['attention_mask'])
            result['visit_mask'].append(v_mask)

        if label_column in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples[label_column]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples[label_column]
        result['other'] = [[float(examples[col][i]) for col in all_features] + examples['drug_code'][i] for i in range(len(result["labels"]))]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["validation"].column_names,
            desc="Running tokenizer on dataset",
        )

    eval_dataset = processed_datasets["validation"]

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding2(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Prepare everything with our `accelerator`.
    model, eval_dataloader = accelerator.prepare(
        model, eval_dataloader
    )

    # Get the metric function
    metric = evaluate.load("./f1_score.py")

    model.eval()
    refs, preds, pred_scores = [], [], []
    all_hidden_states = []
    samples_seen = 0
    progress_eval_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predict_probs = outputs.logits.softmax(-1)[:, 1].contiguous()
        predictions, references, predict_probs = accelerator.gather((predictions, batch["labels"], predict_probs))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(eval_dataloader) - 1:
                predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                references = references[: len(eval_dataloader.dataset) - samples_seen]
                predict_probs = predict_probs[: len(eval_dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
        metric.add_batch(
            predictions=predictions,
            references=references,
            predict_probs=predict_probs
        )
        preds = preds + predictions.cpu().numpy().tolist()
        pred_scores = pred_scores + predict_probs.cpu().numpy().tolist()
        refs = refs + references.cpu().numpy().tolist()
        progress_eval_bar.update(1)

    eval_metric = metric.compute()
    logger.info(f"{eval_metric}")



    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        with open(os.path.join(args.output_dir, f"eval_{args.output_file}_results.json"), "w") as f:
            json.dump(eval_metric, f)

        with open(os.path.join(args.output_dir, f"eval_{args.output_file}_results.pkl"), "wb") as f:
            pickle.dump({
                'prediction': preds, 'prediction_score': pred_scores, 'reference': refs
            }, f)


if __name__ == "__main__":
    main()
