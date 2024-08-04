# coding=utf-8

""" Meng, Y., Speier, W., Ong, M., & Arnold, C. W. (2021).
HCET: Hierarchical Clinical Embedding With Topic Modeling on Electronic Health Records for Predicting Future Depression.
IEEE journal of biomedical and health informatics, 25(4), 1265??1272.
 https://doi.org/10.1109/JBHI.2020.3004072"""

# import wandb

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
    get_scheduler, AutoModel,
)

from model import HcetModel

# wandb.init(project="f_dep_text_4", entity="fengwei")
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
    parser.add_argument(
        "--output_file", type=str, default="test"
    )
    parser.add_argument(
        "--model_path", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
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
        default="all",
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
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_visit_num = max([i['visit_num'] for i in features] + [1])
        batch_size = len(features)
        batch = {
            'labels': torch.tensor([i['labels'] for i in features]).float(),
            'demo': torch.tensor([i['demo'] for i in features]).int(),
        }
        batch['demo'] = batch['demo'].unsqueeze(1).repeat((1, max_visit_num, 1))

        topic_feature = torch.zeros((batch_size, max_visit_num, max(
            [max([len(j) for j in i['topic']] + [1]) for i in features]
        ))).int()
        topic_mask = torch.zeros_like(topic_feature)
        disease_feature = torch.zeros((batch_size, max_visit_num, max(
            [max([len(j) for j in i['disease']] + [1]) for i in features]
        ))).int()
        disease_mask = torch.zeros_like(disease_feature)
        drug_feature = torch.zeros((batch_size, max(
            [len(i['drug_code']) for i in features] + [1]
        )))
        drug_mask = torch.zeros_like(drug_feature)
        for indx, feature in enumerate(features):
            for v, d in enumerate(feature['topic']):
                topic_feature[indx, v, :len(d)] = torch.tensor(d)
                topic_mask[indx, v, :len(d)] = torch.ones((len(d)))
            for v, d in enumerate(feature['disease']):
                disease_feature[indx, v, :len(d)] = torch.tensor(d)
                disease_mask[indx, v, :len(d)] = torch.ones((len(d)))
            if len(feature['drug_code']) > 0:
                drug_feature[indx, :len(feature['drug_code'])] = torch.tensor(feature['drug_code']) + 1
                drug_mask[indx, :len(feature['drug_code'])] = torch.ones((len(feature['drug_code'])))
        batch['drug'] = drug_feature.unsqueeze(1).repeat((1, max_visit_num, 1)).int()
        batch['drug_mask'] = drug_mask.unsqueeze(1).repeat((1, max_visit_num, 1)).int()
        batch['topic'] = topic_feature
        batch['disease'] = disease_feature
        batch['topic_mask'] = topic_mask
        batch['disease_mask'] = disease_mask

        return batch
#4587
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

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    # config.max_pooling = True
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    # bert = AutoModel.from_pretrained(
    #     args.model_name_or_path,
    #     from_tf=bool(".ckpt" in args.model_name_or_path),
    #     config=config,
    #     ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    # )
    model = HcetModel()
    model = torch.load(args.model_path, map_location='cpu')
    # model.embeddings.load_state_dict(bert.embeddings.state_dict())


    label_to_id = {v: i for i, v in enumerate(label_list)}

    # if label_to_id is not None:
    #     model.config.label2id = label_to_id
    #     model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False
    # visit_text = pd.read_pickle(args.visit_text).sort_values(by=['diag_dt'])



    def preprocess_function(examples):

        result = {
            'disease': examples['disease_names'],
            'topic': examples['visit_sn'],
            'visit_num': examples['visit_num'],
            'drug_code': examples['drug_code'],
            "demo": [[age + 2, gender] for age, gender in zip(examples['age'], examples['gender'])]
        }

        if label_column in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples[label_column]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples[label_column]

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
        data_collator = DataCollatorWithPadding()

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Prepare everything with our `accelerator`.
    model, eval_dataloader = accelerator.prepare(
        model, eval_dataloader
    )

    metric = evaluate.load("./f1_score.py")

    model.eval()
    refs, preds, pred_scores = [], [], []
    samples_seen = 0
    progress_eval_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs[1]
        predict_probs = logits.sigmoid()
        predictions = (predict_probs > 0.5).int()
        # predictions = outputs.logits.softmax(dim=-1)[:, -1]
        predictions, predict_probs, references = accelerator.gather((predictions, predict_probs, batch["labels"]))
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
