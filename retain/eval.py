# coding=utf-8

""" Branimir Ljubic, Ameen Abdel Hai, Marija Stanojevic, Wilson Diaz, Daniel Polimac, Martin Pavlovski, Zoran Obradovic,
Predicting complications of diabetes mellitus using advanced machine learning algorithms,
Journal of the American Medical Informatics Association, Volume 27, Issue 9, September 2020, Pages 1343??1351,
https://doi.org/10.1093/jamia/ocaa120"""

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
from scipy.sparse import coo_matrix
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
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

from model import RETAIN

# wandb.init(project="f_dep_text_4", entity="fengwei")
logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
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
    parser.add_argument(
        "--output_file", type=str, default="test"
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--model_path", type=str
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


class VisitSequenceWithLabelDataset(Dataset):
    def __init__(self, file_path, num_features, reverse=True):
        _data = pd.read_json(file_path, lines=True, orient="records")
        seqs = _data['input'].values
        y2 = _data['y2'].values
        if len(seqs) != len(y2):
            raise ValueError("Seqs and Labels have different lengths")

        self.seqs = []
        self.labels = []

        for seq, label in zip(seqs, y2):

            if reverse:
                sequence = list(reversed(seq))
            else:
                sequence = seq

            row = []
            col = []
            val = []
            for i, visit in enumerate(sequence):
                for code in visit:
                    if code < num_features:
                        row.append(i)
                        col.append(code)
                        val.append(1.0)
            if len(row) == 0:
                continue
            self.seqs.append(coo_matrix((np.array(val, dtype=np.float32), (np.array(row), np.array(col))),
                                        shape=(len(sequence), num_features)))
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.seqs[index], self.labels[index]

def visit_collate_fn(batch: Any) -> Dict[str, Any]:
    batch_seq, batch_label = zip(*batch)
    num_features = batch_seq[0].shape[1]
    seq_lengths = list(map(lambda patient_tensor: patient_tensor.shape[0], batch_seq))
    max_length = max(seq_lengths)

    sorted_indices, sorted_lengths = zip(*sorted(enumerate(seq_lengths), key=lambda x: x[1], reverse=True))
    sorted_padded_seqs = []
    sorted_labels = []

    for i in sorted_indices:
        length = batch_seq[i].shape[0]

        if length < max_length:
            padded = np.concatenate(
                (batch_seq[i].toarray(), np.zeros((max_length - length, num_features), dtype=np.float32)), axis=0)
        else:
            padded = batch_seq[i].toarray()

        sorted_padded_seqs.append(padded)
        sorted_labels.append(batch_label[i])

    seq_tensor = np.stack(sorted_padded_seqs, axis=0)
    label_tensor = torch.LongTensor(sorted_labels)

    return {
        'x': torch.from_numpy(seq_tensor),
        'labels': label_tensor,
        'lengths': list(sorted_lengths)
    }


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
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    label_column = "y2"
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    num_features = 2200
    model = torch.load(args.model_path, map_location='cpu')

    eval_dataset = VisitSequenceWithLabelDataset(args.validation_file, num_features)

    eval_dataloader = DataLoader(eval_dataset, collate_fn=visit_collate_fn, batch_size=args.per_device_eval_batch_size)

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
        predictions = logits.argmax(dim=-1)
        predict_probs = logits.softmax(-1)[:, -1].contiguous()
        # predictions = (predict_probs > 0.5).int()
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
