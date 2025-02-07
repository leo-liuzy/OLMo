"""
Script for preparing the Tulu V2 data for fine-tuning an OLMo model.
"""

import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
import json
import os

import datasets as ds
import numpy as np
import pandas as pd

from tqdm import tqdm  # <-- use tqdm instead of "rich.progress.track"

from olmo.tokenizer import Tokenizer
from olmo.util import prepare_cli_environment

log = logging.getLogger(__name__)

def load_jsonlines(fname: str):
    with open(fname, "r") as f:
        return [json.loads(line) for line in f]

def preprocess(example, tokenizer: Tokenizer, max_seq_len: int):
    input_ids = [tokenizer.eos_token_id]
    label_mask = [False]
    for msg in example["messages"]:
        role_tokens = tokenizer.encode(f"<|{msg['role']}|>\n", add_special_tokens=False)
        label_mask += [False] * len(role_tokens)
        input_ids += role_tokens

        if msg["role"] == "assistant":
            content_tokens = tokenizer.encode(
                msg["content"].strip() + tokenizer.eos_token + "\n", add_special_tokens=False
            )
            label_mask += [True] * len(content_tokens)
            assert content_tokens[-2] == tokenizer.eos_token_id
            label_mask[-1] = False
        else:
            content_tokens = tokenizer.encode(
                msg["content"].strip() + "\n", add_special_tokens=False
            )
            label_mask += [False] * len(content_tokens)

        input_ids += content_tokens

    input_ids = input_ids[:max_seq_len]
    label_mask = label_mask[:max_seq_len]
    if len(input_ids) < max_seq_len:
        pad_len = max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        label_mask += [False] * pad_len

    assert len(input_ids) == len(label_mask)
    n_labels = sum(label_mask)
    return {"input_ids": input_ids, "label_mask": label_mask, "n_labels": n_labels}

def dscoder_preprocess(example, tokenizer: Tokenizer, max_seq_len: int):
    input_ids = [tokenizer.eos_token_id]
    label_mask = [False]
    for msg in example["messages"]:
        role_tokens = tokenizer.encode(f"<|{msg['role'].title()}|>\n", add_special_tokens=False)
        label_mask += [False] * len(role_tokens)
        input_ids += role_tokens

        if msg["role"] == "assistant":
            content_tokens = tokenizer.encode(
                msg["content"].strip() + tokenizer.eos_token + "\n", add_special_tokens=False
            )
            label_mask += [True] * len(content_tokens)
            assert content_tokens[-2] == tokenizer.eos_token_id
            label_mask[-1] = False
        else:
            content_tokens = tokenizer.encode(
                msg["content"].strip() + "\n", add_special_tokens=False
            )
            label_mask += [False] * len(content_tokens)

        input_ids += content_tokens

    input_ids = input_ids[:max_seq_len]
    label_mask = label_mask[:max_seq_len]
    if len(input_ids) < max_seq_len:
        pad_len = max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        label_mask += [False] * pad_len

    assert len(input_ids) == len(label_mask)
    n_labels = sum(label_mask)
    return {"input_ids": input_ids, "label_mask": label_mask, "n_labels": n_labels}

def dscoder_cpt_preprocess(example, tokenizer: Tokenizer, max_seq_len: int):
    input_ids = [tokenizer.eos_token_id]
    label_mask = [False]
    text = example["text"].strip() + tokenizer.eos_token + "\n"
    content_tokens = tokenizer.encode(text, add_special_tokens=False)
    input_ids += content_tokens
    label_mask += [True] * len(content_tokens)
    if len(content_tokens) >= 2:
        assert content_tokens[-2] == tokenizer.eos_token_id
        label_mask[-1] = False

    input_ids = input_ids[:max_seq_len]
    label_mask = label_mask[:max_seq_len]
    if len(input_ids) < max_seq_len:
        pad_len = max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        label_mask += [False] * pad_len
    n_labels = sum(label_mask)
    return {
        "input_ids": input_ids,
        "label_mask": label_mask,
        "n_labels": n_labels
    }

def filter(example):
    return example["n_labels"] > 0

def main(opts) -> None:
    if Path(opts.tokenizer).is_file():
        tokenizer = Tokenizer.from_file(
            opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad
        )
    else:
        tokenizer = Tokenizer.from_pretrained(
            opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad
        )

    log.info("Tokenizing dataset...")

    if opts.dataset == "allenai/tulu-v2-sft-mixture":
        dataset = ds.load_dataset("allenai/tulu-v2-sft-mixture", split="train")
        dataset = dataset.map(
            partial(preprocess, tokenizer=tokenizer, max_seq_len=opts.seq_len),
            batched=False,
            remove_columns=["dataset", "id", "messages"],
            num_proc=opts.num_proc
        )
    else:
        assert os.path.exists(opts.dataset)
        if "cpt" in opts.dataset: 
            dataset = ds.load_dataset("json", data_files=opts.dataset, streaming=True)["train"]
            dataset = dataset.map(
                partial(dscoder_cpt_preprocess, tokenizer=tokenizer, max_seq_len=opts.seq_len),
                batched=False,
            )
        else:
            # Everything else: read fully into memory
            rows = load_jsonlines(opts.dataset)
            dataset = ds.Dataset.from_pandas(pd.DataFrame(rows))
            dataset = dataset.map(
                partial(dscoder_preprocess, tokenizer=tokenizer, max_seq_len=opts.seq_len),
                batched=False,
                remove_columns=dataset.column_names,
            )

    log.info("Filtering dataset...")

    # Streaming or non-streaming
    if isinstance(dataset, ds.IterableDataset):
        log.info("Streaming dataset detected: filtering in single process, skipping length-based logs.")
        dataset = dataset.filter(filter, batched=False)
    else:
        n = len(dataset)
        dataset = dataset.filter(filter, batched=False, num_proc=opts.num_proc)
        log.info(f"Filtered out {n - len(dataset):,d} examples")

    log.info("Counting tokens...")
    total_tokens = 0

    # Use tqdm to show how many examples we've processed
    for ex in tqdm(dataset, desc="Counting tokens", unit="ex"):
        assert len(ex["input_ids"]) == opts.seq_len, "Unexpected seq length!"
        total_tokens += len(ex["input_ids"])

    log.info(f"Total tokens: {total_tokens:,d}")
    log.info(f"Saving results to '{opts.output_dir}'...")

    output_dir = Path(opts.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    input_ids_file = np.memmap(
        str(output_dir / "input_ids.npy"),
        dtype=np.uint16,
        mode="w+",
        shape=(total_tokens,)
    )
    label_mask_file = np.memmap(
        str(output_dir / "label_mask.npy"),
        dtype=np.bool_,
        mode="w+",
        shape=(total_tokens,)
    )

    offset = 0
    for ex in tqdm(dataset, desc="Writing examples", unit="ex"):
        ex_len = len(ex["input_ids"])
        input_ids_file[offset : offset + ex_len] = ex["input_ids"]
        label_mask_file[offset : offset + ex_len] = ex["label_mask"]
        offset += ex_len

    input_ids_file.flush()
    label_mask_file.flush()

    log.info("Done!")

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Prepare Tulu V2 dataset")
    parser.add_argument("--output_dir", type=str, help="Directory to save the results.")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to a local file or Hugging Face dataset identifier.",
        default="allenai/tulu-v2-sft-mixture"
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        help="Tokenizer path or identifier.",
        default=Path(__file__).parent / "tokenizers" / "allenai_eleuther-ai-gpt-neox-20b-pii-special.json",
    )
    parser.add_argument("-s", "--seq-len", type=int, help="Max sequence length.", default=8192)
    parser.add_argument("--eos", type=int, help="EOS token ID.", default=32014)
    parser.add_argument("--pad", type=int, help="PAD token ID.", default=1)
    parser.add_argument("-j", "--num-proc", type=int, help="Number of workers for map/filter.", default=8)
    return parser

if __name__ == "__main__":
    prepare_cli_environment()
    opts = get_parser().parse_args()
    main(opts)
