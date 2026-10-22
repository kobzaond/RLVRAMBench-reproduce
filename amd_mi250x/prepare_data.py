#!/usr/bin/env python3
"""Prepare GSM8K and MATH parquet files in VERL's standard RL format.

Mirrors VERL's examples/data_preprocess/{gsm8k,math_dataset}.py so that the
MI250X runs use the same prompt templates and rule rewards as the stock
VERL examples.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import datasets
import pyarrow.parquet as pq

GSM8K_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'
MATH_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def last_boxed_only_string(text: str) -> str | None:
    index = text.rfind("\\boxed")
    if index < 0:
        index = text.rfind("\\fbox")
        if index < 0:
            return None
    depth = 0
    for position in range(index, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[index:position + 1]
    return None


def remove_boxed(text: str) -> str:
    if text.startswith("\\boxed "):
        return text[len("\\boxed "):]
    left = "\\boxed{"
    assert text.startswith(left) and text.endswith("}"), text
    return text[len(left):-1]


def write(data: datasets.Dataset, path: Path) -> None:
    # Drop the Hugging Face feature metadata: older `datasets` releases in
    # the training container cannot parse feature types written by newer ones.
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(data.data.table.replace_schema_metadata(None), str(path))


def gsm8k(output: Path) -> None:
    source = "openai/gsm8k"
    loaded = datasets.load_dataset(source, "main")

    def convert(split):
        def row(example, index):
            answer = re.search(r"#### (\-?[0-9\.\,]+)", example["answer"]).group(1).replace(",", "")
            return {
                "data_source": source,
                "prompt": [{"role": "user", "content": example["question"] + " " + GSM8K_INSTRUCTION}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"split": split, "index": index},
            }
        return row

    for split, name in (("train", "train"), ("test", "test")):
        data = loaded[split].map(convert(split), with_indices=True, remove_columns=loaded[split].column_names)
        write(data.flatten_indices(), output / "gsm8k" / f"{name}.parquet")


def math(output: Path) -> None:
    source = "DigitalLearningGmbH/MATH-lighteval"
    loaded = datasets.load_dataset(source, trust_remote_code=True)

    def convert(split):
        def row(example, index):
            return {
                "data_source": source,
                "prompt": [{"role": "user", "content": example["problem"] + " " + MATH_INSTRUCTION}],
                "ability": "math",
                "reward_model": {"style": "rule",
                                 "ground_truth": remove_boxed(last_boxed_only_string(example["solution"]))},
                "extra_info": {"split": split, "index": index},
            }
        return row

    for split in ("train", "test"):
        # A few solutions carry no boxed answer and cannot be rule-scored.
        boxed = loaded[split].filter(lambda example: last_boxed_only_string(example["solution"]) is not None)
        data = boxed.map(convert(split), with_indices=True, remove_columns=boxed.column_names)
        write(data.flatten_indices(), output / "math" / f"{split}.parquet")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gsm8k(args.output)
    math(args.output)


if __name__ == "__main__":
    main()
