import argparse
import os
import re
import sys


DATASETS = [
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "ARC-Challenge",
    "ARC-Easy",
    "openbookqa",
]


def extract_last_accuracy(path: str):
    last = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = re.search(r"test:\d+/\d+ \| accuracy \d+\s+([0-9.]+)", line)
            if match:
                last = float(match.group(1))
    return last


def collect_scores(root: str):
    values = []
    lines = []
    for dataset in DATASETS:
        path = os.path.join(root, f"{dataset}.txt")
        if not os.path.exists(path):
            lines.append(f"{dataset}: MISSING")
            continue
        value = extract_last_accuracy(path)
        if value is None:
            lines.append(f"{dataset}: NO_ACCURACY")
            continue
        values.append(value)
        lines.append(f"{dataset}: {value:.6f}")
    return lines, values


def main(root: str, average_only: bool = False):
    lines, values = collect_scores(root)
    if values:
        average = sum(values) / len(values)
        if average_only:
            print(f"{average:.6f}")
            return
        for line in lines:
            print(line)
        print(f"average: {average:.6f}")
    else:
        if average_only:
            raise SystemExit(1)
        for line in lines:
            print(line)
        print("average: N/A")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    parser.add_argument("--average-only", action="store_true")
    args = parser.parse_args()
    main(args.result_dir, average_only=args.average_only)
