import argparse
import json
from pathlib import Path

try:
    from math_code_genetation.pipeline_utils import (
        convert_code_feedback_record,
        convert_metamath_record,
        write_json_array,
    )
except ImportError:
    from pipeline_utils import convert_code_feedback_record, convert_metamath_record, write_json_array


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_metamath(source: Path, output: Path) -> None:
    records = [convert_metamath_record(record) for record in load_jsonl(source)]
    write_json_array(records, output)
    print(f"MetaMathQA -> {output} ({len(records)} samples)")


def build_codefeedback(source: Path, output: Path) -> None:
    records = []
    for record in load_jsonl(source):
        records.extend(convert_code_feedback_record(record))
    write_json_array(records, output)
    print(f"Code-Feedback -> {output} ({len(records)} samples)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    args = parser.parse_args()

    build_metamath(
        args.data_dir / "MetaMathQA.jsonl",
        args.data_dir / "metamathqa_train.json",
    )
    build_codefeedback(
        args.data_dir / "Code-Feedback.jsonl",
        args.data_dir / "codefeedback_train.json",
    )


if __name__ == "__main__":
    main()
