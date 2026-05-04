import argparse
import json
import re
from pathlib import Path

from human_eval.data import read_problems
from human_eval.evaluation import evaluate_functional_correctness

try:
    from math_code_genetation.eval_common import (
        batched,
        build_code_prompt,
        generate_batch,
        load_peft_model,
        read_jsonl,
        run_python_tests,
        write_jsonl,
    )
    from math_code_genetation.pipeline_utils import clean_code_completion, extract_mbpp_entry_point
except ImportError:
    from eval_common import batched, build_code_prompt, generate_batch, load_peft_model, read_jsonl, run_python_tests, write_jsonl
    from pipeline_utils import clean_code_completion, extract_mbpp_entry_point


def evaluate_humaneval(model, tokenizer, dataset_path: Path, samples_path: Path, batch_size: int, max_new_tokens: int):
    problems = read_problems(str(dataset_path))
    task_ids = sorted(problems.keys())
    prompts = [build_code_prompt(problems[task_id]["prompt"]) for task_id in task_ids]
    completions = []
    for prompt_batch in batched(prompts, batch_size):
        outputs = generate_batch(model, tokenizer, prompt_batch, max_new_tokens=max_new_tokens)
        completions.extend(outputs)

    records = []
    for task_id, completion in zip(task_ids, completions):
        records.append({"task_id": task_id, "completion": clean_code_completion(completion)})
    write_jsonl(records, samples_path)
    result = evaluate_functional_correctness(
        str(samples_path),
        k=[1],
        n_workers=4,
        timeout=3.0,
        problem_file=str(dataset_path),
    )
    return float(result["pass@1"]), records


def evaluate_mbpp(model, tokenizer, dataset_path: Path, output_path: Path, batch_size: int, max_new_tokens: int):
    rows = read_jsonl(dataset_path)
    prompts = []
    for row in rows:
        prompt = f"{row['text']}\nTest examples:\n" + "\n".join(row["test_list"])
        prompts.append(build_code_prompt(prompt))

    completions = []
    for prompt_batch in batched(prompts, batch_size):
        outputs = generate_batch(model, tokenizer, prompt_batch, max_new_tokens=max_new_tokens)
        completions.extend(outputs)

    results = []
    correct = 0
    for row, completion in zip(rows, completions):
        cleaned = clean_code_completion(completion)
        entry_point = extract_mbpp_entry_point(row["test_list"])
        tests = list(row["test_list"]) + list(row.get("challenge_test_list", []))
        passed = bool(entry_point) and run_python_tests(cleaned, tests, timeout=3.0)
        correct += int(passed)
        results.append(
            {
                "task_id": row["task_id"],
                "entry_point": entry_point,
                "completion": cleaned,
                "correct": passed,
            }
        )
    write_jsonl(results, output_path)
    return correct / max(len(results), 1), results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_weights", required=True)
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    model, tokenizer = load_peft_model(args.base_model, args.lora_weights)
    if args.dataset == "humaneval":
        score, results = evaluate_humaneval(model, tokenizer, args.dataset_path, args.output_path, args.batch_size, args.max_new_tokens)
        metric_name = "pass@1"
    else:
        score, results = evaluate_mbpp(model, tokenizer, args.dataset_path, args.output_path, args.batch_size, args.max_new_tokens)
        metric_name = "pass@1"

    print(json.dumps({"dataset": args.dataset, metric_name: score, "num_examples": len(results)}, indent=2))


if __name__ == "__main__":
    main()
