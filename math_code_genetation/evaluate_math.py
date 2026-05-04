import argparse
import json
import re
from pathlib import Path

try:
    from math_code_genetation.eval_common import (
        batched,
        build_math_prompt,
        extract_boxed_answer,
        generate_batch,
        load_peft_model,
        normalize_math_string,
        write_jsonl,
    )
    from math_code_genetation.pipeline_utils import extract_gsm8k_final_answer
except ImportError:
    from eval_common import batched, build_math_prompt, extract_boxed_answer, generate_batch, load_peft_model, normalize_math_string, write_jsonl
    from pipeline_utils import extract_gsm8k_final_answer


def evaluate_gsm8k(model, tokenizer, dataset_path: Path, batch_size: int, max_new_tokens: int):
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    prompts = [build_math_prompt(row["question"]) for row in rows]
    predictions = []
    for prompt_batch in batched(prompts, batch_size):
        outputs = generate_batch(model, tokenizer, prompt_batch, max_new_tokens=max_new_tokens)
        predictions.extend(outputs)

    results = []
    correct = 0
    for row, output in zip(rows, predictions):
        gold = extract_gsm8k_final_answer(row["answer"])
        pred = extract_gsm8k_final_answer(output)
        is_correct = pred == gold
        correct += int(is_correct)
        results.append({"question": row["question"], "gold": gold, "prediction": pred, "output": output, "correct": is_correct})
    return correct / max(len(results), 1), results


def evaluate_math(model, tokenizer, dataset_path: Path, batch_size: int, max_new_tokens: int):
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    prompts = [build_math_prompt(row["instruction"]) for row in rows]
    predictions = []
    for prompt_batch in batched(prompts, batch_size):
        outputs = generate_batch(model, tokenizer, prompt_batch, max_new_tokens=max_new_tokens)
        predictions.extend(outputs)

    results = []
    correct = 0
    for row, output in zip(rows, predictions):
        gold = extract_boxed_answer(row["output"]) or row["output"]
        pred = extract_boxed_answer(output)
        if not pred:
            tail = re.search(r"The answer is:\s*([^\n]+)", output)
            pred = tail.group(1).strip() if tail else output.strip().splitlines()[-1]
        is_correct = normalize_math_string(pred) == normalize_math_string(gold)
        correct += int(is_correct)
        results.append({"instruction": row["instruction"], "gold": gold, "prediction": pred, "output": output, "correct": is_correct})
    return correct / max(len(results), 1), results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gsm8k", "MATH"], required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_weights", required=True)
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    model, tokenizer = load_peft_model(args.base_model, args.lora_weights)
    if args.dataset == "gsm8k":
        accuracy, results = evaluate_gsm8k(model, tokenizer, args.dataset_path, args.batch_size, args.max_new_tokens)
    else:
        accuracy, results = evaluate_math(model, tokenizer, args.dataset_path, args.batch_size, args.max_new_tokens)

    write_jsonl(results, args.output_path)
    print(json.dumps({"dataset": args.dataset, "accuracy": accuracy, "num_examples": len(results)}, indent=2))


if __name__ == "__main__":
    main()
