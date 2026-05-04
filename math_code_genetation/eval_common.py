import json
import multiprocessing as mp
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LlamaTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "commonsense_reasoning" / "peft" / "src"))
from peft import PeftModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_peft_model(base_model: str, lora_weights: str, load_8bit: bool = False):
    if "Llama-2" in base_model or "llama-7b-hf" in base_model:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = 0

    if DEVICE == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, lora_weights)
        if not load_8bit:
            model.half()

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 50,
    num_beams: int = 1,
    max_new_tokens: int = 256,
) -> List[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)
    config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        do_sample=temperature > 0,
    )
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=config,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=False,
        )
    suffix = output.sequences[:, input_ids.shape[1] :]
    return tokenizer.batch_decode(suffix, skip_special_tokens=True)


def build_math_prompt(question: str) -> str:
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{question}\n\n### Response: Let's think step by step."
    )


def build_code_prompt(problem_text: str) -> str:
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"Create a Python script for this problem:\n{problem_text}\n\n"
        "### Response:"
    )


def normalize_math_string(text: str) -> str:
    return re.sub(r"\s+", "", text.replace("\\left", "").replace("\\right", ""))


def extract_boxed_answer(text: str) -> str:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return ""
    right = text.find("}", idx)
    if right < 0:
        return ""
    chunk = text[idx : right + 1]
    return chunk.replace("\\boxed{", "").rstrip("}")


def batched(items: List, batch_size: int) -> Iterable[List]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def write_jsonl(records: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_python_tests(code: str, tests: List[str], timeout: float = 3.0) -> bool:
    def _worker(queue: mp.Queue, src: str, assertions: List[str]):
        ns = {}
        try:
            exec(src, ns, ns)
            for test in assertions:
                exec(test, ns, ns)
            queue.put(True)
        except Exception:
            queue.put(False)

    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_worker, args=(queue, code, tests))
    proc.start()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False
    return bool(queue.get()) if not queue.empty() else False
