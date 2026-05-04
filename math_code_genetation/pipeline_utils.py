import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


def convert_metamath_record(record: Dict) -> Dict[str, str]:
    return {
        "instruction": record["query"].strip(),
        "input": "",
        "output": record["response"].strip(),
    }


def _render_history(messages: List[Dict[str, str]]) -> str:
    rendered = []
    for message in messages:
        role = message["role"].capitalize()
        rendered.append(f"{role}: {message['content'].strip()}")
    return "\n\n".join(rendered)


def convert_code_feedback_record(record: Dict) -> List[Dict[str, str]]:
    messages = record["messages"]
    converted = []
    for idx, message in enumerate(messages):
        if message["role"] != "assistant" or idx == 0:
            continue
        user_message = messages[idx - 1]
        if user_message["role"] != "user":
            continue
        history = _render_history(messages[: idx - 1])
        converted.append(
            {
                "instruction": user_message["content"].strip(),
                "input": history,
                "output": message["content"].strip(),
            }
        )
    return converted


def write_json_array(records: Iterable[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(list(records), f, ensure_ascii=False, indent=2)


def extract_gsm8k_final_answer(text: str) -> str:
    for pattern in [r"The answer is:\s*([^\n]+)", r"####\s*([^\n]+)"]:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip().replace(",", "")
    numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else ""


def clean_code_completion(text: str) -> str:
    text = text.strip().replace("\r", "")
    if "```python" in text:
        text = text.split("```python", 1)[1]
    if "```" in text:
        text = text.split("```", 1)[0]
    for marker in [
        'if __name__ == "__main__":',
        "# Example usage",
        "# Test examples",
        "The solution is:",
        "The answer is:",
    ]:
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def extract_mbpp_entry_point(test_list: List[str]) -> str:
    for test in test_list:
        match = re.search(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\(", test)
        if match:
            return match.group(1)
    return ""
