from typing import Any, Dict, Optional


def coerce_cli_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    return bool(value)


def build_instruction_text(instruction: str, input_text: Optional[str] = None) -> str:
    if input_text:
        return f"{instruction}\n\nInput:\n{input_text}"
    return instruction


def _require_chat_template(tokenizer: Any) -> None:
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("tokenizer with apply_chat_template is required for llama3_chat prompt style")


def render_prompt(
    *,
    instruction: str,
    input_text: Optional[str] = None,
    output_text: Optional[str] = None,
    prompt_style: str = "alpaca",
    tokenizer: Any = None,
    add_generation_prompt: bool = False,
) -> str:
    if prompt_style == "llama3_chat":
        _require_chat_template(tokenizer)
        messages = [
            {
                "role": "user",
                "content": build_instruction_text(instruction, input_text),
            }
        ]
        if output_text is not None:
            messages.append({"role": "assistant", "content": output_text})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt and output_text is None,
        )

    if input_text:
        response = output_text or ""
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}
                
                ### Input:
                {input_text}
                
                ### Response:
                {response}"""

    response = output_text or ""
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {instruction}
                
                ### Response:
                {response}"""


def render_training_prompt(data_point: Dict[str, str], *, prompt_style: str = "alpaca", tokenizer: Any = None) -> str:
    return render_prompt(
        instruction=data_point["instruction"],
        input_text=data_point.get("input"),
        output_text=data_point.get("output", ""),
        prompt_style=prompt_style,
        tokenizer=tokenizer,
        add_generation_prompt=False,
    )


def render_user_prompt(data_point: Dict[str, str], *, prompt_style: str = "alpaca", tokenizer: Any = None) -> str:
    return render_prompt(
        instruction=data_point["instruction"],
        input_text=data_point.get("input"),
        output_text=None,
        prompt_style=prompt_style,
        tokenizer=tokenizer,
        add_generation_prompt=True,
    )
