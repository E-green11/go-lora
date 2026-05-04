import unittest

from commonsense_reasoning.prompting import (
    build_instruction_text,
    coerce_cli_bool,
    render_training_prompt,
    render_user_prompt,
)


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return f"CHAT::{messages[-1]['role']}::{add_generation_prompt}"


class PromptingTest(unittest.TestCase):
    def test_coerce_cli_bool_handles_false_string(self):
        self.assertFalse(coerce_cli_bool("false"))
        self.assertFalse(coerce_cli_bool("0"))

    def test_coerce_cli_bool_handles_true_string(self):
        self.assertTrue(coerce_cli_bool("true"))
        self.assertTrue(coerce_cli_bool("1"))

    def test_build_instruction_text_appends_input_block(self):
        text = build_instruction_text("Question?", "Context")
        self.assertEqual(text, "Question?\n\nInput:\nContext")

    def test_render_training_prompt_uses_llama3_chat_template(self):
        tokenizer = FakeTokenizer()
        prompt = render_training_prompt(
            {
                "instruction": "Pick one.",
                "input": "",
                "output": "answer3",
            },
            prompt_style="llama3_chat",
            tokenizer=tokenizer,
        )
        self.assertEqual(prompt, "CHAT::assistant::False")
        self.assertEqual(tokenizer.calls[0]["messages"][0]["role"], "user")
        self.assertEqual(tokenizer.calls[0]["messages"][1]["role"], "assistant")
        self.assertFalse(tokenizer.calls[0]["add_generation_prompt"])

    def test_render_user_prompt_uses_generation_prompt_for_llama3_chat(self):
        tokenizer = FakeTokenizer()
        prompt = render_user_prompt(
            {
                "instruction": "Pick one.",
                "input": "Context",
                "output": "",
            },
            prompt_style="llama3_chat",
            tokenizer=tokenizer,
        )
        self.assertEqual(prompt, "CHAT::user::True")
        self.assertEqual(len(tokenizer.calls[0]["messages"]), 1)
        self.assertTrue(tokenizer.calls[0]["add_generation_prompt"])


if __name__ == "__main__":
    unittest.main()
