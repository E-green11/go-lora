import unittest

from math_code_genetation.pipeline_utils import (
    clean_code_completion,
    convert_code_feedback_record,
    convert_metamath_record,
    extract_gsm8k_final_answer,
    extract_mbpp_entry_point,
)


class PipelineUtilsTest(unittest.TestCase):
    def test_convert_metamath_record(self):
        record = {
            "type": "GSM_Rephrased",
            "query": "What is 1+1?",
            "original_question": "What is 1+1?",
            "response": "Let's think. The answer is: 2",
        }
        converted = convert_metamath_record(record)
        self.assertEqual(converted["instruction"], "What is 1+1?")
        self.assertEqual(converted["input"], "")
        self.assertIn("The answer is: 2", converted["output"])

    def test_convert_code_feedback_record_emits_one_sample_per_assistant_turn(self):
        record = {
            "id": 1,
            "messages": [
                {"role": "user", "content": "Write foo."},
                {"role": "assistant", "content": "def foo():\n    pass"},
                {"role": "user", "content": "Fix foo."},
                {"role": "assistant", "content": "def foo(x):\n    return x"},
            ],
        }
        converted = convert_code_feedback_record(record)
        self.assertEqual(len(converted), 2)
        self.assertEqual(converted[0]["instruction"], "Write foo.")
        self.assertEqual(converted[0]["input"], "")
        self.assertEqual(converted[1]["instruction"], "Fix foo.")
        self.assertIn("User: Write foo.", converted[1]["input"])
        self.assertIn("Assistant: def foo():", converted[1]["input"])

    def test_extract_gsm8k_final_answer(self):
        self.assertEqual(extract_gsm8k_final_answer("Some reasoning.\nThe answer is: 42"), "42")
        self.assertEqual(extract_gsm8k_final_answer("#### 18"), "18")

    def test_clean_code_completion_removes_fences(self):
        cleaned = clean_code_completion("```python\ndef foo():\n    return 1\n```")
        self.assertEqual(cleaned, "def foo():\n    return 1")

    def test_extract_mbpp_entry_point(self):
        entry = extract_mbpp_entry_point(["assert sort_matrix([[1]]) == [[1]]"])
        self.assertEqual(entry, "sort_matrix")


if __name__ == "__main__":
    unittest.main()
