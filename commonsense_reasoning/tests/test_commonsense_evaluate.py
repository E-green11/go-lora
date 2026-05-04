import unittest

from commonsense_reasoning.commonsense_evaluate import extract_primary_response


class CommonsenseEvaluateParsingTest(unittest.TestCase):
    def test_extract_primary_response_prefers_first_response_block(self):
        decoded = """Below is an instruction that describes a task.

### Instruction:
Please choose the correct ending.

### Response:
the correct answer is ending4

Answer format: ending1/ending2/ending3/ending4

### Response:
the correct answer is ending
"""

        self.assertEqual(
            extract_primary_response(decoded),
            "the correct answer is ending4\n\nAnswer format: ending1/ending2/ending3/ending4",
        )

    def test_extract_primary_response_stops_before_followup_section(self):
        decoded = """Below is an instruction that describes a task.

### Instruction:
Please answer true or false.

### Response:
the correct answer is true

### Explanation:
because ...
"""

        self.assertEqual(
            extract_primary_response(decoded),
            "the correct answer is true",
        )


if __name__ == "__main__":
    unittest.main()
