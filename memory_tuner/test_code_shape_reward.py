import unittest

from memory_tuner.code_shape_reward import compute_score, extract_python


class CodeShapeRewardTest(unittest.TestCase):
    def test_extracts_fenced_python(self):
        self.assertEqual(
            extract_python("text\n```python\nprint(input())\n```\n"),
            "print(input())",
        )

    def test_scores_parseable_io_without_execution(self):
        self.assertEqual(
            compute_score(
                "codecontests",
                "```python\nvalue = input()\nprint(value)\n```",
                "{}",
            ),
            1.0,
        )

    def test_rejects_invalid_python(self):
        self.assertEqual(
            compute_score("codecontests", "```python\nif:\n```", "{}"),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
