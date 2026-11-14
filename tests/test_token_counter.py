import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from token_counter import count_call, measure_baseline
from cost_calculator import calculate_cost


class FakeBackend:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "message": {"content": "Fake answer"},
            "prompt_eval_count": 100,
            "eval_count": 20,
            "done": True,
            "done_reason": "stop",
        }


class TokenCounterTests(unittest.TestCase):
    def test_extracts_ollama_counts_and_cost(self):
        fake = FakeBackend()
        result = count_call(fake, "Hello", "Be helpful.")
        self.assertEqual(result["input_tokens"], 100)
        self.assertEqual(result["output_tokens"], 20)
        self.assertAlmostEqual(result["input_cost_usd"], 0.000015)
        self.assertAlmostEqual(result["output_cost_usd"], 0.000012)
        self.assertAlmostEqual(result["cost_usd"], 0.000027)
        self.assertEqual(result["output"], "Fake answer")

    def test_passes_same_generation_settings_to_backend(self):
        fake = FakeBackend()
        count_call(fake, "Hi", "Fixed system.", temperature=0, seed=42, max_tokens=256)
        self.assertEqual(fake.calls[0]["temperature"], 0)
        self.assertEqual(fake.calls[0]["seed"], 42)
        self.assertEqual(fake.calls[0]["max_tokens"], 256)
        self.assertEqual(fake.calls[0]["system_prompt"], "Fixed system.")

    def test_baseline_reads_prompt_eval_count(self):
        fake = FakeBackend()
        baseline = measure_baseline(fake, "Fixed system.")
        self.assertEqual(baseline["baseline_prompt_eval_count"], 100)
        self.assertEqual(baseline["baseline_eval_count"], 20)

    def test_negative_tokens_rejected(self):
        with self.assertRaises(ValueError):
            calculate_cost(-1, 10)


if __name__ == "__main__":
    unittest.main()
