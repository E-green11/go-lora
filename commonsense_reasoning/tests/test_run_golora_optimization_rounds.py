import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "commonsense_reasoning" / "run_golora_optimization_rounds.sh"


class RunGoloraOptimizationScriptTest(unittest.TestCase):
    def test_script_uses_full_epoch_training_not_short_max_steps(self):
        script_text = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("--num_epochs 2", script_text)
        self.assertNotIn("--max_steps", script_text)
        self.assertNotIn("geo_g2_dropout_yuanban_seed42", script_text)
        self.assertIn("full_golora_g2_trigfix_seed42", script_text)
        self.assertIn("bash ./llama_7B_Dora_eval.sh", script_text)
        self.assertIn("best_model_checkpoint", script_text)
        self.assertNotIn("PROXY_RESULTS=", script_text)
        self.assertNotIn("extract_proxy_eval_loss", script_text)
        self.assertIn("python experiment/golora_handoff.py", script_text)
        self.assertIn("golora-state.json", script_text)
        self.assertIn("golora-journal.md", script_text)
        self.assertIn("golora-next-step.md", script_text)
        self.assertIn('tee "$weights_dir/average_summary.txt" >/dev/stderr', script_text)
        self.assertIn('python experiment/collect_eval_average.py --average-only "$weights_dir"', script_text)


if __name__ == "__main__":
    unittest.main()
