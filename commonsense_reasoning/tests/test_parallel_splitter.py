import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITTER_SCRIPT = (
    REPO_ROOT / "commonsense_reasoning" / "split_remaining_variants_after_first.sh"
)


class ParallelSplitterScriptTest(unittest.TestCase):
    def test_splitter_waits_for_first_result_then_launches_gpu5_and_gpu7(self):
        script_text = SPLITTER_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('CURRENT_SESSION="${CURRENT_SESSION:-golora-auto}"', script_text)
        self.assertIn('CURRENT_RESULTS="${CURRENT_RESULTS:-./finetuned_result/llama7_r_dora_ablation_autoruns/formal_results.tsv}"', script_text)
        self.assertIn('GPU5_SPEC="${GPU5_SPEC:-full_golora_g2_trigfix_theta0p20_seed42|0.20|False|100|true|true}"', script_text)
        self.assertIn('GPU7_SPEC="${GPU7_SPEC:-full_golora_g2_trigfix_theta0p10_seed42|0.10|False|100|true|true}"', script_text)
        self.assertIn('tmux kill-session -t "$CURRENT_SESSION"', script_text)
        self.assertIn('bash ./start_golora_single_variant_tmux.sh 5 "$GPU5_SPEC"', script_text)
        self.assertIn('bash ./start_golora_single_variant_tmux.sh 7 "$GPU7_SPEC"', script_text)


if __name__ == "__main__":
    unittest.main()
