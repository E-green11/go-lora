import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SINGLE_VARIANT_SCRIPT = (
    REPO_ROOT / "commonsense_reasoning" / "run_golora_single_variant.sh"
)
SINGLE_VARIANT_LAUNCHER = (
    REPO_ROOT / "commonsense_reasoning" / "start_golora_single_variant_tmux.sh"
)


class ParallelVariantLauncherTest(unittest.TestCase):
    def test_single_variant_runner_uses_isolated_variant_spec_and_outputs(self):
        script_text = SINGLE_VARIANT_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('VARIANT_SPEC="${VARIANT_SPEC:?VARIANT_SPEC is required}"', script_text)
        self.assertIn('BASE_MODEL="${BASE_MODEL:-./llama-7b-hf}"', script_text)
        self.assertNotIn('BASE_MODEL="./llama-7b-hf"', script_text)
        self.assertIn('MODEL_NAME="${MODEL_NAME:-LLaMA-7B}"', script_text)
        self.assertIn('ADAPTER_NAME="${ADAPTER_NAME:-DoRA}"', script_text)
        self.assertIn('TARGET_MODULES="${TARGET_MODULES:-[\\"q_proj\\", \\"k_proj\\", \\"v_proj\\", \\"up_proj\\", \\"down_proj\\"]}"', script_text)
        self.assertIn('PROMPT_STYLE="${PROMPT_STYLE:-alpaca}"', script_text)
        self.assertIn('TRAIN_ON_INPUTS="${TRAIN_ON_INPUTS:-true}"', script_text)
        self.assertIn('OUT_ROOT="${OUT_ROOT:-./finetuned_result/llama7_r_dora_ablation_parallel}"', script_text)
        self.assertIn('FORMAL_RESULTS="$OUT_ROOT/formal_results.tsv"', script_text)
        self.assertIn('weights_dir="$(pick_weights_path "$outdir")"', script_text)
        self.assertIn('MODEL_NAME="$MODEL_NAME" ADAPTER_NAME="$ADAPTER_NAME" BASE_MODEL="$BASE_MODEL" PROMPT_STYLE="$PROMPT_STYLE" bash ./llama_7B_Dora_eval.sh "$weights_dir" "$GPU"', script_text)
        self.assertIn('python experiment/golora_handoff.py start-variant', script_text)
        self.assertIn('python experiment/golora_handoff.py finish-variant', script_text)
        self.assertIn('tee "$weights_dir/average_summary.txt" >/dev/stderr', script_text)
        self.assertIn('python experiment/collect_eval_average.py --average-only "$weights_dir"', script_text)
        self.assertNotIn('VARIANTS=(', script_text)

    def test_single_variant_tmux_launcher_wires_session_docs_and_out_root(self):
        launcher_text = SINGLE_VARIANT_LAUNCHER.read_text(encoding="utf-8")

        self.assertIn('VARIANT_SPEC="${2:?variant spec is required}"', launcher_text)
        self.assertIn('SESSION_NAME="${SESSION_NAME:-golora-', launcher_text)
        self.assertIn('DOCS_DIR="${DOCS_DIR:-$ROOT_DIR/../docs/codex/', launcher_text)
        self.assertIn('OUT_ROOT="${OUT_ROOT:-./finetuned_result/llama7_r_dora_ablation_parallel/', launcher_text)
        self.assertIn("tmux new-session -d -s \"$SESSION_NAME\"", launcher_text)
        self.assertIn("export VARIANT_SPEC=", launcher_text)
        self.assertIn("BASE_MODEL=", launcher_text)
        self.assertIn("MODEL_NAME=", launcher_text)
        self.assertIn("ADAPTER_NAME=", launcher_text)
        self.assertIn("TARGET_MODULES=", launcher_text)
        self.assertIn("PROMPT_STYLE=", launcher_text)
        self.assertIn("TRAIN_ON_INPUTS=", launcher_text)
        self.assertIn("DORA_GEO_THETA_CLAMP_MODE=", launcher_text)
        self.assertIn("DORA_GEO_THETA_SOFTBAND_START=", launcher_text)
        self.assertIn("DORA_GEO_ANCHOR_MODE=", launcher_text)
        self.assertIn("DORA_GEO_ANCHOR_MOMENTUM=", launcher_text)
        self.assertIn("bash ./run_golora_single_variant.sh", launcher_text)


if __name__ == "__main__":
    unittest.main()
