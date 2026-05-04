import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "commonsense_reasoning" / "experiment" / "golora_handoff.py"
TMUX_LAUNCHER = REPO_ROOT / "commonsense_reasoning" / "start_golora_tmux_run.sh"


class GoloraHandoffCliTest(unittest.TestCase):
    def test_start_variant_updates_recovery_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir)
            state_path = docs_dir / "golora-state.json"
            journal_path = docs_dir / "golora-journal.md"
            next_step_path = docs_dir / "golora-next-step.md"

            proc = subprocess.run(
                [
                    "python",
                    str(SCRIPT_PATH),
                    "start-variant",
                    "--state-path",
                    str(state_path),
                    "--journal-path",
                    str(journal_path),
                    "--next-step-path",
                    str(next_step_path),
                    "--run-id",
                    "golora-auto-20260412-010203",
                    "--tmux-session",
                    "golora-auto",
                    "--gpu",
                    "5",
                    "--variant",
                    "full_golora_g2_trigfix_seed42",
                    "--target-score",
                    "0.789537",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(state_path.exists())
            self.assertTrue(journal_path.exists())
            self.assertTrue(next_step_path.exists())

            state_text = state_path.read_text(encoding="utf-8")
            journal_text = journal_path.read_text(encoding="utf-8")
            next_step_text = next_step_path.read_text(encoding="utf-8")

            self.assertIn("full_golora_g2_trigfix_seed42", state_text)
            self.assertIn("\"status\": \"running_variant\"", state_text)
            self.assertIn("golora-auto", state_text)
            self.assertIn("full_golora_g2_trigfix_seed42", journal_text)
            self.assertIn("tmux attach -t golora-auto", next_step_text)

    def test_tmux_launcher_is_wired_for_handoff_files(self):
        launcher_text = TMUX_LAUNCHER.read_text(encoding="utf-8")

        self.assertIn("tmux new-session -d", launcher_text)
        self.assertIn("golora-state.json", launcher_text)
        self.assertIn("golora-journal.md", launcher_text)
        self.assertIn("golora-next-step.md", launcher_text)
        self.assertIn("python experiment/golora_handoff.py", launcher_text)


if __name__ == "__main__":
    unittest.main()
