import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "commonsense_reasoning" / "experiment" / "collect_eval_average.py"


class CollectEvalAverageCliTest(unittest.TestCase):
    def test_average_only_outputs_numeric_average(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = Path(tmpdir)
            for dataset, score in {
                "boolq": "0.600000",
                "piqa": "0.500000",
            }.items():
                (result_dir / f"{dataset}.txt").write_text(
                    f"test:500/500 | accuracy 300  {score}\n",
                    encoding="utf-8",
                )

            proc = subprocess.run(
                ["python", str(SCRIPT_PATH), "--average-only", str(result_dir)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertEqual(proc.stdout.strip(), "0.550000")


if __name__ == "__main__":
    unittest.main()
