"""Judge model responses on the exam-framed real-emergency split.

Thin wrapper around eval_real_emergency.py: the binary emergency-recognition
judge is identical, only the input/output paths differ.

In:  results/results_real_emergency_exam.json
Out: results/eval_results_real_emergency_exam.json
"""

from pathlib import Path

import eval_real_emergency as base

ROOT = Path(__file__).resolve().parent.parent

base.SRC = ROOT / "results" / "results_real_emergency_exam.json"
base.OUT = ROOT / "results" / "eval_results_real_emergency_exam.json"
base.OUT.parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    base.main()
