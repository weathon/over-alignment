"""Judge real-emergency responses for the reasoning-effort sweep.

Thin wrapper around eval_real_emergency.py: reuses the judge prompt, schema,
parsing, and resume logic, but swaps the input/output paths to the
reasoning-sweep files.

In:  results/results_real_emergency_reasoning_sweep.json
Out: results/eval_results_real_emergency_reasoning_sweep.json
"""

from pathlib import Path

import eval_real_emergency as base

ROOT = Path(__file__).resolve().parent.parent

base.SRC = ROOT / "results" / "results_real_emergency_reasoning_sweep.json"
base.OUT = ROOT / "results" / "eval_results_real_emergency_reasoning_sweep.json"
base.OUT.parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    base.main()
