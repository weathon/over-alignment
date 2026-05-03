"""Judge over-caution responses for the reasoning-effort sweep.

Thin wrapper around eval.py: reuses the same judge prompt, schema, blacklist
filter, gt matching, self-risk parsing, and resume logic, but swaps paths to
the reasoning-sweep files.

In:  results/results_reasoning_sweep.json
Out: results/eval_results_reasoning_sweep.json
"""

from pathlib import Path

import eval as base

ROOT = Path(__file__).resolve().parent.parent

base.SRC = ROOT / "results" / "results_reasoning_sweep.json"
base.OUT = ROOT / "results" / "eval_results_reasoning_sweep.json"
base.REVERSE_BENCH_RESULTS = False
base.OUT.parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    base.main()
