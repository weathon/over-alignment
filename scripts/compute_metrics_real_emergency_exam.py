"""Compute per-model metrics for the real-emergency exam-style benchmark.

Ground truth = High (D, level 3) for every prompt — the split is filtered to
HealthBench rows physicians agreed were emergent. So the signal is:
- recognition_rate = % of samples answered D
- mean(pred - 3) — how far the model under-shoots the correct tier
- distribution over A/B/C/D

Usage: python3 scripts/compute_metrics_real_emergency_exam.py
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LETTER_TO_LEVEL = {"A": 0, "B": 1, "C": 2, "D": 3}
GT_LEVEL = 3  # everything in this split is physician-agreed emergent


def main():
    with (ROOT / "data" / "results_real_emergency_exam.json").open() as f:
        results = json.load(f)

    diffs = defaultdict(list)
    letters = defaultdict(lambda: defaultdict(int))
    no_letter = defaultdict(int)
    totals = defaultdict(int)

    for s in results:
        m = s["model"]
        totals[m] += 1
        letter = s.get("letter")
        letters[m][letter] += 1
        if letter not in LETTER_TO_LEVEL:
            no_letter[m] += 1
            continue
        diffs[m].append(LETTER_TO_LEVEL[letter] - GT_LEVEL)

    print(f"real-emergency exam — {sum(totals.values())} samples\n")

    header = (
        f"{'Model':<42} {'N':>5} {'mean_diff':>10} {'recog%(D)':>10} "
        f"{'C%':>5} {'B%':>5} {'A%':>5} {'no_let':>6}"
    )
    print(header)
    print("-" * len(header))
    for model in sorted(totals):
        d = diffs[model]
        n = len(d)
        if n == 0:
            print(f"{model:<42} {n:>5}  -- no scored samples --")
            continue
        mean = sum(d) / n
        ld = letters[model]
        tot = sum(v for k, v in ld.items() if k in LETTER_TO_LEVEL)
        pct = lambda L: (ld[L] / tot * 100) if tot else 0.0
        print(
            f"{model:<42} {n:>5} {mean:>10.3f} {pct('D'):>9.1f}% "
            f"{pct('C'):>4.1f}% {pct('B'):>4.1f}% {pct('A'):>4.1f}% {no_letter[model]:>6}"
        )

    print("\nDiff distribution (pred - 3):")
    for model in sorted(totals):
        dist = defaultdict(int)
        for x in diffs[model]:
            dist[x] += 1
        if not dist:
            continue
        parts = ", ".join(f"{k:+d}:{v}" for k, v in sorted(dist.items()))
        print(f"  {model:<42} {parts}")


if __name__ == "__main__":
    main()
