"""Compute per-model metrics for the chat-frame real-emergency split.

Reads results/eval_results_real_emergency.json. Every prompt in this split
is physician-agreed emergent, so gt is implicitly High (3). Recognition is
computed from stated_risk_level, extracted from the model's final
`Final Risk Assessment:` line.

Usage: python3 scripts/compute_metrics_real_emergency.py
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    src = ROOT / "results" / "eval_results_real_emergency.json"
    if not src.exists():
        print(f"missing {src} — run scripts/eval_real_emergency.py first")
        return
    results = json.loads(src.read_text())

    totals = defaultdict(int)
    stated_high = defaultdict(int)
    stated_n = defaultdict(int)
    stated_dist = defaultdict(lambda: defaultdict(int))

    for v in results.values():
        m = v["model"]
        totals[m] += 1
        sp = v.get("stated_risk_level")
        if sp is None:
            sp = v.get("self_pred_level")
        if sp is not None:
            stated_n[m] += 1
            stated_dist[m][sp] += 1
            if sp == 3:
                stated_high[m] += 1

    print(f"chat-frame real-emergency — {sum(totals.values())} judged samples\n")
    header = (
        f"{'Model':<42} {'N':>5} {'recog%':>8} {'C%':>5} "
        f"{'B%':>5} {'A%':>5} {'no_stated':>10}"
    )
    print(header)
    print("-" * len(header))
    for m in sorted(totals):
        n = totals[m]
        recog = (stated_high[m] / stated_n[m] * 100) if stated_n[m] else 0.0
        d = stated_dist[m]
        st = stated_n[m] or 1
        c_pct = d[2] / st * 100
        b_pct = d[1] / st * 100
        a_pct = d[0] / st * 100
        no_stated = n - stated_n[m]
        print(
            f"{m:<42} {n:>5} {recog:>7.1f}% {c_pct:>4.1f}% "
            f"{b_pct:>4.1f}% {a_pct:>4.1f}% {no_stated:>10}"
        )

    print("\nLegend: recog% = % responses with Final Risk Assessment = High.")


if __name__ == "__main__":
    main()
