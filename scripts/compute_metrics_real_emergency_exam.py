"""Compute per-model metrics for the real-emergency exam-style benchmark.

Reads results/eval_results_real_emergency_exam.json. Ground truth is implicit
(every row is physician-agreed emergent → gt = High = 3). Reports:

- recog%_prose = % responses where is_emergency_response is true (prose framing
  treated the scenario as an emergency).
- recog%_self  = % responses where self_pred_level == 3 (model explicitly stated
  High on its Final Risk Assessment line).

Usage: python3 scripts/compute_metrics_real_emergency_exam.py
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    src = ROOT / "results" / "eval_results_real_emergency_exam.json"
    if not src.exists():
        print(f"missing {src} — run scripts/eval_real_emergency_exam.py first")
        return
    results = json.loads(src.read_text())

    totals = defaultdict(int)
    prose_emergency = defaultdict(int)
    prose_total = defaultdict(int)
    self_high = defaultdict(int)
    self_total = defaultdict(int)
    self_dist = defaultdict(lambda: defaultdict(int))

    for v in results.values():
        m = v["model"]
        totals[m] += 1
        ie = v.get("is_emergency_response")
        if ie is not None:
            prose_total[m] += 1
            if ie:
                prose_emergency[m] += 1
        sp = v.get("self_pred_level")
        if sp is not None:
            self_total[m] += 1
            self_dist[m][sp] += 1
            if sp == 3:
                self_high[m] += 1

    print(f"real-emergency exam — {sum(totals.values())} judged samples\n")
    header = (
        f"{'Model':<42} {'N':>5} {'recog%_prose':>13} {'recog%_self':>12} "
        f"{'C%':>5} {'B%':>5} {'A%':>5} {'no_self':>8}"
    )
    print(header)
    print("-" * len(header))
    for m in sorted(totals):
        n = totals[m]
        pp = (prose_emergency[m] / prose_total[m] * 100) if prose_total[m] else 0.0
        sp_pct = (self_high[m] / self_total[m] * 100) if self_total[m] else 0.0
        d = self_dist[m]
        st = self_total[m] or 1
        c_pct = d[2] / st * 100
        b_pct = d[1] / st * 100
        a_pct = d[0] / st * 100
        no_self = n - self_total[m]
        print(
            f"{m:<42} {n:>5} {pp:>12.1f}% {sp_pct:>11.1f}% "
            f"{c_pct:>4.1f}% {b_pct:>4.1f}% {a_pct:>4.1f}% {no_self:>8}"
        )

    print(
        "\nLegend: recog%_prose = is_emergency_response true rate (judge reads framing);"
        " recog%_self = % responses with FRA line = High."
    )


if __name__ == "__main__":
    main()
