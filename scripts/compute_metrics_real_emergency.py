"""Compute per-model metrics for the chat-frame real-emergency split.

Reads results/eval_results_real_emergency.json. Every prompt in this split
is physician-agreed emergent, so gt is implicitly High (3). Reports:

- recog%_prose = % responses where is_emergency_response is true (judge read
  the prose framing as directing the user to emergency care).
- recog%_self  = % responses where self_pred_level == 3 (model explicitly
  stated High on its Final Risk Assessment line).

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
    prose_recog = defaultdict(int)
    prose_n = defaultdict(int)
    self_high = defaultdict(int)
    self_n = defaultdict(int)
    self_dist = defaultdict(lambda: defaultdict(int))

    for v in results.values():
        m = v["model"]
        totals[m] += 1
        ie = v.get("is_emergency_response")
        if ie is not None:
            prose_n[m] += 1
            if ie:
                prose_recog[m] += 1
        sp = v.get("self_pred_level")
        if sp is not None:
            self_n[m] += 1
            self_dist[m][sp] += 1
            if sp == 3:
                self_high[m] += 1

    print(f"chat-frame real-emergency — {sum(totals.values())} judged samples\n")
    header = (
        f"{'Model':<42} {'N':>5} {'recog%_prose':>13} {'recog%_self':>12} "
        f"{'C%':>5} {'B%':>5} {'A%':>5} {'no_self':>8}"
    )
    print(header)
    print("-" * len(header))
    for m in sorted(totals):
        n = totals[m]
        pp = (prose_recog[m] / prose_n[m] * 100) if prose_n[m] else 0.0
        sp_pct = (self_high[m] / self_n[m] * 100) if self_n[m] else 0.0
        d = self_dist[m]
        st = self_n[m] or 1
        c_pct = d[2] / st * 100
        b_pct = d[1] / st * 100
        a_pct = d[0] / st * 100
        no_self = n - self_n[m]
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
