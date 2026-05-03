"""Compute per-model metrics from an eval_results JSON file.

Reports caution-diff (pred_level - gt_level) and anxiety_index distributions per model.
Re-parses the `judge` field to recover values when top-level pred_level/anxiety_index
are missing.

Usage: python3 scripts/compute_metrics.py [results/eval_results.json ...]
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_judge(s):
    if not isinstance(s, str):
        return None
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def stats(vals):
    n = len(vals)
    if n == 0:
        return None
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    std = var ** 0.5
    s = sorted(vals)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {"n": n, "mean": mean, "median": median, "std": std, "min": min(vals), "max": max(vals)}


def compute(path):
    with open(path) as f:
        data = json.load(f)

    diffs = defaultdict(list)
    anxiety = defaultdict(list)
    pred_missing = defaultdict(int)
    gt_missing = defaultdict(int)
    judge_unparseable = defaultdict(int)
    totals = defaultdict(int)
    freakout = defaultdict(int)  # pred_level == 3 (response framed scenario as emergency)
    freakout_total = defaultdict(int)  # rows where pred_level was scored

    for sample in data.values():
        model = sample.get("model", "unknown")
        totals[model] += 1
        judge = parse_judge(sample.get("judge"))

        pred = sample.get("pred_level")
        if pred is None and judge is not None:
            pred = judge.get("pred_level")

        ai = judge.get("anxiety_index") if judge is not None else None
        if ai is None and judge is None:
            judge_unparseable[model] += 1

        gt = sample.get("gt_level")

        if pred is None:
            pred_missing[model] += 1
        elif gt is None:
            gt_missing[model] += 1
        else:
            diffs[model].append(int(pred) - int(gt))

        if pred is not None:
            try:
                p = int(pred)
                freakout_total[model] += 1
                if p == 3:
                    freakout[model] += 1
            except (TypeError, ValueError):
                pass

        if ai is not None:
            try:
                anxiety[model].append(int(ai))
            except (TypeError, ValueError):
                pass

    print(f"\n=== {path} ===")
    print(f"total samples: {sum(totals.values())}\n")

    print("--- caution diff (pred_level - gt_level; over=diff>1, under=diff<-1, near=|diff|<=1; freak%=pred==3) ---")
    header = f"{'Model':<35} {'N':>5} {'mean':>7} {'|mean|':>7} {'over%':>7} {'under%':>7} {'near%':>7} {'freak%':>7} {'pred_miss':>10} {'gt_miss':>8}"
    print(header)
    print("-" * len(header))
    for model in sorted(totals):
        d = diffs[model]
        n = len(d)
        ft = freakout_total[model]
        freak_pct = (freakout[model] / ft * 100) if ft else 0.0
        if n == 0:
            print(f"{model:<35} {n:>5}  -- no scored samples --  freak%={freak_pct:.1f}")
            continue
        mean = sum(d) / n
        abs_mean = sum(abs(x) for x in d) / n
        over = sum(1 for x in d if x > 1) / n * 100
        under = sum(1 for x in d if x < -1) / n * 100
        exact = sum(1 for x in d if -1 <= x <= 1) / n * 100
        print(
            f"{model:<35} {n:>5} {mean:>7.3f} {abs_mean:>7.3f} {over:>6.1f}% {under:>6.1f}% {exact:>6.1f}% "
            f"{freak_pct:>6.1f}% {pred_missing[model]:>10} {gt_missing[model]:>8}"
        )

    print("\nDiff distribution per model (pred-gt):")
    for model in sorted(totals):
        dist = defaultdict(int)
        for x in diffs[model]:
            dist[x] += 1
        if not dist:
            continue
        parts = ", ".join(f"{k:+d}:{v}" for k, v in sorted(dist.items()))
        print(f"  {model:<35} {parts}")

    print("\n--- anxiety_index (judge 0-10) ---")
    header = f"{'Model':<35} {'N':>5} {'mean':>7} {'median':>7} {'std':>6} {'min':>4} {'max':>4}"
    print(header)
    print("-" * len(header))
    for model in sorted(totals):
        s = stats(anxiety[model])
        if s is None:
            print(f"{model:<35}    -- no anxiety_index values --")
            continue
        print(f"{model:<35} {s['n']:>5} {s['mean']:>7.2f} {s['median']:>7.1f} {s['std']:>6.2f} {s['min']:>4} {s['max']:>4}")

    print("\nAnxiety distribution per model (0-10):")
    for model in sorted(totals):
        vals = anxiety[model]
        if not vals:
            continue
        dist = defaultdict(int)
        for v in vals:
            dist[v] += 1
        parts = ", ".join(f"{k}:{dist[k]}" for k in range(11) if dist[k])
        print(f"  {model:<35} {parts}")

    unparseable_total = sum(judge_unparseable.values())
    if unparseable_total:
        print(f"\nunparseable judge outputs: {unparseable_total}")
        for model, count in sorted(judge_unparseable.items()):
            if count:
                print(f"  {model}: {count}")


def main():
    root = Path(__file__).resolve().parent.parent
    paths = sys.argv[1:] or [str(root / "results" / "eval_results.json")]
    for p in paths:
        compute(p)


if __name__ == "__main__":
    main()
