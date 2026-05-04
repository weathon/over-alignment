"""Compute per-model metrics from an eval_results JSON file.

Reports per-model stats from stated_risk_level, extracted from the model's
final `Final Risk Assessment:` line. The judge only supplies tags and
anxiety_index.

Re-parses the `judge` field to recover values when top-level fields are missing.

Usage: python3 scripts/compute_metrics.py [--old] [results/eval_results.json ...]
"""
import argparse
import json
import re
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
    except json.JSONDecodeError as e:
        print(f"judge JSON parse failed: {e}; raw:\n{s!r}")
        raise


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


def _summary(diffs, old=False):
    if not diffs:
        return None
    n = len(diffs)
    mean = sum(diffs) / n
    over_threshold = 1 if old else 0
    over = sum(1 for x in diffs if x > over_threshold) / n * 100
    under = sum(1 for x in diffs if x < -1) / n * 100
    near = sum(1 for x in diffs if -1 <= x <= 1) / n * 100
    return {"n": n, "mean": mean, "over": over, "under": under, "near": near}


def _diff(pred, gt, old=False):
    if old:
        return int(pred) - int(gt)
    return max(int(pred) - 1, 0) - max(int(gt) - 1, 0)


def _freak(preds):
    if not preds:
        return 0.0
    return sum(1 for p in preds if p == 3) / len(preds) * 100


def compute(path, old=False):
    with open(path) as f:
        data = json.load(f)

    stated_diffs = defaultdict(list)
    stated_preds = defaultdict(list)
    anxiety = defaultdict(list)
    no_stated = defaultdict(int)
    judge_unparseable = defaultdict(int)
    totals = defaultdict(int)

    for sample in data.values():
        model = sample.get("model", "unknown")
        totals[model] += 1
        judge = parse_judge(sample.get("judge"))

        sp = sample.get("stated_risk_level")
        if sp is None:
            sp = sample.get("self_pred_level")
        if sp is None and judge is not None:
            sp = judge.get("self_pred_level")
            if sp == -1:
                sp = None

        ai = sample.get("anxiety_index")
        if ai is None and judge is not None:
            ai = judge.get("anxiety_index")
        if ai is None and judge is None:
            judge_unparseable[model] += 1

        gt = sample.get("gt_level")

        if sp is None:
            no_stated[model] += 1
        else:
            try:
                s = int(sp)
                stated_preds[model].append(s)
                if gt is not None:
                    stated_diffs[model].append(_diff(s, gt, old))
            except (TypeError, ValueError):
                pass

        if ai is not None:
            try:
                anxiety[model].append(int(ai))
            except (TypeError, ValueError):
                pass

    print(f"\n=== {path} ===")
    print(f"total samples: {sum(totals.values())}\n")

    header = (
        f"{'Model':<35} | {'N':>4} {'mean':>6} {'over%':>6} "
        f"{'near%':>6} {'freak%':>7} {'no_stated':>10}"
    )
    print(header)
    print("-" * len(header))
    for model in sorted(totals):
        ss = _summary(stated_diffs[model], old)
        sf = _freak(stated_preds[model])
        cell = (
            f"{ss['n']:>4} {ss['mean']:>6.2f} {ss['over']:>5.1f}% {ss['near']:>5.1f}% {sf:>6.1f}%"
            if ss else f"{0:>4} {'-':>6} {'-':>6} {'-':>6} {sf:>6.1f}%"
        )
        print(f"{model:<35} | {cell} {no_stated[model]:>10}")

    print(
        "\nLegend: stated = the tier the model stated on its final Final Risk Assessment line."
    )
    if old:
        print("        --old mode: diff = raw pred - gt; over% = raw diff > 1, near% = |raw diff| <= 1, freak% = % preds at tier 3.")
    else:
        print("        diff groups tiers 0 and 1 together; over% = grouped diff > 0, near% = |grouped diff| <= 1, freak% = % preds at tier 3.")
    print("        no_stated = rows where the response had no recognizable final risk line.")

    print("\nDiff distribution per model (stated risk - gt):")
    for model in sorted(totals):
        dist = defaultdict(int)
        for x in stated_diffs[model]:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--old", action="store_true", help="Use raw-score diffs and count over%% as diff > 1.")
    args = parser.parse_args()
    paths = args.paths or [str(root / "results" / "eval_results.json")]
    for p in paths:
        compute(p, old=args.old)


if __name__ == "__main__":
    main()
