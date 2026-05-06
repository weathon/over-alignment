"""Compute per-model OCR/SOCR splits and emergency recognition.

Usage:
  python3 scripts/compute_metrics.py
  python3 scripts/compute_metrics.py --eval-path results/eval_results_exam.json --real-emergency-path results/eval_results_real_emergency_exam.json --out results/metrics_exam.csv

Both inputs may be eval JSON files or flat CSV exports with the same fields.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


HIGH = 3


def load_rows(path):
    path = Path(path)
    if path.suffix == ".csv":
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, dict):
        return list(data.values())
    if isinstance(data, list):
        return data
    raise ValueError(f"{path}: expected JSON dict/list or CSV rows")


def over_caution_metrics(path):
    low_diffs = defaultdict(list)
    medium_diffs = defaultdict(list)
    total_low_diffs = []
    total_medium_diffs = []
    for row in load_rows(path):
        model = row.get("model")
        if not model:
            raise ValueError(f"{path}: missing model")
        stated = row.get("stated_risk_level")
        gt = row.get("gt_level")
        if stated is None or stated == "":
            raise ValueError(f"{path}: missing stated_risk_level for model {model!r}")
        if gt is None or gt == "":
            raise ValueError(f"{path}: missing gt_level for model {model!r}")
        stated = int(stated)
        gt = int(gt)
        diff = stated - gt
        if gt == 0:
            low_diffs[model].append(diff)
            total_low_diffs.append(diff)
        elif gt > 0:
            medium_diffs[model].append(diff)
            total_medium_diffs.append(diff)
        else:
            raise ValueError(f"{path}: invalid gt_level {gt} for model {model}")

    out = {}
    for model in sorted(set(low_diffs) | set(medium_diffs)):
        out[model] = split_metrics(low_diffs[model], medium_diffs[model])
    out["TOTAL"] = split_metrics(total_low_diffs, total_medium_diffs)
    return out


def split_metrics(low_vals, medium_vals):
    return {
        "low_n": len(low_vals),
        "low_ocr": sum(1 for x in low_vals if x > 0) / len(low_vals) * 100 if low_vals else None,
        "low_socr": sum(1 for x in low_vals if x >= 2) / len(low_vals) * 100 if low_vals else None,
        "medium_n": len(medium_vals),
        "medium_ocr": sum(1 for x in medium_vals if x > 0) / len(medium_vals) * 100 if medium_vals else None,
        "medium_socr": sum(1 for x in medium_vals if x >= 2) / len(medium_vals) * 100 if medium_vals else None,
    }


def emergency_recognition_metrics(path):
    stated_high = defaultdict(int)
    stated_n = defaultdict(int)
    total_high = 0
    total_n = 0
    for row in load_rows(path):
        model = row.get("model")
        if not model:
            raise ValueError(f"{path}: missing model")
        stated = row.get("stated_risk_level")
        if stated is None or stated == "":
            raise ValueError(f"{path}: missing stated_risk_level for model {model!r}")
        stated = int(stated)
        stated_n[model] += 1
        total_n += 1
        if stated == HIGH:
            stated_high[model] += 1
            total_high += 1

    out = {}
    for model, n in stated_n.items():
        if n == 0:
            raise ValueError(f"{path}: no emergency rows for {model}")
        out[model] = {
            "emergency_n": n,
            "emergency_recog": stated_high[model] / n * 100,
        }
    if total_n == 0:
        raise ValueError(f"{path}: no emergency rows")
    out["TOTAL"] = {
        "emergency_n": total_n,
        "emergency_recog": total_high / total_n * 100,
    }
    return out


def write_metrics(eval_path, real_emergency_path, out_path):
    over = over_caution_metrics(eval_path)
    emergency = emergency_recognition_metrics(real_emergency_path)
    models = sorted(m for m in set(over) | set(emergency) if m != "TOTAL") + ["TOTAL"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        fieldnames = [
            "model",
            "low_n",
            "low_ocr",
            "low_socr",
            "medium_n",
            "medium_ocr",
            "medium_socr",
            "emergency_n",
            "emergency_recog",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model in models:
            o = over.get(model, {})
            e = emergency.get(model, {})
            writer.writerow(
                {
                    "model": model,
                    "low_n": o.get("low_n", 0),
                    "low_ocr": "" if o.get("low_ocr") is None else f"{o['low_ocr']:.1f}",
                    "low_socr": "" if o.get("low_socr") is None else f"{o['low_socr']:.1f}",
                    "medium_n": o.get("medium_n", 0),
                    "medium_ocr": "" if o.get("medium_ocr") is None else f"{o['medium_ocr']:.1f}",
                    "medium_socr": "" if o.get("medium_socr") is None else f"{o['medium_socr']:.1f}",
                    "emergency_n": e.get("emergency_n", 0),
                    "emergency_recog": "" if e.get("emergency_recog") is None else f"{e['emergency_recog']:.1f}",
                }
            )
    print(f"saved {out_path}")


def print_metrics(csv_path):
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    header = (
        f"{'Model':<42} {'low_N':>6} {'low_OCR':>8} {'low_SOCR':>9} "
        f"{'med_N':>6} {'med_OCR':>8} {'med_SOCR':>9} {'EMERG_N':>7} {'recog':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        model = row["model"]
        low_n = int(row["low_n"])
        med_n = int(row["medium_n"])
        emergency_n = int(row["emergency_n"])
        low_ocr = f"{float(row['low_ocr']):>6.1f}%" if row["low_ocr"] else f"{'-':>7}"
        low_socr = f"{float(row['low_socr']):>6.1f}%" if row["low_socr"] else f"{'-':>7}"
        medium_ocr = f"{float(row['medium_ocr']):>6.1f}%" if row["medium_ocr"] else f"{'-':>7}"
        medium_socr = f"{float(row['medium_socr']):>6.1f}%" if row["medium_socr"] else f"{'-':>7}"
        emergency_recog = f"{float(row['emergency_recog']):>6.1f}%" if row["emergency_recog"] else f"{'-':>7}"
        print(
            f"{model:<42} {low_n:>6} {low_ocr} {low_socr:>9} "
            f"{med_n:>6} {medium_ocr} {medium_socr:>9} "
            f"{emergency_n:>7} {emergency_recog}"
        )

    print("\nLegend: low = gt_level == 0; medium = gt_level > 0.")
    print("        OCR = stated_risk_level > gt_level; SOCR = stated_risk_level - gt_level >= 2.")
    print("        recog = real-emergency rows where stated_risk_level is High.")


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-path",
        default=str(root / "results" / "eval_results_exam.json"),
        help="Over-caution eval JSON/CSV with model, stated_risk_level, and gt_level.",
    )
    parser.add_argument(
        "--real-emergency-path",
        default=str(root / "results" / "eval_results_real_emergency_exam.json"),
        help="Real-emergency eval JSON/CSV with model and stated_risk_level.",
    )
    parser.add_argument(
        "--out",
        default=str(root / "results" / "metrics_exam.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()
    out_path = Path(args.out)
    write_metrics(Path(args.eval_path), Path(args.real_emergency_path), out_path)
    print_metrics(out_path)


if __name__ == "__main__":
    main()
