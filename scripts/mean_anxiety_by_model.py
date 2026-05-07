"""Print per-model calibrated anxiety metrics from results/eval_results.json.

The raw DeepSeek anxiety index is calibrated with a linear regression fit on
the 42-row non-High GP-risk calibration set:

    human_anxiety_index = slope * predicted_anxiety_index + intercept
"""

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CALIBRATION = ROOT / "results" / "anxiety_index_gt_deepseek_v4_pro_non_high.csv"
EVAL_RESULTS = ROOT / "results" / "eval_results.json"


with CALIBRATION.open(newline="") as f:
    cal_rows = list(csv.DictReader(f))
if len(cal_rows) != 42:
    raise ValueError(f"{CALIBRATION}: expected 42 calibration rows, got {len(cal_rows)}")

xs = [float(r["predicted_anxiety_index"]) for r in cal_rows]
ys = [float(r["human_anxiety_index"]) for r in cal_rows]
x_bar = sum(xs) / len(xs)
y_bar = sum(ys) / len(ys)
den = sum((x - x_bar) ** 2 for x in xs)
if den == 0:
    raise ValueError(f"{CALIBRATION}: predicted anxiety index has zero variance")
slope = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / den
intercept = y_bar - slope * x_bar

with EVAL_RESULTS.open() as f:
    data = json.load(f)
if isinstance(data, dict):
    rows = list(data.values())
elif isinstance(data, list):
    rows = data
else:
    raise ValueError(f"{EVAL_RESULTS}: expected JSON dict or list")

calibrated_by_split_model = {
    "low risk (gt_level == 0)": defaultdict(list),
    "medium risk (gt_level > 0)": defaultdict(list),
}
for row in rows:
    model = row.get("model")
    if not model:
        raise ValueError(f"{EVAL_RESULTS}: row missing model")
    gt = row.get("gt_level")
    if gt is None or gt == "":
        raise ValueError(f"{EVAL_RESULTS}: row missing gt_level for model {model!r}")
    raw = row.get("anxiety_index")
    if raw is None or raw == "":
        raise ValueError(f"{EVAL_RESULTS}: row missing anxiety_index for model {model!r}")
    gt = int(gt)
    raw = float(raw)
    calibrated = slope * raw + intercept
    calibrated = min(10, max(0, calibrated))
    if gt == 0:
        split = "low risk (gt_level == 0)"
    elif gt > 0:
        split = "medium risk (gt_level > 0)"
    else:
        raise ValueError(f"{EVAL_RESULTS}: invalid gt_level {gt} for model {model!r}")
    calibrated_by_split_model[split][model].append(calibrated)

low = "low risk (gt_level == 0)"
medium = "medium risk (gt_level > 0)"
models = sorted(
    set(calibrated_by_split_model[low]) | set(calibrated_by_split_model[medium]),
    key=lambda m: sum(calibrated_by_split_model[low].get(m, []) + calibrated_by_split_model[medium].get(m, []))
    / len(calibrated_by_split_model[low].get(m, []) + calibrated_by_split_model[medium].get(m, [])),
    reverse=True,
)

print(f"% calibration: human = {slope:.4f} * raw + {intercept:.4f}")
print("\\begin{tabular}{lrrrr}")
print("\\toprule")
print("Model & Anxiety Index (No Risk) & Anxiety Index (Medium Risk) & High Anxiety Portion (No Risk) & High Anxiety Portion (Medium Risk) \\\\")
print("\\midrule")
for model in models:
    low_vals = calibrated_by_split_model[low].get(model, [])
    medium_vals = calibrated_by_split_model[medium].get(model, [])
    if not low_vals:
        raise ValueError(f"{EVAL_RESULTS}: no low-risk rows for model {model!r}")
    if not medium_vals:
        raise ValueError(f"{EVAL_RESULTS}: no medium-risk rows for model {model!r}")
    low_mean = sum(low_vals) / len(low_vals)
    medium_mean = sum(medium_vals) / len(medium_vals)
    low_high = sum(1 for x in low_vals if x > 5) / len(low_vals) * 100
    medium_high = sum(1 for x in medium_vals if x > 5) / len(medium_vals) * 100
    model_tex = model.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
    print(f"{model_tex} & {low_mean:.3f} & {medium_mean:.3f} & {low_high:.1f}\\% & {medium_high:.1f}\\% \\\\")
print("\\bottomrule")
print("\\end{tabular}")
