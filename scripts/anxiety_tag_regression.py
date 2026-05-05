import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "anxiety_index_gt_tag_proxy.csv"
OUT_CSV = ROOT / "results" / "anxiety_index_gt_tag_regression.csv"
OUT_PNG = ROOT / "results" / "anxiety_index_gt_tag_regression_coefficients.png"

TAG_NAMES = {
    1: "Catastrophize",
    2: "Acknowledge Low Risk",
    3: "Provide Anxiety Help",
    4: "Medicalize Anxiety",
    5: "Over-test",
    6: "Avoidance Advice",
    7: "Needless Urgency",
    8: "Validate Irrational Fear",
}


df = pd.read_csv(INPUT)
if len(df) == 0:
    raise RuntimeError(f"{INPUT} has no rows")

rows = []
for _, row in df.iterrows():
    tags = ast.literal_eval(row["tags"])
    out = {"human_anxiety_index": float(row["human_anxiety_index"])}
    for tag_id in TAG_NAMES:
        out[f"tag_{tag_id}"] = int(tag_id in tags)
    rows.append(out)

reg = pd.DataFrame(rows)
y = reg["human_anxiety_index"].to_numpy(dtype=float)
X = reg[[f"tag_{tag_id}" for tag_id in TAG_NAMES]].to_numpy(dtype=float)
X = pd.DataFrame(X, columns=[f"tag_{tag_id}" for tag_id in TAG_NAMES])
X.insert(0, "intercept", 1.0)

beta = pd.Series(
    data=np.linalg.lstsq(X.to_numpy(), y, rcond=None)[0],
    index=X.columns,
)
pred = X.to_numpy() @ beta.to_numpy()
resid = y - pred

ss_res = float((resid**2).sum())
ss_tot = float(((y - y.mean()) ** 2).sum())
r2 = 1.0 - ss_res / ss_tot

summary = []
summary.append(
    {
        "feature": "intercept",
        "tag_id": "",
        "tag_name": "Intercept",
        "coefficient": beta["intercept"],
        "prevalence": 1.0,
        "r2": r2,
    }
)
for tag_id, tag_name in TAG_NAMES.items():
    col = f"tag_{tag_id}"
    summary.append(
        {
            "feature": col,
            "tag_id": tag_id,
            "tag_name": tag_name,
            "coefficient": beta[col],
            "prevalence": reg[col].mean(),
            "r2": r2,
        }
    )

summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUT_CSV, index=False)

plot_df = summary_df[summary_df["feature"] != "intercept"].sort_values("coefficient")
colors = ["#b84a5b" if val > 0 else "#2f7d6d" for val in plot_df["coefficient"]]
plt.figure(figsize=(9, 5))
plt.barh(plot_df["tag_name"], plot_df["coefficient"], color=colors)
plt.axvline(0, color="#333333", linewidth=1)
plt.xlabel("Linear regression coefficient on human anxiety index")
plt.title(f"Tag Binary Regression Coefficients (n={len(reg)}, R2={r2:.3f})")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)

print(f"n={len(reg)}")
print(f"r2={r2:.4f}")
print(summary_df[["tag_name", "coefficient", "prevalence"]].to_string(index=False))
print(f"wrote {OUT_CSV}")
print(f"wrote {OUT_PNG}")
