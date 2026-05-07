"""Plot over-cautious rate (low/medium) vs emergency recall, by family,
ordered by version number within each family."""
import pandas as pd
import matplotlib.pyplot as plt
import re

df = pd.read_csv("results/metrics.csv")
df = df[df["model"] != "TOTAL"].copy()

# Group by family. Manually curate ordering within each family by version.
# Each entry: (display_name, model_id_in_csv)
families = {
    "Claude": [
        ("3.5-haiku", "anthropic/claude-3.5-haiku"),
        ("3.7-sonnet", "anthropic/claude-3.7-sonnet"),
        ("sonnet-4", "anthropic/claude-sonnet-4"),
        ("sonnet-4.6", "anthropic/claude-sonnet-4.6"),
        ("sonnet-4.6:thinking", "anthropic/claude-sonnet-4.6:thinking"),
        ("opus-4.7", "anthropic/claude-opus-4.7"),
    ],
    "Gemini": [
        ("2.0-flash", "google/gemini-2.0-flash-001"),
        ("2.5-flash", "google/gemini-2.5-flash"),
        ("3-flash-preview", "google/gemini-3-flash-preview"),
        ("3-flash:thinking", "google/gemini-3-flash-preview:thinking"),
    ],
    "Gemma": [
        ("3-27b", "google/gemma-3-27b-it"),
        ("4-31b", "google/gemma-4-31b-it"),
    ],
    "GPT": [
        ("3.5-turbo", "openai/gpt-3.5-turbo"),
        ("4-turbo", "openai/gpt-4-turbo"),
        ("4o-2024-05", "openai/gpt-4o-2024-05-13"),
        ("4o-2024-11", "openai/gpt-4o-2024-11-20"),
        ("4.1", "openai/gpt-4.1"),
        ("5-chat", "openai/gpt-5-chat"),
        ("5.3-chat", "openai/gpt-5.3-chat"),
        ("5.5:thinking", "openai/gpt-5.5:thinking"),
    ],
    "Qwen": [
        ("3.6-plus", "qwen/qwen3.6-plus"),
    ],
    "Grok": [
        ("4.20", "x-ai/grok-4.20"),
    ],
}

colors = {
    "Claude": "#D97757",
    "Gemini": "#1A73E8",
    "Gemma": "#34A853",
    "GPT": "#10A37F",
    "Qwen": "#9333EA",
    "Grok": "#000000",
}

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
df["over_cautious"] = df["low_ocr"] + df["medium_ocr"]
metrics = [("over_cautious", "Over-Cautious Rate: Low + Medium (%)"),
           ("emergency_recog", "Emergency Recall (%)")]

for ax, (col, title) in zip(axes, metrics):
    for fam, models in families.items():
        xs, ys, labels = [], [], []
        for i, (label, mid) in enumerate(models):
            row = df[df["model"] == mid]
            if row.empty:
                continue
            xs.append(i)
            ys.append(row[col].values[0])
            labels.append(label)
        if not xs:
            continue
        ax.plot(xs, ys, "-o", color=colors[fam], label=fam,
                markersize=8, linewidth=2)
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(lbl, (x, y), fontsize=7,
                        xytext=(4, 4), textcoords="offset points",
                        color=colors[fam])
    ax.set_xlabel("Version order within family")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

plt.tight_layout()
out = "results/family_evolution.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
