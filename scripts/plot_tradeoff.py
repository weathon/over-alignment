"""Scatter: over-cautious (low+medium) vs emergency recall, families connected by version order."""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/metrics.csv")
df = df[df["model"] != "TOTAL"].copy()
df["over_cautious"] = df["low_ocr"] + df["medium_ocr"]

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
    "Qwen": [("3.6-plus", "qwen/qwen3.6-plus")],
    "Grok": [("4.20", "x-ai/grok-4.20")],
}

colors = {"Claude": "#D97757", "Gemini": "#1A73E8", "Gemma": "#34A853",
          "GPT": "#10A37F", "Qwen": "#9333EA", "Grok": "#000000"}

fig, ax = plt.subplots(figsize=(11, 8))
for fam, models in families.items():
    xs, ys, labels = [], [], []
    for label, mid in models:
        row = df[df["model"] == mid]
        if row.empty:
            continue
        xs.append(row["emergency_recog"].values[0])
        ys.append(row["over_cautious"].values[0])
        labels.append(label)
    if not xs:
        continue
    ax.plot(xs, ys, "-o", color=colors[fam], label=fam,
            markersize=9, linewidth=2, alpha=0.85)
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), fontsize=7,
                    xytext=(5, 5), textcoords="offset points",
                    color=colors[fam])

ax.set_xlabel("Emergency Recall (%)")
ax.set_ylabel("Over-Cautious Rate: Low + Medium (%)")
ax.set_title("Over-Cautious vs Emergency Recall (lines = version progression)")
ax.grid(True, alpha=0.3)
ax.legend(loc="best", fontsize=10)
plt.tight_layout()
out = "results/tradeoff.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
