"""Gradio UI for re-labelling anxiety index on data/anxiety_index_gt.csv.

Shows only:
- GP risk tier
- Question
- Answer

It saves the new human label together with the old label for later agreement
analysis, and reports aggregate correlation once enough rows are labelled.
"""

import csv
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "anxiety_index_gt.csv"
OUT = ROOT / "data" / "anxiety_index_relabel_gradio.csv"

REQUIRED = ["id", "datapoint_id", "gp_risk_tier", "question", "answer", "Annotator1_Response"]

df = pd.read_csv(SRC)
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    raise ValueError(f"{SRC}: missing columns {missing}")
df["old_anxiety_index"] = pd.to_numeric(df["Annotator1_Response"])
bad = df[~df["old_anxiety_index"].between(0, 10)]
if len(bad):
    raise ValueError(f"{SRC}: old anxiety index outside 0-10 for ids {bad['id'].tolist()}")

if OUT.exists():
    labels = pd.read_csv(OUT)
    missing_out = [c for c in ["datapoint_id", "new_anxiety_index"] if c not in labels.columns]
    if missing_out:
        raise ValueError(f"{OUT}: missing columns {missing_out}")
else:
    labels = pd.DataFrame(
        columns=[
            "id",
            "datapoint_id",
            "gp_risk_tier",
            "question",
            "answer",
            "old_anxiety_index",
            "new_anxiety_index",
            "labelled_at",
        ]
    )


def current_label(datapoint_id):
    hit = labels[labels["datapoint_id"] == datapoint_id]
    if len(hit) > 1:
        raise ValueError(f"{OUT}: duplicate label for datapoint_id {datapoint_id}")
    if len(hit) == 0:
        return 5
    return int(hit["new_anxiety_index"].iloc[0])


def page(i):
    i = int(i)
    row = df.iloc[i]
    return (
        i,
        f"{i + 1} / {len(df)}",
        row["gp_risk_tier"],
        row["question"],
        row["answer"],
        current_label(row["datapoint_id"]),
        stats(),
    )


def stats():
    if len(labels) == 0:
        return "labelled: 0 / 50\ncorrelation: not enough labels"
    merged = df[["datapoint_id", "old_anxiety_index"]].merge(
        labels[["datapoint_id", "new_anxiety_index"]],
        on="datapoint_id",
        how="inner",
    )
    if len(merged) < 2:
        return f"labelled: {len(merged)} / {len(df)}\ncorrelation: not enough labels"
    if merged["old_anxiety_index"].nunique() < 2 or merged["new_anxiety_index"].nunique() < 2:
        return f"labelled: {len(merged)} / {len(df)}\ncorrelation: needs label variance"
    pearson = merged["old_anxiety_index"].corr(merged["new_anxiety_index"], method="pearson")
    spearman = merged["old_anxiety_index"].corr(merged["new_anxiety_index"], method="spearman")
    return f"labelled: {len(merged)} / {len(df)}\npearson: {pearson:.3f}\nspearman: {spearman:.3f}"


def save_label(i, value):
    global labels
    i = int(i)
    value = int(value)
    if value < 0 or value > 10:
        raise ValueError(f"new anxiety index outside 0-10: {value}")
    row = df.iloc[i]
    labels = labels[labels["datapoint_id"] != row["datapoint_id"]].copy()
    labels = pd.concat(
        [
            labels,
            pd.DataFrame(
                [
                    {
                        "id": row["id"],
                        "datapoint_id": row["datapoint_id"],
                        "gp_risk_tier": row["gp_risk_tier"],
                        "question": row["question"],
                        "answer": row["answer"],
                        "old_anxiety_index": int(row["old_anxiety_index"]),
                        "new_anxiety_index": value,
                        "labelled_at": datetime.now().isoformat(timespec="seconds"),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    labels.sort_values("id").to_csv(OUT, index=False, quoting=csv.QUOTE_ALL)
    return stats()


def save_next(i, value):
    save_label(i, value)
    return page(min(int(i) + 1, len(df) - 1))


def previous(i):
    return page(max(int(i) - 1, 0))


def next_row(i):
    return page(min(int(i) + 1, len(df) - 1))


with gr.Blocks(title="Anxiety Index Relabel") as demo:
    idx = gr.State(0)
    gr.Markdown("# Anxiety Index Relabel")
    progress = gr.Textbox(label="Progress", interactive=False)
    gp = gr.Textbox(label="GP risk tier", interactive=False)
    q = gr.Textbox(label="Question", lines=8, interactive=False)
    a = gr.Textbox(label="Answer", lines=16, interactive=False)
    label = gr.Slider(0, 10, value=5, step=1, label="New anxiety index")
    status = gr.Textbox(label="Aggregate agreement", lines=4, interactive=False)
    with gr.Row():
        prev_btn = gr.Button("Previous")
        save_btn = gr.Button("Save")
        save_next_btn = gr.Button("Save + Next", variant="primary")
        next_btn = gr.Button("Next")

    demo.load(page, inputs=[idx], outputs=[idx, progress, gp, q, a, label, status])
    prev_btn.click(previous, inputs=[idx], outputs=[idx, progress, gp, q, a, label, status])
    next_btn.click(next_row, inputs=[idx], outputs=[idx, progress, gp, q, a, label, status])
    save_btn.click(save_label, inputs=[idx, label], outputs=[status])
    save_next_btn.click(save_next, inputs=[idx, label], outputs=[idx, progress, gp, q, a, label, status])


if __name__ == "__main__":
    demo.launch()
