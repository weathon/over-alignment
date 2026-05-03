"""Gradio UI to collect human ground-truth anxiety_index labels and correlate
them against the over-caution judge's predicted anxiety_index.

Source data: results/eval_results.json (over-caution judge output).
Labels written to: results/anxiety_labels.json.

The judge's anxiety_index is loaded server-side but is NEVER sent to the
browser while labelling — it only appears in the correlation panel after the
user clicks "Show correlation". A fixed 50-sample subset is drawn with a
deterministic seed so reruns / resumes hit the same items.

Run:
    /home/wg25r/miniconda/envs/neg/bin/python scripts/anxiety_label_ui.py
"""

import json
import random
from pathlib import Path

import gradio as gr
import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "eval_results.json"
OUT = ROOT / "results" / "anxiety_labels.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 50
SEED = 17


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def load_samples() -> list[dict]:
    """Return exactly N_SAMPLES rows with parseable judge anxiety_index, drawn
    deterministically. Each row: {key, prompt, model, response, model_anxiety}."""
    with SRC.open("r") as f:
        raw = json.load(f)

    rows: list[dict] = []
    items = raw.values() if isinstance(raw, dict) else raw
    for v in items:
        if not v.get("response") or not v.get("judge"):
            continue
        try:
            jd = json.loads(_strip_fences(v["judge"]))
        except (json.JSONDecodeError, TypeError):
            continue
        anx = jd.get("anxiety_index")
        if anx is None:
            continue
        # Stable key so resume works across reruns.
        key = f"{v['model']}||{hash(v['prompt']) & 0xFFFFFFFF:08x}"
        rows.append({
            "key": key,
            "prompt": v["prompt"],
            "model": v["model"],
            "response": v["response"],
            "model_anxiety": int(anx),
        })

    rng = random.Random(SEED)
    rng.shuffle(rows)
    return rows[:N_SAMPLES]


def load_labels() -> dict[str, int]:
    if OUT.exists():
        try:
            with OUT.open("r") as f:
                return {k: int(v) for k, v in json.load(f).items()}
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def save_labels(labels: dict[str, int]) -> None:
    with OUT.open("w") as f:
        json.dump(labels, f, indent=2)


SAMPLES = load_samples()
print(f"loaded {len(SAMPLES)} samples for labelling")


def _next_unlabelled_idx(labels: dict[str, int], start: int = 0) -> int:
    """First index >= start whose key is not yet labelled, or N_SAMPLES if all done."""
    for i in range(start, len(SAMPLES)):
        if SAMPLES[i]["key"] not in labels:
            return i
    return len(SAMPLES)


def render(idx: int, labels: dict[str, int]):
    total = len(SAMPLES)
    done = len(labels)
    progress = f"**{done} / {total}** labelled"
    if idx >= total:
        return (
            progress,
            "(all samples labelled — scroll down for correlation)",
            "",
            "",
            5,
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
    s = SAMPLES[idx]
    header = f"### Sample {idx + 1} / {total} — model: `{s['model']}`"
    # Prefill slider with prior label if user is revisiting.
    prior = labels.get(s["key"], 5)
    return (
        progress,
        header,
        s["prompt"],
        s["response"],
        prior,
        gr.update(interactive=True),
        gr.update(interactive=idx > 0),
    )


def submit(idx: int, score: int, labels: dict[str, int]):
    if idx < len(SAMPLES):
        labels[SAMPLES[idx]["key"]] = int(score)
        save_labels(labels)
    next_idx = _next_unlabelled_idx(labels, idx + 1)
    # If everything before is also done, just step forward by one regardless.
    if next_idx == len(SAMPLES) and idx + 1 < len(SAMPLES):
        next_idx = idx + 1
    return (next_idx, labels, *render(next_idx, labels))


def go_back(idx: int, labels: dict[str, int]):
    new_idx = max(0, idx - 1)
    return (new_idx, labels, *render(new_idx, labels))


def jump_to_first_unlabelled(labels: dict[str, int]):
    idx = _next_unlabelled_idx(labels, 0)
    return (idx, labels, *render(idx, labels))


def compute_correlation(labels: dict[str, int]) -> str:
    paired = [(s["model_anxiety"], labels[s["key"]]) for s in SAMPLES if s["key"] in labels]
    n = len(paired)
    if n < 5:
        return f"Need at least 5 labels to compute correlation. Currently labelled: {n}."
    model_x = np.array([p[0] for p in paired])
    human_y = np.array([p[1] for p in paired])
    pr, pp = pearsonr(model_x, human_y)
    sr, sp = spearmanr(model_x, human_y)
    mae = float(np.mean(np.abs(model_x - human_y)))
    bias = float(np.mean(model_x - human_y))
    lines = [
        f"**n = {n}** paired samples",
        "",
        f"- Pearson r  = **{pr:+.3f}**  (p = {pp:.3g})",
        f"- Spearman ρ = **{sr:+.3f}**  (p = {sp:.3g})",
        f"- MAE        = {mae:.2f}",
        f"- Mean signed bias (model − human) = {bias:+.2f}  "
        f"({'model over-estimates' if bias > 0 else 'model under-estimates'} anxiety)",
        "",
        "**Per-sample (model, human):**",
        ", ".join(f"({m},{h})" for m, h in paired),
    ]
    return "\n".join(lines)


with gr.Blocks(title="Anxiety GT labelling") as demo:
    gr.Markdown(
        f"# Anxiety ground-truth labelling\n"
        f"Read the user's question and the model's response. Rate how likely the "
        f"**response** is to cause anxiety in the user, on a scale of **0 (not at all)** "
        f"to **10 (extremely)**. The judge's prediction is hidden until you open the "
        f"correlation panel at the bottom.\n\n"
        f"Source: `{SRC.relative_to(ROOT)}` — labels persisted to `{OUT.relative_to(ROOT)}`."
    )

    labels_state = gr.State(load_labels())
    initial_labels = labels_state.value
    initial_idx = _next_unlabelled_idx(initial_labels, 0)
    idx_state = gr.State(initial_idx)

    progress_md = gr.Markdown()
    header_md = gr.Markdown()
    prompt_box = gr.Textbox(label="User prompt (Q)", lines=4, interactive=False)
    response_box = gr.Textbox(label="Model response (A)", lines=18, interactive=False)
    score_slider = gr.Slider(
        minimum=0, maximum=10, step=1, value=5,
        label="Your anxiety_index (0 = not anxiety-inducing, 10 = extremely)",
    )
    with gr.Row():
        back_btn = gr.Button("← Back")
        submit_btn = gr.Button("Submit & next →", variant="primary")
        jump_btn = gr.Button("Jump to first un-labelled")

    with gr.Accordion("Show correlation (reveals model predictions)", open=False):
        corr_btn = gr.Button("Recompute correlation")
        corr_out = gr.Markdown()

    outputs = [
        idx_state, labels_state,
        progress_md, header_md, prompt_box, response_box, score_slider,
        submit_btn, back_btn,
    ]

    submit_btn.click(submit, [idx_state, score_slider, labels_state], outputs)
    back_btn.click(go_back, [idx_state, labels_state], outputs)
    jump_btn.click(jump_to_first_unlabelled, [labels_state], outputs)
    corr_btn.click(compute_correlation, [labels_state], [corr_out])

    # Initial render on load.
    demo.load(
        lambda labels: render(_next_unlabelled_idx(labels, 0), labels),
        inputs=[labels_state],
        outputs=[progress_md, header_md, prompt_box, response_box, score_slider, submit_btn, back_btn],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True, inbrowser=False)
