"""Score anxiety-index GT rows with ModernBERT fear probability.

Reads data/anxiety_index_gt.csv, drops physician High-risk rows to match the
simple-judge comparison, and scores each AI answer with
cirimus/modernbert-base-emotions. The FEAR class probability is recorded both
as a raw probability and scaled to 0-10.
"""

import argparse
from pathlib import Path

import pandas as pd
import tqdm
from transformers import pipeline

ROOT = Path(__file__).resolve().parent.parent


def fear_score(predictions):
    matches = [p["score"] for p in predictions if p["label"].upper() == "FEAR"]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one FEAR score, got {predictions}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--input", default=str(ROOT / "data" / "anxiety_index_gt.csv"))
    parser.add_argument("--output", default=str(ROOT / "results" / "anxiety_index_gt_modernbert_fear.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    work = df[df["gp_risk_tier"] != "High"].copy()
    classifier = pipeline(
        "text-classification",
        model="cirimus/modernbert-base-emotions",
        top_k=None,
        device=args.device,
    )

    fear_probs = []
    answers = work["answer"].tolist()
    for i in tqdm.tqdm(range(0, len(answers), args.batch_size), desc="Scoring FEAR"):
        batch = answers[i : i + args.batch_size]
        batch_predictions = classifier(batch)
        fear_probs.extend(fear_score(pred) for pred in batch_predictions)

    work["gt_anxiety_index"] = work["Annotator1_Response"].astype(int)
    work["modernbert_fear_prob"] = fear_probs
    work["modernbert_fear_anxiety_index"] = work["modernbert_fear_prob"] * 10

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    work[
        [
            "id",
            "datapoint_id",
            "gp_risk_tier",
            "gt_anxiety_index",
            "modernbert_fear_prob",
            "modernbert_fear_anxiety_index",
            "question",
            "answer",
        ]
    ].to_csv(out, index=False)
    print(f"wrote {len(work)} rows to {out}")


if __name__ == "__main__":
    main()
