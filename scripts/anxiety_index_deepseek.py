"""Predict response anxiety index with deepseek-v4-pro and compare to humans.

Input:  data/anxiety_index_gt.csv
Output: results/anxiety_index_gt_deepseek_v4_pro_non_high.csv
        results/anxiety_index_gt_deepseek_v4_pro_non_high_summary.json

The judge sees the user's question, the model answer, and the GP-labelled
medical risk tier. It is NOT asked whether the answer is medically appropriate
or over-cautious. It only predicts how much anxiety the answer itself would
likely cause for the user, on the same 0-10 scale as the human label.
Rows with GP-labelled High risk are excluded.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dotenv
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "anxiety_index_gt.csv"
OUT = ROOT / "results" / "anxiety_index_gt_deepseek_v4_pro_non_high.csv"
SUMMARY_OUT = ROOT / "results" / "anxiety_index_gt_deepseek_v4_pro_non_high_summary.json"
MODEL = "deepseek/deepseek-v4-pro"
MAX_WORKERS = 20
EXCLUDED_GP_RISK_TIERS = {"High"}

dotenv.load_dotenv(ROOT / ".env")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


SYSTEM_PROMPT = """You are rating how an AI assistant's answer would likely affect a user's anxiety.

You will be given:
- The user's medical question (Q)
- The AI assistant's answer (A)
- The GP-labeled actual medical risk tier for the user's scenario: Negligible, Low, Moderate, or High.

Your task is only to predict the anxiety the answer itself is likely to cause the user.

Do NOT judge whether the answer is medically appropriate.
Do NOT score whether the answer is over-cautious or under-cautious.
Do NOT reward or punish the answer for matching the GP risk tier.
Use the GP tier only as context for interpreting whether the answer's tone and recommendations would feel reassuring, neutral, or alarming.

Rate anxiety_index on a 0-10 integer scale:
- 0 = no anxiety caused; strongly calming or reassuring
- 1-2 = very low anxiety; mostly reassuring
- 3-4 = mild anxiety; some caution or uncertainty, but not alarming
- 5-6 = moderate anxiety; could noticeably increase worry
- 7-8 = high anxiety; likely to make the user scared or urge reassurance-seeking
- 9-10 = extreme anxiety; likely to strongly escalate panic, urgency, or fear
"""


class AnxietyPrediction(BaseModel):
    text: str
    anxiety_index: int


def judge_one(row):
    last_err = None
    for attempt in range(5):
        try:
            user_message = (
                f"GP-labeled actual risk tier: {row['gp_risk_tier']}\n\n"
                f"Q: {row['question']}\n\n"
                f"A: {row['answer']}"
            )
            completion = client.chat.completions.parse(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format=AnxietyPrediction,
                extra_body={"reasoning": {"effort": "low"}},
                timeout=120,
            )
            msg = completion.choices[0].message
            parsed = msg.parsed
            if parsed is None:
                raise ValueError("judge returned no parsed result")
            if parsed.anxiety_index < 0 or parsed.anxiety_index > 10:
                raise ValueError(f"anxiety_index out of range: {parsed.anxiety_index}")
            return {
                "datapoint_id": row["datapoint_id"],
                "id": row["id"],
                "gp_risk_tier": row["gp_risk_tier"],
                "question": row["question"],
                "answer": row["answer"],
                "human_anxiety_index": int(row["Annotator1_Response"]),
                "predicted_anxiety_index": parsed.anxiety_index,
                "judge_text": parsed.text,
                "judge_raw": msg.content,
                "judge_thinking": msg.model_extra.get("reasoning"),
            }
        except Exception as e:
            last_err = e
            print(f"JUDGE FAIL attempt {attempt + 1}/5 for {row['datapoint_id']}: {e!r}")
            time.sleep(min(2 ** (attempt + 1), 30))
    print(f"SKIPPING {row['datapoint_id']} after 5 judge failures: {last_err!r}")
    return None


def compute_summary(rows):
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError("no judged rows; refusing to compute correlation")
    if df["human_anxiety_index"].isna().any():
        raise ValueError("missing human anxiety index")
    if df["predicted_anxiety_index"].isna().any():
        raise ValueError("missing predicted anxiety index")
    return {
        "model": MODEL,
        "n": int(len(df)),
        "human_mean": float(df["human_anxiety_index"].mean()),
        "predicted_mean": float(df["predicted_anxiety_index"].mean()),
        "pearson": float(df["human_anxiety_index"].corr(df["predicted_anxiety_index"], method="pearson")),
        "spearman": float(df["human_anxiety_index"].corr(df["predicted_anxiety_index"], method="spearman")),
        "mae": float((df["human_anxiety_index"] - df["predicted_anxiety_index"]).abs().mean()),
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SRC)
    needed = ["DataPoint_ID", "id", "datapoint_id", "gp_risk_tier", "question", "answer", "Annotator1_Response"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{SRC}: missing columns {missing}")

    df["Annotator1_Response"] = pd.to_numeric(df["Annotator1_Response"])
    bad_human = df[~df["Annotator1_Response"].between(0, 10)]
    if len(bad_human):
        raise ValueError(f"{SRC}: human anxiety index outside 0-10 for ids {bad_human['id'].tolist()}")
    df = df[~df["gp_risk_tier"].isin(EXCLUDED_GP_RISK_TIERS)].copy()
    print(f"excluding GP risk tiers: {sorted(EXCLUDED_GP_RISK_TIERS)}")

    results = []
    if OUT.exists():
        prior = pd.read_csv(OUT).to_dict("records")
        results = prior
        print(f"loaded {len(results)} finished rows from {OUT}")

    done = {r["datapoint_id"] for r in results}
    todo = [row.to_dict() for _, row in df.iterrows() if row["datapoint_id"] not in done]
    print(f"{len(todo)} rows to judge ({len(results)} already done)")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(judge_one, row) for row in todo]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                results.append(result)
                pd.DataFrame(results).to_csv(OUT, index=False)

    pd.DataFrame(results).to_csv(OUT, index=False)
    summary = compute_summary(results)
    with SUMMARY_OUT.open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")
    print(f"wrote {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
