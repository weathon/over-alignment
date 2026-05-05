"""Correlate eval.py tag proxy with human anxiety-index labels.

Reads data/anxiety_index_gt.csv, runs only eval.py's tags judge on each
human-labelled (question, answer) row, and computes:

    tag_anxiety_proxy = negative_tag_count - positive_tag_count

where positive tags are Acknowledge Low Risk and Provide Anxiety Help, and
negative tags are the remaining anxiety-escalating tags.
"""

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dotenv
import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from eval import tags_system_prompt, _strip_fences


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "anxiety_index_gt.csv"
OUT = ROOT / "results" / "anxiety_index_gt_tag_proxy.csv"
PLOT = ROOT / "results" / "anxiety_index_gt_tag_proxy_scatter.png"

POSITIVE_TAGS = {2, 3}
NEGATIVE_TAGS = {1, 4, 5, 6, 7, 8}


dotenv.load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def rankdata(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i + 1
        while j < len(vals) and vals[order[j]] == vals[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def judge_tags(row):
    user_msg = (
        f"GP-labeled risk tier: {row['gp_risk_tier']}\n\n"
        f"Q: {row['question']}\n\n"
        f"A: {row['answer']}"
    )
    last_err = None
    for attempt in range(5):
        try:
            completion = client.chat.completions.create(
                model="deepseek/deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": tags_system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                extra_body={"reasoning": {"effort": "low"}},
                timeout=180,
            )
            raw = completion.choices[0].message.content
            data = json.loads(_strip_fences(raw))
            tags = [int(t) for t in data["tags"]]
            neg = sum(1 for t in tags if t in NEGATIVE_TAGS)
            pos = sum(1 for t in tags if t in POSITIVE_TAGS)
            out = dict(row)
            out["judge_tags_raw"] = raw
            out["tags"] = tags
            out["negative_tag_count"] = neg
            out["positive_tag_count"] = pos
            out["tag_anxiety_proxy"] = neg - pos
            return out
        except Exception as e:
            last_err = e
            if attempt == 4:
                raise RuntimeError(f"{row['DataPoint_ID']} tag judge failed: {last_err!r}")
            time.sleep(min(2 ** attempt, 10))


def save_rows(rows):
    OUT.parent.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("DataPoint_ID")
    df.to_csv(OUT, index=False)


def main():
    df = pd.read_csv(SRC)
    df["human_anxiety_index"] = df["Annotator1_Response"].astype(int)

    done = {}
    if OUT.exists():
        old = pd.read_csv(OUT)
        for row in old.to_dict("records"):
            done[row["DataPoint_ID"]] = row

    rows = []
    for row in df.to_dict("records"):
        rows.append(done.get(row["DataPoint_ID"], row))

    todo = [row for row in rows if "tag_anxiety_proxy" not in row or pd.isna(row["tag_anxiety_proxy"])]
    print(f"{len(todo)} rows to tag ({len(rows) - len(todo)} already done)")

    if todo:
        by_id = {row["DataPoint_ID"]: row for row in rows}
        with ThreadPoolExecutor(max_workers=20) as ex:
            futures = [ex.submit(judge_tags, row) for row in todo]
            for future in tqdm(as_completed(futures), total=len(futures)):
                judged = future.result()
                by_id[judged["DataPoint_ID"]] = judged
                save_rows(list(by_id.values()))
        rows = list(by_id.values())

    out = pd.DataFrame(rows)
    out["human_anxiety_index"] = out["Annotator1_Response"].astype(int)
    out["tag_anxiety_proxy"] = out["tag_anxiety_proxy"].astype(float)
    r = pearson(out["tag_anxiety_proxy"].tolist(), out["human_anxiety_index"].tolist())
    rho = pearson(rankdata(out["tag_anxiety_proxy"].tolist()), rankdata(out["human_anxiety_index"].tolist()))

    print(f"n={len(out)}")
    print(f"pearson_r={r:.4f}")
    print(f"spearman_r={rho:.4f}")

    ax = out.plot.scatter(x="tag_anxiety_proxy", y="human_anxiety_index", alpha=0.75)
    ax.set_title(f"Tag proxy vs human anxiety index\nPearson r={r:.3f}, Spearman r={rho:.3f}, n={len(out)}")
    ax.set_xlabel("negative tag count - positive tag count")
    ax.set_ylabel("human anxiety index")
    ax.grid(True, alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(PLOT, dpi=200)
    print(f"wrote {OUT}")
    print(f"wrote {PLOT}")


if __name__ == "__main__":
    main()
