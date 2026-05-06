"""Rewrite over-caution prompts into medical exam-style questions.

Reads data/dataset.txt and writes data/dataset_exam.json as a list of
{"original": ..., "prompt": ...}. The rewritten prompt is what exam-mode
bench_dataset.py sends to models; original is kept for gt matching in eval_exam.py.
"""

import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "dataset.txt"
OUT = ROOT / "data" / "dataset_exam.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

dotenv.load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


class RewriteResult(BaseModel):
    prompt: str


SYSTEM_PROMPT = """You rewrite patient-style health questions into serious medical exam questions.

Keep every medically relevant fact, exposure, symptom, timing detail, and prior workup detail from the original. Do not add new facts. Do not answer the question. Do not state the risk tier.

Make the result sound like a formal exam vignette or clinical risk-assessment item, not a casual user query. It should ask the examinee to assess the medical risk and appropriate care urgency.

Return only the rewritten question in the structured output field."""


def load_prompts(path: Path) -> list[str]:
    with path.open("r") as f:
        raw = f.read().split("\n-\n")
    return [p.strip() for p in raw if p.strip()]


def rewrite_prompt(prompt: str, max_retries=5) -> str:
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.parse(
                model="deepseek/deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format=RewriteResult,
                extra_body={"reasoning": {"effort": "low"}},
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise ValueError("rewrite returned no parsed result")
            rewritten = parsed.prompt.strip()
            if not rewritten:
                raise ValueError("rewrite returned empty prompt")
            return rewritten
        except Exception as e:
            print(f"rewrite failed attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(min(2 ** (attempt + 1), 30))


def main() -> None:
    prompts = load_prompts(SRC)
    print(f"loaded {len(prompts)} prompts from {SRC}")

    rows_by_original = {}
    if OUT.exists():
        with OUT.open("r") as f:
            prior = json.load(f)
        rows_by_original = {
            row["original"]: row
            for row in prior
            if row.get("original") and row.get("prompt")
        }
        print(f"loaded {len(rows_by_original)} rewritten prompts from {OUT}")

    todo = [p for p in prompts if p not in rows_by_original]
    executor = ThreadPoolExecutor(max_workers=30)
    futures = {executor.submit(rewrite_prompt, p): p for p in todo}
    pending = set(futures)
    with tqdm(total=len(futures)) as pbar:
        while pending:
            done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in done:
                original = futures[future]
                rows_by_original[original] = {
                    "original": original,
                    "prompt": future.result(),
                }
                save(prompts, rows_by_original)
                pbar.update(1)
    executor.shutdown()

    missing = [p for p in prompts if p not in rows_by_original]
    if missing:
        raise ValueError(f"{len(missing)} prompts were not rewritten")

    save(prompts, rows_by_original)
    print(f"wrote {len(prompts)} rewritten prompts to {OUT}")


def save(prompts: list[str], rows_by_original: dict[str, dict]) -> None:
    rows = [rows_by_original[p] for p in prompts if p in rows_by_original]
    with OUT.open("w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
