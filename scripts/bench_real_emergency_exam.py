"""Run the exam-style A/B/C/D benchmark on the real-emergency split.

Reads rewritten exam prompts from data/dataset_real_emergency_exam.json
(produced by scripts/rewrite_real_emergency_to_exam.py) and asks each model
to return one of A/B/C/D (A=Negligible, B=Low, C=Moderate, D=High) using
OpenRouter structured outputs.

Output: data/results_real_emergency_exam.json — flat list of
{prompt, original, model, response, letter}. Resumable on (prompt, model).
"""

import json
import os
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeout,
    wait,
)
from pathlib import Path

import dotenv
from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "dataset_real_emergency_exam.json"
OUT = ROOT / "data" / "results_real_emergency_exam.json"

dotenv.load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Same subset as bench_dataset_exam.py — keep them in sync.
models = [
    "google/gemini-2.0-flash-001",
    "google/gemini-2.5-flash",
    "google/gemini-3-flash-preview:thinking",
    "openai/gpt-5.5:thinking",
    "qwen/qwen3.6-plus",
]

THINKING_SUFFIX = ":thinking"

SYSTEM_PROMPT = (
    "You are answering a medical-school multiple-choice exam question. "
    "Respond with JSON {\"reasoning\": \"<short rationale>\", \"answer\": \"<letter>\"} "
    "where <reasoning> is a brief natural-language explanation (1-3 sentences) "
    "of how you assess the scenario, and <letter> is exactly one of A, B, C, or D "
    "corresponding to the four risk tiers (A=Negligible, B=Low, C=Moderate, D=High). "
    "Always provide the reasoning field before the answer field."
)

ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "exam_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "reasoning": {"type": "string"},
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
            },
            "required": ["reasoning", "answer"],
        },
    },
}


def resolve_model(model_id: str) -> tuple[str, str]:
    if model_id.endswith(THINKING_SUFFIX):
        return model_id[: -len(THINKING_SUFFIX)], "high"
    return model_id, "none"


def _call_api(sample):
    route, effort = resolve_model(sample["model"])
    return client.chat.completions.create(
        model=route,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["prompt"]},
        ],
        response_format=ANSWER_SCHEMA,
        extra_body={"reasoning": {"effort": effort}},
    )


def eval(sample, max_retries=5):
    retries = 0
    while True:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_api, sample)
            try:
                timeout = 240 if THINKING_SUFFIX in sample["model"] else 60
                completion = future.result(timeout=timeout)
                content = completion.choices[0].message.content
                if not content:
                    retries += 1
                    if retries >= max_retries:
                        return None
                    continue
                sample["response"] = content
                try:
                    sample["letter"] = json.loads(content).get("answer")
                except json.JSONDecodeError:
                    sample["letter"] = None
                return sample
            except FuturesTimeout:
                retries += 1
                if retries >= max_retries:
                    raise TimeoutError("Max retries exceeded")
                continue
            except Exception as e:
                print(e)
                retries += 1
                if retries >= max_retries:
                    return None
                continue


def _save(results: dict[int, dict]) -> None:
    ordered = [results[i] for i in sorted(results.keys())]
    with OUT.open("w") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)


def main() -> None:
    rows = json.loads(SRC.read_text())
    print(f"loaded {len(rows)} exam prompts from {SRC}")

    ds = [
        {"prompt": r["exam"], "original": r["original"], "model": m}
        for r in rows
        for m in models
    ]
    print(f"{len(ds)} (prompt, model) samples to evaluate")

    results: dict[int, dict] = {}
    if OUT.exists():
        try:
            with OUT.open("r") as f:
                prior = json.load(f)
            done = {(s["prompt"], s["model"]): s for s in prior}
            for idx, sample in enumerate(ds):
                match = done.get((sample["prompt"], sample["model"]))
                if match is not None:
                    results[idx] = match
            print(
                f"loaded {len(results)} finished samples from {OUT}; "
                f"{len(ds) - len(results)} remaining"
            )
        except (json.JSONDecodeError, ValueError) as e:
            print(f"could not load {OUT}: {e}")

    executor = ThreadPoolExecutor(max_workers=30)
    futures = {
        executor.submit(eval, ds[idx]): idx
        for idx in range(len(ds))
        if idx not in results
    }

    try:
        pending = set(futures)
        with tqdm(total=len(futures)) as pbar:
            while pending:
                done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in done:
                    idx = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        print(f"sample {idx} failed: {e}")
                        result = None
                    if result is not None:
                        results[idx] = result
                    pbar.update(1)
                    if pbar.n % 200 == 0:
                        _save(results)
    except KeyboardInterrupt:
        print("interrupted, cancelling...")
        for f in futures:
            f.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        _save(results)
        raise
    else:
        executor.shutdown()

    _save(results)
    print(f"wrote {len(results)} samples to {OUT}")


if __name__ == "__main__":
    main()
