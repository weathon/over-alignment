"""Reasoning-effort sweep for the over-caution split.

Same call shape as bench_dataset.py, but restricted to three models and
sweeping reasoning.effort across {minimal, low, medium, high}. Each
(model, effort) pair is stored under a local id of the form
`<base_id>:reasoning=<effort>` so they don't collide on resume.

Output: results/results_reasoning_sweep.json (separate from the main run).
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
SRC = ROOT / "data" / "dataset.txt"
OUT = ROOT / "results" / "results_reasoning_sweep.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

dotenv.load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

BASE_MODELS = [
    "openai/gpt-5.5",
    "google/gemini-3-flash-preview",
    "anthropic/claude-sonnet-4.6",
]

EFFORT_LEVELS = ["minimal", "low", "medium", "high"]

REASONING_PREFIX = ":reasoning="

models = [f"{m}{REASONING_PREFIX}{e}" for m in BASE_MODELS for e in EFFORT_LEVELS]


def resolve_model(model_id: str) -> tuple[str, str]:
    """Return (openrouter_route, reasoning_effort). The trailing
    `:reasoning=<level>` is a local marker — strip it before calling and
    pass the level through as reasoning.effort."""
    idx = model_id.rfind(REASONING_PREFIX)
    if idx == -1:
        return model_id, "none"
    return model_id[:idx], model_id[idx + len(REASONING_PREFIX):]


def load_prompts(path: Path) -> list[str]:
    with path.open("r") as f:
        raw = f.read().split("\n-\n")
    return [p.strip() for p in raw if p.strip()]


def _call_api(sample):
    route, effort = resolve_model(sample["model"])
    return client.chat.completions.create(
        model=route,
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": sample["prompt"]}],
            },
        ],
        extra_body={"reasoning": {"effort": effort}},
    )


def eval(sample, max_retries=5):
    retries = 0
    while True:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_api, sample)
            try:
                # medium/high efforts can take a while; give them headroom.
                _, effort = resolve_model(sample["model"])
                timeout = 240 if effort in {"medium", "high"} else 90
                completion = future.result(timeout=timeout)
                if completion.choices[0].message.content:
                    sample["response"] = completion.choices[0].message.content
                    return sample
            except FuturesTimeout:
                retries += 1
                if retries >= max_retries:
                    raise TimeoutError("Max retries exceeded")
                continue
            except Exception as e:
                print(e)
                continue


def main() -> None:
    prompts = load_prompts(SRC)
    print(f"loaded {len(prompts)} prompts from {SRC}")

    ds = [{"prompt": p, "model": m} for p in prompts for m in models]
    print(f"{len(ds)} (prompt, model) samples to evaluate "
          f"({len(BASE_MODELS)} models × {len(EFFORT_LEVELS)} effort levels)")

    def _prompt_str(p):
        return p[0]["text"] if isinstance(p, list) else p

    results: dict[int, dict] = {}
    if OUT.exists():
        try:
            with OUT.open("r") as f:
                prior = json.load(f)
            done = {(_prompt_str(s["prompt"]), s["model"]): s for s in prior}
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


def _save(results: dict[int, dict]) -> None:
    ordered = [results[i] for i in sorted(results.keys())]
    with OUT.open("w") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
