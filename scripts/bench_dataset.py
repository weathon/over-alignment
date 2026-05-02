"""Generate model responses for the over-caution benchmark.

Reads prompts from data/dataset.txt (one per "\\n-\\n"-separated block), fans
out across the model list (no-reasoning baselines plus :thinking variants for
reasoning-capable models), and writes a flat list of {prompt, model, response}
dicts to data/results.json. Resumes from any prior run by skipping
(prompt, model) pairs already present in the output file.

Extracted from notebooks/bench_dataset.ipynb so it can be rerun headless and
checkpointed by the harness without spinning up a Jupyter kernel.
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
OUT = ROOT / "data" / "results.json"

dotenv.load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

"""Model-id convention:
- A bare id (e.g. "anthropic/claude-sonnet-4.6") is the no-reasoning baseline;
  we pass reasoning.effort=none on the API call.
- A `:thinking` suffix is OUR local marker, NOT an OpenRouter route. We strip it
  before calling and pass reasoning.effort=high. Results are stored under the
  suffixed name so reasoning vs. no-reasoning runs don't collide on resume.
"""

models = [
    "google/gemini-2.0-flash-001",
    "google/gemini-2.5-flash",
    "google/gemini-3-flash-preview",
    "google/gemini-3-flash-preview:thinking",
    "openai/gpt-5.3-chat",
    "openai/gpt-5-chat",
    "openai/gpt-5.5:thinking",
    "openai/gpt-4.1",
    "openai/gpt-4o-2024-11-20",
    "openai/gpt-4o-2024-05-13",
    "openai/gpt-4-turbo",
    "openai/gpt-3.5-turbo",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-sonnet-4.6:thinking",
    "x-ai/grok-4.20",
    "anthropic/claude-3.5-haiku",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-sonnet-4",
    "qwen/qwen3.6-plus",
]

THINKING_SUFFIX = ":thinking"


def resolve_model(model_id: str) -> tuple[str, str]:
    if model_id.endswith(THINKING_SUFFIX):
        return model_id[: -len(THINKING_SUFFIX)], "high"
    return model_id, "none"


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
                # Reasoning runs (effort=high) need more headroom than the 60s baseline.
                timeout = 240 if sample["model"].endswith(THINKING_SUFFIX) else 60
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
    print(f"{len(ds)} (prompt, model) samples to evaluate")

    # Some legacy rows store the prompt as the OpenAI list-of-content shape
    # ([{"type":"text","text":"..."}]) instead of a bare string — normalise
    # before using it as a dict key, otherwise the resume dict-comp blows up
    # with TypeError: unhashable type: 'list'.
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
