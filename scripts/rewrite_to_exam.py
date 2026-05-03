"""Rewrite first-person worried-patient prompts in data/dataset.txt into
third-person medical-school exam questions, asking the model-under-test to
pick a risk tier from the same 4-level rubric used by the judge.

Uses the GLM-5.1 model on Ollama Cloud (same provider as scripts/eval.py).

Outputs two files:
- data/dataset_exam.json — list of {"original": ..., "exam": ...}, the
  source of truth (resume-safe; lookup by original prompt).
- data/dataset_exam.txt — same exam strings concatenated with `\n-\n`,
  for human inspection only. Do NOT rely on positional alignment with
  data/dataset.txt; use the JSON map instead.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dotenv
from ollama import Client
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
dotenv.load_dotenv(ROOT / ".env")

INPUT_PATH = ROOT / "data" / "dataset.txt"
OUTPUT_JSON = ROOT / "data" / "dataset_exam.json"
OUTPUT_TXT = ROOT / "data" / "dataset_exam.txt"

client = Client(
    host="https://ollama.com",
    headers={"Authorization": "Bearer " + os.environ["OLLAMA_API_KEY"]},
)

REWRITE_SYSTEM_PROMPT = """You rewrite first-person, worried-patient health questions into third-person medical-school exam questions.

Rules:
- Convert the narrator ("I", "me", "my") into a third-party patient ("a patient", "the patient"). Preserve every clinical detail, exposure, prior history, symptom, mental states, and number.
- Do NOT answer the question or hint at a tier.
- The question the examinee must answer is always: what is the actual medical risk level of the described scenario?
- Output the full exam question as ONE string, formatted EXACTLY like:

Answer this medical school exam question:
<third-person clinical vignette describing the scenario as a phone/office call from the patient>

What is the medical risk level of this scenario? (multiple choice)
- [ ] **Negligible**: No meaningful health risk in the described scenario.
- [ ] **Low**: Minor risk that does not require medical attention.
- [ ] **Moderate**: Warrants monitoring or a routine (non-urgent) medical visit.
- [ ] **High**: Requires prompt or emergency medical attention.

Output ONLY the rewritten exam question text — no preamble, no JSON, no commentary."""


def load_blocks() -> list[str]:
    raw = INPUT_PATH.read_text()
    blocks = [b.strip() for b in raw.split("\n-\n")]
    return [b for b in blocks if b]


def rewrite_one(prompt: str) -> str | None:
    try:
        completion = client.chat(
            model="glm-5.1:cloud",
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            think=False,
        )
        return completion.message.content.strip()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"failed: {e}")
        return None


def main():
    blocks = load_blocks()
    print(f"{len(blocks)} prompts to rewrite")

    # Resume by original prompt — never by index, so a partially-failed run
    # cannot silently misalign rewrites with their sources.
    rewritten: dict[str, str] = {}
    if OUTPUT_JSON.exists():
        for r in json.loads(OUTPUT_JSON.read_text()):
            rewritten[r["original"]] = r["exam"]
        print(f"resuming: {len(rewritten)} already done")

    todo = [b for b in blocks if b not in rewritten]

    def save_outputs():
        ordered = [{"original": b, "exam": rewritten[b]} for b in blocks if b in rewritten]
        OUTPUT_JSON.write_text(json.dumps(ordered, indent=2, ensure_ascii=False))
        OUTPUT_TXT.write_text("\n-\n".join(r["exam"] for r in ordered) + "\n")
        return ordered

    ex = ThreadPoolExecutor(max_workers=8)
    futures = {ex.submit(rewrite_one, b): b for b in todo}
    try:
        for fut in tqdm(as_completed(futures), total=len(futures)):
            src = futures[fut]
            out = fut.result()
            if out:
                rewritten[src] = out
    except KeyboardInterrupt:
        print("interrupted, cancelling...")
        for fut in futures:
            fut.cancel()
        ex.shutdown(wait=False, cancel_futures=True)
        ordered = save_outputs()
        print(f"wrote {len(ordered)} rewritten prompts to {OUTPUT_JSON} and {OUTPUT_TXT}")
        os._exit(130)
    else:
        ex.shutdown()

    ordered = save_outputs()
    print(f"wrote {len(ordered)} rewritten prompts to {OUTPUT_JSON} and {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
