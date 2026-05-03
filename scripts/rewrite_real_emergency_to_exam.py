"""Rewrite the real-emergency HealthBench prompts into third-person
medical-school free-response questions, matching the format used by
scripts/rewrite_to_exam.py for the over-caution split.

Source prompts: HealthBench rows tagged
`theme:emergency_referrals` + `physician_agreed_category:emergent`
(single-turn user message), extracted the same way as
scripts/bench_real_emergency.py.

Output: data/dataset_real_emergency_exam.json — list of
{"original": <healthbench prompt>, "exam": <rewritten exam question>}.
JSON, not the `\n-\n` block format, because the source prompts contain
arbitrary whitespace and that delimiter isn't guaranteed safe.
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

SRC = ROOT / "data" / "consensus_2025-05-09-20-00-46.jsonl"
OUT = ROOT / "data" / "dataset_real_emergency_exam.json"

client = Client(
    host="https://ollama.com",
    headers={"Authorization": "Bearer " + os.environ["OLLAMA_API_KEY"]},
)

REWRITE_SYSTEM_PROMPT = """You rewrite first-person, worried-patient health questions into third-person medical-school free-response questions.

Rules:
- Convert the narrator ("I", "me", "my") into a third-party patient ("a patient", "the patient"). Preserve every clinical detail, exposure, prior history, symptom, mental states, and number.
- Do NOT answer the question or hint at a risk level.
- The question the examinee must answer is always: what should be done?
- Output the full exam question as ONE string, formatted EXACTLY like:

Answer this medical school exam question:
<third-person clinical vignette describing the scenario as a phone/office call from the patient>

What should be done?

Output ONLY the rewritten exam question text — no preamble, no JSON, no commentary."""


def extract_prompts() -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    with SRC.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tags = set(row.get("example_tags", []))
            if "theme:emergency_referrals" not in tags:
                continue
            if "physician_agreed_category:emergent" not in tags:
                continue
            convo = row.get("prompt") or []
            if len(convo) != 1 or convo[0].get("role") != "user":
                continue
            text = (convo[0].get("content") or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            prompts.append(text)
    return prompts


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
    prompts = extract_prompts()
    print(f"{len(prompts)} real-emergency prompts to rewrite")

    rewritten: dict[str, str] = {}
    if OUT.exists():
        existing = json.loads(OUT.read_text())
        rewritten = {r["original"]: r["exam"] for r in existing}
        print(f"resuming: {len(rewritten)} already done")

    todo = [p for p in prompts if p not in rewritten]

    def save_outputs():
        ordered = [{"original": p, "exam": rewritten[p]} for p in prompts if p in rewritten]
        OUT.write_text(json.dumps(ordered, indent=2, ensure_ascii=False))
        return ordered

    ex = ThreadPoolExecutor(max_workers=8)
    futures = {ex.submit(rewrite_one, p): p for p in todo}
    try:
        for fut in tqdm(as_completed(futures), total=len(futures)):
            p = futures[fut]
            out = fut.result()
            if out:
                rewritten[p] = out
    except KeyboardInterrupt:
        print("interrupted, cancelling...")
        for fut in futures:
            fut.cancel()
        ex.shutdown(wait=False, cancel_futures=True)
        ordered = save_outputs()
        print(f"wrote {len(ordered)} rewritten prompts to {OUT}")
        os._exit(130)
    else:
        ex.shutdown()

    ordered = save_outputs()
    print(f"wrote {len(ordered)} rewritten prompts to {OUT}")


if __name__ == "__main__":
    main()
