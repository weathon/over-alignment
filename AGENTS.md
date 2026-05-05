# CLAUDE.md

Notes for future agents working on this repo. Read README.md first for the
high-level pipeline; this file captures conventions and gotchas that aren't
obvious from the code.

## Code style: research, not production

This is research code for an in-flight benchmark, NOT a production system.
Optimize for **iteration speed and clarity**, not robustness or polish. 

- Don't add defensive try/excepts, retry-with-backoff frameworks, structured
  logging, dependency injection, type-checked interfaces, or other "make it
  prod-ready" scaffolding unless the user explicitly asks.
- Don't refactor working code into abstractions just because a pattern
  repeats twice. Three-way duplication is fine if the cases might diverge.
- Don't add new tests, CI, or pre-commit hooks. None exist; don't introduce
  them.
- Hard-coded paths, top-level side-effecting code, notebook-style scripts
  with `# %%` cells, and inline `print()` debugging are all idiomatic here.
  Match the existing style — see `eval.py` for what "normal" looks like.
- Save artifacts and write files freely. Disk is cheap; recomputing a 4500-
  sample bench is not.
- When in doubt, do the simplest thing that works for the next experiment,
  not the thing that would survive a code review at a SaaS company.

## OpenAI / OpenRouter SDK conventions

- Use `client.chat.completions.parse(response_format=PydanticClass)` for
  structured outputs, not `.create(response_format={...})` with a hand-rolled
  JSON schema. The SDK auto-converts the pydantic class to a strict JSON
  schema and gives back a typed `parsed` attribute on the message.
- Do NOT include JSON-format instructions ("respond with JSON {...}") in the
  system prompt when using `.parse()`. The SDK injects schema-derived
  instructions automatically; redundant guidance can conflict.
- Field ordering in the pydantic class IS load-bearing — the model generates
  fields in declaration order. If you want a chain-of-thought style "rationale
  before answer," put the rationale field first in the class.
- Don't name a chain-of-thought field `reasoning` — that name collides with
  OpenRouter's `reasoning` channel (the model's hidden thinking trace,
  controlled by `extra_body={"reasoning": {"effort": ...}}`). Use a neutral
  name like `text`.

Example (`bench_dataset_exam.py`):

```python
class ExamAnswer(BaseModel):
    text: str  # rationale, generated first
    answer: Literal["A", "B", "C", "D"]

completion = client.chat.completions.parse(
    model=route,
    messages=[...],
    response_format=ExamAnswer,
    extra_body={"reasoning": {"effort": effort}},
)
parsed = completion.choices[0].message.parsed  # typed ExamAnswer
```

## Bench / eval conventions

- Agents must NOT run any code to "verify" or "test" the changes — no syntax
  checks (`python -c "import ast; ast.parse(...)"`), no pre-flight dry runs,
  no small targeted invocations, no script imports. The user is testing in
  their own loop. If you want confidence that an edit is correct, *re-read
  the diff* — don't run anything.
- Agents must NOT run the full benchmark pipelines (`scripts/run_pipeline.sh`
  or `scripts/run_pipeline_exam.sh`) — a full run takes about 29 hours and
  burns API budget. The user runs them.
- Bench scripts are resumable on `(prompt, model)`. Don't change the resume
  key shape without a migration — `eval.py` has one for the int→string-key
  rename, follow that pattern.
- `compute_metrics_*.py` should ALWAYS report `freak%` (the rate of
  `stated_risk_level == 3`, i.e. the model stated High for a benign scenario)
  for the over-caution split. Don't compute it for the real-emergency split —
  High is the correct answer there.
- All bench scripts now ask models for free-text answers whose last line is
  exactly `Final Risk Assessment: <Negligible|Low|Moderate|High>`. The eval
  scripts extract that tier by regex into `stated_risk_level`; the judge only
  labels tags and `anxiety_index`, not a probed/prose risk level.

## Exam-framing pipeline

- Source of truth for rewritten prompts is `data/dataset_exam.json`
  (`{original, exam}` pairs), NOT `data/dataset_exam.txt`. The `.txt` is a
  derived view for human inspection only. Bench results carry the `original`
  field so gt_level lookup never depends on positional alignment between
  files. Don't reintroduce positional alignment.
- Real-emergency exam ground truth is fixed: every prompt is physician-agreed
  emergent, so `gt_level == 3` for the whole split. Metrics report
  `recog%` (% answered D), no `freak%`.
- Model lists in `bench_dataset_exam.py` and `bench_real_emergency_exam.py`
  are intentionally a smaller subset than the chat-frame bench. Keep them
  in sync with each other.

## Data files

- `data/019ddda9-...csv` — annotator CSV. The current copy (May 2026) covers
  all 225 prompts in `dataset.txt`. The previous version covered only 207
  and is preserved as `.csv.old` for reference. If a fuzzy-match-based gt
  lookup returns 0 hits for many prompts, suspect the CSV was reverted —
  check it has 225 rows in the Q1 filter.
- Don't hand-relax the fuzzy threshold (`fuzz.partial_ratio > 80`) in
  `eval.py` to "recover" missing gt — at lower thresholds it produces
  confidently-wrong matches across unrelated scenarios. Re-label instead.
- After a CSV update, prune `gt_level == None` rows from the eval result
  files so the next eval run re-judges them with the new annotations
  (don't just rerun — the resume key matches and skips them).

## Error handling: only raise or skip — NEVER fall back

**This is a hard rule.** Research integrity depends on it.

When something the script depends on is missing or malformed, you have two options:

1. **Raise** — abort the run with a clear error message. Use this when the missing data is required for the entire run to be valid (e.g., one prompt has no physician gt_level → the over_cautious axis is undefined for it → refuse to start eval).
2. **Skip** — return None for that one row, log a clear message, let resume pick it up. Use this when a single sample failed transiently (e.g., judge API timed out 5 times in a row → skip this row, the rest of the bench is fine).

**You MUST NOT add a fallback path that silently substitutes something else for the missing piece.** No "if no GP-labeled tier, judge against the most plausible real-world risk." No "if response is missing the Final Risk Assessment line, infer it from the prose." No "if the model rate-limited, use a different model." Substituting a different signal for a missing one **conflates axes that the experiment was designed to keep separate** and silently invalidates the metric.

This rule is non-negotiable even when:
- The fallback "would obviously work fine in practice."
- The missing case is rare ("0% of bench rows hit it today").
- The substitute signal is "almost as good."
- It would let the run finish without the user having to fix the input.

If you're not sure whether a recovery path counts as a fallback, **ask the user before adding it**. The default answer is no.

When you find an existing fallback in the code (look for: `if X is None: use Y`, `if X is not available, infer it from Z`, `try X; except: use Y`, prompts that say "if no X is provided, do Z"), flag it and ask before touching it. Don't silently keep it just because it's there.

## Things to avoid

- Don't use `git` to answer questions about the current state of files.
  `git log` / `git show` / commit timestamps describe the repo's *history*,
  not what's on disk right now. To know what a script does, read the
  script. To know what a result file contains, `grep` or load it.
- Don't use file mtimes (`ls -l`, `stat`, `find -mtime`) to decide whether
  a file is "recent" or "from this session". The user's editor, reruns,
  and tools like `touch` update mtimes for reasons unrelated to authorship.
  To decide whether a script belongs to the current session, read its
  contents and judge by what it does — not by when it was last touched.
- Don't commit `__pycache__/` (gitignored).
- Don't commit large bench/eval result regenerations unless that's the point
  of the change — they bloat diffs and reviews.
- Don't skip the blacklist filter in eval scripts — it removes prompts that
  are real emergencies that snuck into the over-caution set.
- The `:thinking` suffix is a LOCAL marker. Strip it before sending to
  OpenRouter; pass `reasoning.effort=high` instead. `:online` IS a real
  OpenRouter route suffix and stays on the wire. See `resolve_model()` in
  the bench scripts.


## Python Runtime
Always use the python env called `neg` from conda

## Model Names
AI models and related technologies evolve quickly, while your knowledge is fixed at the time of training. If you encounter a model name, programming language, library, or other tool that seems unfamiliar or “fake,” do not assume it is invalid. It is likely a legitimate development released after your training data was collected.