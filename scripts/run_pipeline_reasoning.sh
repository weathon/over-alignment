#!/usr/bin/env bash
# Reasoning-effort sweep pipeline. Mirrors run_pipeline.sh but for the
# {gpt-5.5, gemini-3-flash, claude-sonnet-4.6} × {minimal, low, medium, high}
# sweep, writing to a separate set of JSON files.

set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/bench_dataset_reasoning.py        & python scripts/bench_real_emergency_reasoning.py & wait

python scripts/eval_reasoning.py                 & python scripts/eval_real_emergency_reasoning.py  & wait
