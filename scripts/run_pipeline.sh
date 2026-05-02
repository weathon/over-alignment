#!/usr/bin/env bash
# End-to-end rerun: generate any missing bench responses, then judge them.
#
# All four scripts resume on (prompt, model) content keys, so re-running is
# idempotent — finished rows are skipped, only new ones (e.g. :thinking
# variants, or anything that failed last time) actually hit the API.
#
# Order matters: judges read the bench output files, so benches go first.
#
# Usage:
#     bash scripts/run_pipeline.sh

set -euo pipefail

PY=/home/wg25r/miniconda/envs/neg/bin/python
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== [1/4] over-caution bench → data/results.json ==="
"$PY" scripts/bench_dataset.py

echo
echo "=== [2/4] real-emergency bench → results/results_real_emergency.json ==="
"$PY" scripts/bench_real_emergency.py

echo
echo "=== [3/4] over-caution judge → results/eval_results.json ==="
"$PY" scripts/eval.py

echo
echo "=== [4/4] real-emergency judge → results/eval_results_real_emergency.json ==="
"$PY" scripts/eval_real_emergency.py

echo
echo "done."
