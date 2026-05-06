#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

cleanup() {
    set +e
    echo "interrupted, killing pipeline jobs..."
    jobs -pr | xargs -r kill
    wait
    exit 130
}

trap cleanup INT TERM

python scripts/bench_dataset.py &
python scripts/bench_real_emergency.py &
wait

python scripts/eval_exam.py &
python scripts/eval_real_emergency.py &
wait

python scripts/compute_metrics.py
python scripts/compute_metrics_real_emergency.py
