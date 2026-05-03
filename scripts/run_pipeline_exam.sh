#!/usr/bin/env bash
# Exam-framing pipeline: rewrite both splits to third-person clinical
# vignettes, bench them with the same free-text + Final Risk Assessment
# format as the chat-frame pipeline, judge with the same eval logic, then
# print the combined metrics table.
#
# Rewriters, benches, and evals are all resumable on (prompt, model), so
# rerunning is cheap.

set -euo pipefail

# Rewrites are one-time — uncomment if data/dataset_exam.json or
# data/dataset_real_emergency_exam.json need to be (re)generated.
# python scripts/rewrite_to_exam.py &
# python scripts/rewrite_real_emergency_to_exam.py &
# wait

python scripts/bench_dataset_exam.py &
python scripts/bench_real_emergency_exam.py &
wait

python scripts/eval_exam.py &
python scripts/eval_real_emergency_exam.py &
wait

python scripts/compute_metrics_exam_combined.py
