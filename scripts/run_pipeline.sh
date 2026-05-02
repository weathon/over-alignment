python scripts/bench_dataset.py &
python scripts/bench_real_emergency.py &
wait

python scripts/eval.py &
python scripts/eval_real_emergency.py &
wait
