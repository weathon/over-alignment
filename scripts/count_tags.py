"""Count judge tag frequencies per model.

Usage:
    conda run -n neg python scripts/count_tags.py [results/eval_results.json ...]

The counts are row counts: if tag 4 appears in a model's judged response, that
adds 1 to tag 4 for that model.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


TAG_NAMES = {
    1: "Symptoms Checking",
    2: "Acknowledge Low Risk",
    3: "Catastrophic Thinking",
    4: "Better Safe Than Sorry",
    5: "Provide Anxiety Help",
    6: "Reassurance Undermining",
}


def iter_samples(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key, sample in data.items():
            yield key, sample
    elif isinstance(data, list):
        for i, sample in enumerate(data):
            yield str(i), sample
    else:
        raise TypeError(f"{path} must contain a JSON dict or list, got {type(data).__name__}")


def count_path(path):
    totals = Counter()
    no_tags = Counter()
    counts = defaultdict(Counter)

    for key, sample in iter_samples(path):
        model = sample["model"]
        tags = sample["tags"]
        if not isinstance(tags, list):
            raise TypeError(f"{path}:{key} tags must be a list, got {type(tags).__name__}")
        totals[model] += 1
        if not tags:
            no_tags[model] += 1
        for tag in tags:
            tag = int(tag)
            if tag not in TAG_NAMES:
                raise ValueError(f"{path}:{key} has unknown tag {tag}")
            counts[model][tag] += 1

    print(f"\n=== {path} ===")
    print(f"total judged rows: {sum(totals.values())}")
    print()
    for tag, name in TAG_NAMES.items():
        print(f"{tag}: {name}")
    print()

    header = f"{'Model':<40} {'N':>5} {'none':>9} " + " ".join(f"tag{tag:>1}".rjust(9) for tag in TAG_NAMES)
    print(header)
    print("-" * len(header))
    for model in sorted(totals):
        n = totals[model]
        cells = [f"{no_tags[model]:>4} {no_tags[model] / n * 100:>4.1f}%"]
        for tag in TAG_NAMES:
            count = counts[model][tag]
            cells.append(f"{count:>4} {count / n * 100:>4.1f}%")
        print(f"{model:<40} {n:>5} " + " ".join(cells))


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=[root / "results" / "eval_results.json"])
    args = parser.parse_args()
    for path in args.paths:
        count_path(path)


if __name__ == "__main__":
    main()
