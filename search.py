#!/usr/bin/env python3
import argparse
import json
import os
import re

from rank_bm25 import BM25Okapi


def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def main():
    parser = argparse.ArgumentParser(description="BM25 search over all_chats.json.")
    parser.add_argument("query", help="Search query string")
    parser.add_argument("n", type=int, help="Number of top results to return")
    parser.add_argument("--data", default="all_chats.json", help="Path to chats JSON")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = args.data if os.path.isabs(args.data) else os.path.join(base_dir, args.data)

    with open(data_path, "r") as f:
        data = json.load(f)

    corpus = [s if isinstance(s, str) else "" for s in data]
    tokenized_corpus = [tokenize(s) for s in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    query_tokens = tokenize(args.query)
    scores = bm25.get_scores(query_tokens)

    top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[: args.n]

    for rank, idx in enumerate(top_idx, 1):
        print(f"=== {rank}. idx={idx} score={scores[idx]:.4f} ===")
        print(data[idx])
        print()


if __name__ == "__main__":
    main()
