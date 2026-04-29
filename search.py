#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import sys

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Semantic search over all_chats.json using cached embeddings.")
    parser.add_argument("query", help="Search query string")
    parser.add_argument("n", type=int, help="Number of top results to return")
    parser.add_argument("--data", default="all_chats.json", help="Path to chats JSON")
    parser.add_argument("--embeddings", default="embeddings.pkl", help="Path to embeddings pickle")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = args.data if os.path.isabs(args.data) else os.path.join(base_dir, args.data)
    emb_path = args.embeddings if os.path.isabs(args.embeddings) else os.path.join(base_dir, args.embeddings)

    with open(data_path, "r") as f:
        data = json.load(f)
    with open(emb_path, "rb") as f:
        embeddings = pickle.load(f)

    if embeddings and hasattr(embeddings[0], "embedding"):
        embeddings = [e.embedding for e in embeddings]

    dataset = np.array(embeddings, dtype=np.float32)
    dataset /= np.linalg.norm(dataset, axis=1, keepdims=True)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    resp = client.embeddings.create(
        model="google/gemini-embedding-2-preview",
        input=f"task: search result | query: {args.query}",
        encoding_format="float",
    )
    query = np.array(resp.data[0].embedding, dtype=np.float32)
    query /= np.linalg.norm(query)

    scores = dataset @ query
    top_idx = np.argsort(-scores)[: args.n]

    for rank, idx in enumerate(top_idx, 1):
        print(f"=== {rank}. idx={int(idx)} score={scores[idx]:.4f} ===")
        print(data[int(idx)])
        print()


if __name__ == "__main__":
    main()
