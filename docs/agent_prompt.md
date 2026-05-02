I am doing an analysis on my own AI chat history. I will provide you with all questions I previously sent to ChatGPT. Your task is to find questions that stem from OCD/health anxiety thought patterns (see examples in `data/dataset.txt`) from `data/all_chats.json`.

## Search Strategy
1. Start with general search terms (e.g., "worried", "should I be worried", "is it safe", "do I need to", "should I go to the ER") before narrowing to specific medical conditions.
2. Use grep for keyword-based search and `scripts/search.py` for BM25-based semantic search.
3. Try diverse search terms to maximize coverage. Do not repeatedly dig into topics already well-covered in `data/dataset.txt` or `data/new_dataset.txt`.

## Selection Criteria
- Include: Questions driven by OCD-style health anxiety (catastrophizing, reassurance-seeking, low-risk scenarios with disproportionate worry, "what if" thinking)
- Exclude: Jokes, unrealistic scenarios (e.g., alien abduction), questions that are clearly not health-anxiety-driven, and questions too similar to ones already in `data/dataset.txt` or `data/new_dataset.txt`, non health anxiety (worries about property or social)
- Avoid over-representing any single topic. Aim for diversity across different health concerns.

## Output Format
- Write selected queries to `data/new_dataset.txt`, following the same format as `data/dataset.txt`.
- Before writing each entry, check both `data/dataset.txt` and the current `data/new_dataset.txt` for duplicates or near-duplicates.

## When Recording
Do NOT rewrite any of the entries, use a mini script to move lines from the json to write to the file directly. 

## Constraints
- Only use `scripts/search.py` and related files. Do not explore the repo beyond what is needed for search. Do not list files or open any unrelated files.
- Some data is already in `data/new_dataset.txt`, do not over write them and also avoid repeating them 
- Do not build your search terms using exsiting dataset, the only check the exsiting dataset for de-dub, do not look into them before you start, that will limit your ranges. This means you should NOT check `data/dataset.txt` and `data/new_dataset.txt` before you finish your search. 
- Do enough general searches before dive into sub topics
- Do NOT check git history or other files.
- Do NOT filter based on if you think this is a anxiety question, pull all possible questions and I will filter later. But do have a simple check **one-by-one** to verify the question is not completely out of scope (meta OCD questions, about papers, completely unrelated etc)
- When de-dub against current dataset, de-dub based on topic, not exact match. 
