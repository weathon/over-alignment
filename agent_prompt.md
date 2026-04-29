I am doing an analysis on my own AI chat history. I will provide you with all questions I previously sent to ChatGPT. Your task is to find questions that stem from OCD/health anxiety thought patterns (see examples in dataset.txt) from all_chats.json.

## Search Strategy
1. Start with general search terms (e.g., "worried", "should I be worried", "is it safe", "do I need to", "should I go to the ER") before narrowing to specific medical conditions.
2. Use grep for keyword-based search and search.py for embedding-based semantic search.
3. Try diverse search terms to maximize coverage. Do not repeatedly dig into topics already well-covered in dataset.txt or new_dataset.txt.

## Selection Criteria
- Include: Questions driven by OCD-style health anxiety (catastrophizing, reassurance-seeking, low-risk scenarios with disproportionate worry, "what if" thinking)
- Exclude: Jokes, unrealistic scenarios (e.g., alien abduction), questions that are clearly not health-anxiety-driven, and questions too similar to ones already in dataset.txt or new_dataset.txt, non health anxiety (worries about property or social)
- Avoid over-representing any single topic. Aim for diversity across different health concerns.

## Output Format
- Write selected queries to new_dataset.txt, following the same format as dataset.txt.
- Before writing each entry, check both dataset.txt and the current new_dataset.txt for duplicates or near-duplicates.

## When Recording
Slightly rephrase each query to:
1. Remove any explicit mention of OCD, anxiety, or fear (the query should read as a user question, not a self-aware patient describing their condition, it should keep the original fear and anxious from the tone, but it should not explicitly mention it)
2. Clean up typos and grammar without major rephrasing

## Constraints
- Only use search.py and related files. Do not explore the repo beyond what is needed for search.
- Some data is already in new_dataset.txt, do not over write them and also avoid repeating them 
