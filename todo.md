1. clearify anxiety inducing and risk assesment 
2. check gemma and medgemma
3. 最终结论是：对于大部分query，llm不会夸大风险，不管是自己说的还是prob的，但是会渲染恐慌。对于难的问题（我自己标注难，但是同时GP标注风险低），还是有over cautious。写的时候和health bench一样
4. 重跑一次，模型不自己输出risk！这个很重要（？）
5. 用minor risk的GP来做
6. Fix compute_metrics.py: it incorrectly treats the small human annotation subset as per-model metric data via id_model_map.
