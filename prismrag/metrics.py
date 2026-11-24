"""Deterministic retrieval metrics use binary, document-level evidence labels."""
import math

METRICS = {
    "Answer accuracy": "Local judge checks correctness and completeness against a curated reference answer (0–1). Available only for an exact bundled question; not measured for unlabeled questions.",
    "Precision@k": "Fraction of retrieved sources that are labeled relevant. Higher is better; measures noise. Repeated chunks from one source count once.",
    "Recall@k": "Fraction of all labeled relevant sources retrieved. Higher is better; measures missing evidence.",
    "MRR": "Reciprocal rank of the first relevant source. 1 means the first source is relevant; 0 means none were found.",
    "nDCG@k": "Discounted gain divided by the ideal ranking for binary relevance. Rewards putting relevant sources early.",
    "Answer token F1": "Token overlap with the reference answer. Useful for regression checks, but penalizes valid paraphrases and does not prove factuality.",
    "Faithfulness": "Local LLM judge scores whether answer claims follow from retrieved context (0–1), with a reason. A fallible estimate, not ground truth.",
    "Answer relevance": "Local LLM judge scores how directly the answer addresses the question (0–1), with a reason. Does not measure factual accuracy.",
    "Context relevance": "Local LLM judge scores usefulness of the retrieved passages for the question (0–1), with a reason. Does not replace labeled recall.",
    "Latency": "Wall-clock seconds for retrieval plus answer generation. Excludes judge time and indexing. Lower is better; warm and cold runs differ."
}

def retrieval_metrics(retrieved, relevant):
    ranked = list(dict.fromkeys(retrieved))
    gold = set(relevant)
    if not gold:
        raise ValueError("Relevant sources must be nonempty for labeled retrieval metrics.")
    hits = [int(source in gold) for source in ranked]
    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(gold), len(ranked))))
    return {"Precision@k": sum(hits) / len(ranked) if ranked else 0.0,
            "Recall@k": sum(hits) / len(gold),
            "MRR": next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0),
            "nDCG@k": dcg / ideal if ideal else 0.0}

def token_f1(answer, reference):
    import re
    from collections import Counter
    a, b = (Counter(re.findall(r"\w+", text.lower())) for text in (answer, reference))
    overlap = sum((a & b).values())
    return 2 * overlap / (sum(a.values()) + sum(b.values())) if a and b else 0.0
