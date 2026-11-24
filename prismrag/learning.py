"""Explain measurements and suggest experiments from observed results."""
from .metrics import METRICS


def learning_guidance(index, baseline, turn=None, judgments=None):
    lines = ["### What the current results suggest"]
    means = baseline.get('means',{})
    if means:
        recall, precision = means['Recall@k'], means['Precision@k']
        lines.append(f"- **Evidence coverage:** labeled-source recall is {recall:.0%}; precision is {precision:.0%} on {baseline['count']} bundled development questions.")
        if recall >= .9 and precision < .6:
            lines.append("- **First experiment: reduce retrieval noise.** Coverage is high but some returned sources are unlabeled. Try k=3 instead of 4 or add a reranker; compare recall and nDCG before keeping the change. The labels may omit other useful evidence, so inspect examples too.")
        elif recall < .9:
            lines.append("- **First experiment: recover missing evidence.** Inspect missed questions and their chunk boundaries; try a larger k or heading-based chunks before changing the answer model.")
        else:
            lines.append("- **Next experiment: test harder questions.** Add paraphrases, multi-document and unanswerable cases to a held-out evaluation set.")
        if means['MRR'] < 1:
            lines.append("- **Ranking still has room to improve.** The first source is not always labeled relevant. Inspect current versus archived pricing and try a reranker or explicit current-policy metadata filter.")
    else:
        lines.append("- Retrieval recommendations appear automatically once the bundled benchmark finishes.")
    if index is not None:
        from .topics import topic_for
        import numpy as np
        labels = [topic_for(doc) for doc in index.chunks]
        sources = [doc.metadata['source'] for doc in index.chunks]
        sims = index.vectors @ index.vectors.T
        agreements = []
        for i in range(len(labels)):
            candidates = [j for j in range(len(labels)) if sources[j] != sources[i]]
            if candidates:
                nearest = max(candidates, key=lambda j: sims[i,j])
                agreements.append(labels[i] == labels[nearest])
        if agreements and np.mean(agreements) < .5:
            lines.append(f"- **Inspect the mixed topic neighborhoods.** Only {np.mean(agreements):.0%} of nearest chunks from other documents share the topic label. Review the broad labels and overlapping policies first. Then test moving repeated boilerplate into metadata or splitting by headings; compare recall on the same questions instead of forcing a prettier plot.")
    if judgments:
        if judgments.faithfulness.score < .8:
            lines.append("- **Review unsupported claims.** The local judge gave low faithfulness. Verify its reason against the actual passages; if correct, tighten grounding and require abstention when context is missing.")
        accuracy = getattr(judgments,'answer_accuracy',None)
        if accuracy and accuracy.score < .8:
            lines.append("- **Check the reference mismatch.** Review the answer and reference first. Missing evidence points to retrieval; available evidence with a wrong answer points to prompting or generation. The judge can also be wrong.")
        if judgments.answer_relevance.score < .8:
            lines.append("- **Check whether the answer addresses the question.** Try a prompt that asks for a direct answer first; do not assume a more fluent response is more accurate.")
    if turn and turn['seconds'] > 10:
        lines.append("- **Latency is over 10 seconds in this run.** Profile cold model loading separately; then compare a smaller model or shorter context while retaining the same evaluation set.")
    lines.append("\nSuggestions are experiments, not automatic fixes. Change one setting at a time and compare the same labeled questions. The bundled set is development data, so good scores do not establish general accuracy.")
    return '\n\n'.join(lines)


def metric_glossary():
    extra = {
        "Precision@k": "Example: one labeled source among four distinct retrieved sources gives 0.25. Lower precision can add distracting context.",
        "Recall@k": "Example: finding one of two labeled sources gives 0.50. Missing evidence limits what the answer model can know.",
        "MRR": "A first relevant result at rank 2 gives 0.50. This exposes ranking errors that recall alone misses.",
        "nDCG@k": "Finding all relevant sources late can give full recall but lower nDCG. A reranker may help ordering.",
        "Answer accuracy": "A faithful answer can still be wrong if it follows outdated evidence. Accuracy compares with the curated reference, and is a fallible local-judge estimate for the latest labeled question, not a percentage correct across all chats.",
        "Faithfulness": "An invented price is unfaithful even if the response sounds plausible. Context support and real-world truth are separate checks.",
        "Answer relevance": "A factually correct discussion of security does not answer a question about pricing.",
        "Context relevance": "Checks passage usefulness without needing reference labels. The model may judge relevant distractors too generously.",
        "Answer token F1": "A correct paraphrase can score poorly. F1 is a lexical overlap score, not an accuracy score.",
        "Latency": "A faster response is useful only if answer quality remains acceptable. Judge time is excluded from this measurement."
    }
    return '\n\n'.join(f"**{name}.** {description} {extra.get(name,'')}" for name,description in METRICS.items())
