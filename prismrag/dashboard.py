"""A compact scorecard. Retrieval coverage is never presented as answer accuracy."""
import html
import math
from .automatic import reference_case


def simple_dashboard(rows, baseline, turn):
    values={name:(str(value),str(note)) for name,value,note in rows}
    recall=baseline.get('means',{}).get('Recall@k')
    evidence=f'{recall:.0%}' if recall is not None else 'Unavailable' if baseline.get('error') else 'Loading…'
    evidence_note=f"Expected sources found · {baseline.get('count',0)} test questions" if recall is not None else 'Document retrieval test'
    accuracy,note=values.get('Answer accuracy',('',''))
    case=reference_case(baseline,turn['question']) if turn else None
    accuracy_value,accuracy_note='Not scored','Ask a suggested question'
    if turn and not case:
        accuracy_note='No reference for this question'
    elif turn and case:
        try:
            number=float(accuracy)
            if not math.isfinite(number) or not 0<=number<=1:
                raise ValueError('Invalid score')
            accuracy_value=f'{number*100:.0f} / 100'
            accuracy_note='Latest answer · model estimate'
        except ValueError:
            accuracy_value='Checking…' if 'Evaluating' in accuracy else 'Unavailable'
            accuracy_note='Latest answer · reference comparison'
    seconds=turn.get('seconds') if turn else None
    latency=f'{seconds:.1f}s' if seconds is not None else '—'
    cards=[('Evidence found',evidence,evidence_note),('Answer accuracy',accuracy_value,accuracy_note),('Response time',latency,'Latest answer · excludes evaluation')]
    scorecard='<div class="scorecards">'+''.join(
        f'<section class="scorecard"><div class="scorecard-label">{html.escape(label)}</div><div class="scorecard-value">{html.escape(value)}</div><div class="scorecard-note">{html.escape(caption)}</div></section>' for label,value,caption in cards)+'</div>'
    def compact_score(name, retrieval=False):
        raw = baseline.get('means',{}).get(name) if retrieval else values.get(name,('', ''))[0]
        try:
            number = float(raw)
            if not math.isfinite(number) or not 0 <= number <= 1:
                raise ValueError('Invalid score')
            return f'{number:.3f}'
        except (TypeError, ValueError):
            return 'Checking…' if 'Evaluating' in str(raw) else '—'
    groups = [
        (f"Retrieval quality · {baseline.get('count',0)} test questions" if baseline.get('means') else 'Retrieval quality · test questions', True,
         [('Precision@k', 'How much retrieved evidence is labeled useful'), ('MRR', 'How early the first relevant source appears'), ('nDCG@k', 'How well relevant sources are ranked')]),
        ('Answer quality · latest answer · model estimates', False,
         [('Faithfulness', 'Claims supported by retrieved passages'), ('Answer relevance', 'How directly the answer addresses the question'), ('Context relevance', 'How useful the retrieved passages are')])]
    for title, is_retrieval, metrics in groups:
        scorecard += f'<h3 class="metric-group-title">{html.escape(title)}</h3><div class="metric-strip">'
        for name, caption in metrics:
            scorecard += (f'<section class="metric-item"><div class="scorecard-label">{html.escape(name)}</div>'
                          f'<div class="metric-value">{compact_score(name,is_retrieval)}</div>'
                          f'<div class="scorecard-note">{html.escape(caption)}</div></section>')
        scorecard += '</div>'
    scorecard += '<p class="scorecard-note">Supporting scores range from 0 to 1. Higher is better. Evidence found above is recall.</p>'
    if baseline.get('error'):
        summary='Evidence check unavailable. '+html.escape(str(baseline['error']))
    elif not turn:
        summary='Evidence check complete. Ask a suggested question to check answer accuracy.' if recall is not None else 'Checking the documents…'
    elif not case:
        summary='No reference answer is available for this question, so accuracy is not scored.'
    elif accuracy_value=='Checking…':
        summary='Checking the latest answer against its reference…'
    elif accuracy_value=='Unavailable':
        summary='The answer check was unavailable. The evidence result is still valid.'
    else:
        summary='Accuracy compares your latest answer with a reference. It is a model estimate, not overall pipeline accuracy.'
    details='**How scores are calculated**\n\nEvidence found: average share of labeled sources retrieved across the bundled test questions. Answer accuracy: a local model compares the latest answer with its reference. Response time: retrieval plus answer generation.\n\n'
    details+='| Measurement | Result | Applies to |\n|---|---|---|\n'
    for name,(value,_) in values.items():
        scope=f"{baseline.get('count',0)} test questions" if name in {'Recall@k','Precision@k','MRR','nDCG@k'} else 'Latest answer'
        details+=f"| {html.escape(name)} | {html.escape(value).replace('|','/')} | {scope} |\n"
    if turn:
        details+='\n**Question:** '+html.escape(turn['question'])+'\n\n**Answer:** '+html.escape(turn['answer'])
        if case:
            details+='\n\n**Reference:** '+html.escape(case['reference'])
        for name,(_,description) in values.items():
            if 'Reason: ' in description:
                details+='\n\n**'+html.escape(name)+':** '+html.escape(description.split('Reason: ',1)[1])
    details+='\n\nThe test questions are demonstration data. The same local model answers and evaluates; scores can be wrong. Word overlap and faithfulness are supporting checks, not overall accuracy.'
    return scorecard,summary,details
