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
