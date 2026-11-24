"""Present measured results with explicit scope, method and interpretation."""
import html

RETRIEVAL = {
    'Recall@k': 'How much of the labeled evidence was found? Higher is better.',
    'Precision@k': 'How much of what we retrieved is labeled useful? Higher means less noise.',
    'MRR': 'How early is the first labeled source? 1 means it ranks first.',
    'nDCG@k': 'Are labeled sources ordered near the top? Higher is better.'}
ANSWER = {
    'Answer accuracy': 'Correct against a curated reference?',
    'Faithfulness': 'Supported by the retrieved passages?',
    'Answer relevance': 'Directly answers the question?',
    'Context relevance': 'Retrieved passages useful for this question?'}


def quote(value):
    return '\n'.join('> '+html.escape(str(line)) for line in str(value).splitlines())


def evaluation_sections(rows, baseline, turn):
    values={name:(str(value),str(note)) for name,value,note in rows}
    def score(name):
        return values.get(name,('Waiting for results',''))[0].replace('|','/')
    retrieval='| Metric | Result | What it checks |\n|---|---|---|\n'+'\n'.join(
        f'| {name} | **{score(name)}** | {meaning} |' for name,meaning in RETRIEVAL.items())
    means=baseline.get('means',{})
    if means:
        recall,precision=means['Recall@k'],means['Precision@k']
        retrieval+=f"\n\n**Discussion:** Across {baseline['count']} labeled questions, we found {recall:.0%} of the expected sources on average; {precision:.0%} of distinct retrieved sources were labeled relevant. These are two different checks."
        if recall >= .9 and precision < .6:
            retrieval+=' Coverage is high, but the returned context contains extra, unlabeled material. Some may still be useful—the labels are not exhaustive.'
        retrieval+=' These results evaluate retrieval only; they do not tell us whether an answer is correct.'
    elif baseline.get('error'):
        retrieval+='\n\n**Benchmark unavailable:** '+html.escape(baseline['error'])
    example='The worked example will appear when the automatic benchmark finishes.'
    if baseline.get('details'):
        item=next((d for d in baseline['details'] if d['metrics']['MRR']<1),baseline['details'][0])
        names=', '.join(item['retrieved_sources'])
        expected=', '.join(item['relevant_sources'])
        example=(f"**Question**\n\n{quote(item['question'])}\n\n"
                 f"**Expected evidence (authored labels):** {html.escape(expected)}\n\n"
                 f"**Actual retrieval, in order (distinct sources):** {html.escape(names)}\n\n"
                 f"**This question's scores:** Recall {item['metrics']['Recall@k']:.3f}; precision {item['metrics']['Precision@k']:.3f}; MRR {item['metrics']['MRR']:.3f}; nDCG {item['metrics']['nDCG@k']:.3f}.\n\n"
                 "We compare source filenames with the curated evidence labels. Multiple chunks from the same source count once. A source found at rank 2 helps recall, but MRR penalizes it being behind another source. This example does not generate or grade an answer.")
    answers='| Check | Latest result | Question it answers |\n|---|---|---|\n'+'\n'.join(
        f'| {name} | **{score(name)}** | {meaning} |' for name,meaning in ANSWER.items())
    answers+=f"\n\n**Response time:** {score('Latency')} · retrieval + generation; excludes judging and startup."
    if not turn:
        discussion='Ask a question in Chat to see the actual answer, reference (when available), and judge explanations here. Suggested questions have curated references.'
    else:
        from .automatic import reference_case
        case=reference_case(baseline,turn['question'])
        discussion=f"**Question**\n\n{quote(turn['question'])}\n\n**Actual answer**\n\n{quote(turn['answer'])}"
        if case:
            discussion+=f"\n\n**Curated reference**\n\n{quote(case['reference'])}\n\nAccuracy compares the actual answer with this reference in a separate local judge call; valid paraphrases can receive credit. It is a model estimate for this one answer, not an overall accuracy percentage."
        else:
            discussion+='\n\n**No curated reference for this question.** Accuracy and token overlap are not measured. Faithfulness can still check support in the supplied passages.'
        discussion+='\n\n**Judge explanations**'
        for name in ANSWER:
            note=values.get(name,('',''))[1]
            reason=note.split('Reason: ',1)[1] if 'Reason: ' in note else score(name)
            discussion+=f'\n\n**{name}:** {html.escape(reason)}'
        discussion+=f"\n\n**Supporting check — token F1: {score('Answer token F1')}.** This measures shared words with the reference. Low overlap may simply be a correct paraphrase; it is not another accuracy score."
    return retrieval,example,answers,discussion
