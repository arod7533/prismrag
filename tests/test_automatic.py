from prismrag.automatic import measurement_rows, AnswerJudgments, judge_turn
from prismrag.evaluation import Judgment
from prismrag.metrics import METRICS

def test_missing_reference_does_not_invent_zero_score():
    rows = {r[0]:r for r in measurement_rows({"means":{"Recall@k":.75},"count":20,"cases":[]}, {"question":"Unlabeled", "answer":"Anything", "seconds":1.2})}
    assert rows['Answer token F1'][1] == 'No reference for this question'
    assert rows['Recall@k'][1] == '0.750'
    assert rows['Faithfulness'][1] == 'Evaluating…'
    assert set(rows) == set(METRICS)

def test_reference_match_and_failed_judge():
    baseline = {"cases":[{"question":"Known question?", "reference":"the answer"}]}
    turn = {"question":" known QUESTION? ","answer":"the answer","seconds":2}
    rows = {r[0]:r for r in measurement_rows(baseline, turn, judge_error='offline')}
    assert rows['Answer token F1'][1] == '1.000'
    assert rows['Faithfulness'][1] == 'Unavailable'
    assert 'offline' in rows['Faithfulness'][2]

def test_judgment_reasons_are_visible():
    j=AnswerJudgments(faithfulness=Judgment(score=.4,reason='Unsupported date'), answer_relevance=Judgment(score=1,reason='Direct'),context_relevance=Judgment(score=.8,reason='Useful'))
    rows={r[0]:r for r in measurement_rows({}, {"question":"q","answer":"a","seconds":1}, judgments=j)}
    assert rows['Faithfulness'][1] == '0.400'
    assert 'Unsupported date' in rows['Faithfulness'][2]
