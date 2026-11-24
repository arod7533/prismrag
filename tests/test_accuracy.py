from prismrag.automatic import AnswerJudgments, QualityJudgment, measurement_rows
from prismrag.evaluation import Judgment


def test_accuracy_requires_reference_even_if_judge_invents_one():
    judgments=AnswerJudgments(faithfulness=Judgment(score=1,reason='Supported'),answer_relevance=Judgment(score=1,reason='Direct'),context_relevance=Judgment(score=1,reason='Useful'),answer_accuracy=QualityJudgment(score=1,reason='Correct'))
    turn={'question':'Unknown?','answer':'Answer','seconds':1}
    rows={r[0]:r for r in measurement_rows({},turn,judgments)}
    assert rows['Answer accuracy'][1]=='Not measured · no reference'


def test_accuracy_and_faithfulness_can_differ():
    judgments=AnswerJudgments(faithfulness=Judgment(score=1,reason='Follows outdated context'),answer_relevance=Judgment(score=1,reason='Direct'),context_relevance=Judgment(score=.5,reason='Old policy'),answer_accuracy=QualityJudgment(score=0,reason='Conflicts with current reference'))
    turn={'question':'Known?','answer':'Old price','seconds':1}
    baseline={'cases':[{'question':'Known?','reference':'New price'}]}
    rows={r[0]:r for r in measurement_rows(baseline,turn,judgments)}
    assert rows['Faithfulness'][1]=='1.000'
    assert rows['Answer accuracy'][1]=='0.000'
