from prismrag.dashboard import simple_dashboard
from prismrag.automatic import measurement_rows

B={'count':20,'means':{'Recall@k':1.,'Precision@k':.35,'MRR':.9,'nDCG@k':.92},'cases':[{'question':'Price?','reference':'149'}]}


def test_coverage_is_not_presented_as_answer_accuracy():
    cards,summary,details=simple_dashboard(measurement_rows(B),B,None)
    assert '100%' in cards
    assert 'Not scored' in cards
    assert '20 test questions' in cards
    assert '100 / 100' not in cards


def test_unknown_question_never_gets_accuracy_even_with_a_numeric_judge_row():
    turn={'question':'Unknown','answer':'Hello','seconds':2}
    cards,summary,details=simple_dashboard([['Answer accuracy','1.000','']],B,turn)
    assert 'Not scored' in cards
    assert '100 / 100' not in cards
    assert 'No reference' in summary


def test_known_answer_score_and_timing_have_clear_scope():
    turn={'question':'Price?','answer':'149','seconds':2.1}
    cards,summary,details=simple_dashboard([['Answer accuracy','.8','Reason: Correct amount']],B,turn)
    assert '80 / 100' in cards
    assert 'Latest answer · model estimate' in cards
    assert '2.1s' in cards
    assert '**Reference:** 149' in details
    assert 'Correct amount' in details
