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


def test_supporting_retrieval_scores_use_benchmark_means():
    cards,_,_=simple_dashboard(measurement_rows(B),B,None)
    assert all(name in cards for name in ['Precision@k','MRR','nDCG@k'])
    assert '0.350' in cards and '0.900' in cards and '0.920' in cards
    assert cards.count('class="scorecard"')==3
    assert cards.count('class="metric-item"')==6
    assert '20 test questions' in cards


def test_answer_support_scores_dont_require_reference_or_invent_zero():
    turn={'question':'Unknown','answer':'Hello','seconds':2}
    rows=[['Faithfulness','.7',''],['Answer relevance','.8',''],['Context relevance','Evaluating…','']]
    cards,_,_=simple_dashboard(rows,B,turn)
    assert '0.700' in cards and '0.800' in cards
    assert 'Checking…' in cards
    assert 'Not scored' in cards
    empty,_,_=simple_dashboard(measurement_rows(B),B,None)
    assert '0.000' not in empty
