from prismrag.evaluation_view import evaluation_sections
from prismrag.automatic import measurement_rows


def baseline():
    scores={'Recall@k':1.,'Precision@k':.5,'MRR':.5,'nDCG@k':.63}
    return {'count':1,'means':scores,'cases':[{'question':'Price?','reference':'$149'}],
            'details':[{'question':'Price?','relevant_sources':['current.md'],
                        'retrieved_sources':['archive.md','current.md'],'metrics':scores}]}


def test_retrieval_scope_and_real_walkthrough():
    b=baseline()
    retrieval,example,answer,discussion=evaluation_sections(measurement_rows(b),b,None)
    assert '1 labeled questions' in retrieval
    assert 'they do not tell us whether an answer is correct' in retrieval
    assert 'archive.md, current.md' in example
    assert 'MRR 0.500' in example
    assert 'does not generate or grade an answer' in example
    assert 'Ask a question in Chat' in discussion
    assert 'Reference' not in answer  # no invented reference or answer on startup


def test_unlabeled_answer_is_not_given_accuracy_or_reference():
    b=baseline();turn={'question':'Other question?','answer':'<script>hi</script>','seconds':2}
    _,_,answer,discussion=evaluation_sections(measurement_rows(b,turn),b,turn)
    assert 'Not measured · no reference' in answer
    assert 'No curated reference' in discussion
    assert '<script>' not in discussion
    assert '&lt;script&gt;' in discussion


def test_labeled_reference_and_lexical_check_are_separate():
    b=baseline();turn={'question':'Price?','answer':'$149','seconds':2}
    _,_,_,discussion=evaluation_sections(measurement_rows(b,turn),b,turn)
    assert 'Curated reference' in discussion
    assert 'not an overall accuracy percentage' in discussion
    assert 'not another accuracy score' in discussion
