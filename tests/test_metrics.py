import pytest
from prismrag.metrics import retrieval_metrics, token_f1

def test_ranking_and_duplicate_sources():
    result = retrieval_metrics(["noise", "a", "a", "b"], ["a", "b"])
    assert result["Precision@k"] == pytest.approx(2/3)
    assert result["Recall@k"] == 1
    assert result["MRR"] == .5
    assert 0 < result["nDCG@k"] < 1

def test_no_hits_and_empty_retrieval():
    assert all(v == 0 for v in retrieval_metrics([], ["a"]).values())
    assert retrieval_metrics(["b"], ["a"])["MRR"] == 0
    with pytest.raises(ValueError):
        retrieval_metrics(["a"], [])

def test_token_counts():
    assert token_f1("red red blue", "red blue") == pytest.approx(.8)
    assert token_f1("", "x") == 0
