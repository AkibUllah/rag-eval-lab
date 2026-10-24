import pytest

from rag_eval_lab.documents import Chunk
from rag_eval_lab.evaluation import metrics as m
from rag_eval_lab.evaluation.dataset import Evidence, covers
from rag_eval_lab.generation.base import Answer


def chunk(doc, idx, start, end, text="x"):
    return Chunk(doc_id=doc, index=idx, text=text, start=start, end=end, search_text=text)


EV = [Evidence("fees", "…", 100, 140), Evidence("fees", "…", 300, 320)]


def test_covers_requires_half_of_the_evidence_span():
    assert covers(chunk("fees", 0, 90, 125), EV[0])  # 25/40 covered
    assert not covers(chunk("fees", 0, 90, 105), EV[0])  # 5/40
    assert not covers(chunk("other", 0, 0, 999), EV[0])


def test_rank_metrics_on_a_hand_worked_example():
    ranked = [chunk("fees", 0, 0, 50), chunk("fees", 1, 95, 150), chunk("fees", 2, 290, 330)]
    cov = m.coverage_by_rank(ranked, EV)
    assert cov == [set(), {0}, {1}]
    assert m.recall_at_k(cov, 2, k=2) == 0.5
    assert m.recall_at_k(cov, 2, k=3) == 1.0
    assert m.hit_at_k(cov, 1) == 0.0 and m.hit_at_k(cov, 2) == 1.0
    assert m.reciprocal_rank(cov) == 0.5
    import math

    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert m.ndcg_at_k(cov, 2, k=3) == pytest.approx(expected)


def test_duplicate_chunks_do_not_inflate_ndcg():
    cov = [{0}, {0}, {0}]
    assert m.ndcg_at_k(cov, 2, k=3) == pytest.approx(1 / (1 + 1 / 1.584962500721156))
    assert m.recall_at_k(cov, 2, k=3) == 0.5


def test_no_evidence_and_no_hits():
    assert m.recall_at_k([], 0, 5) == 0.0
    assert m.ndcg_at_k([set()], 0, 5) == 0.0
    assert m.reciprocal_rank([set(), set()]) == 0.0


def test_token_f1_and_exact_match_ignore_citations_and_case():
    assert m.exact_match("At midnight. [limits#4]", "at midnight") == 1.0
    assert m.exact_match("Midnight", "") == 0.0
    assert m.token_f1("The BDT 5 per transaction [send#1]", "BDT 5 per transaction") == 1.0
    assert 0 < m.token_f1("BDT 5 fee", "BDT 5 per transaction") < 1
    assert m.token_f1("", "x") == 0.0


def test_citation_precision():
    assert m.citation_precision(Answer("a", citations=["a#1", "b#2"]), {"a#1"}) == 0.5
    assert m.citation_precision(Answer("a"), {"a#1"}) is None


def test_claims_split_on_citation_runs():
    text = "Fee is 1.5% [a#1]. Minimum 10 [a#1][b#2]. Trailing claim"
    assert m.claims(text) == [
        ("Fee is 1.5%", ["a#1"]),
        ("Minimum 10", ["a#1", "b#2"]),
        ("Trailing claim", []),
    ]


def test_faithfulness_checks_claims_against_cited_chunks():
    ctx = {
        "a#1": chunk("a", 1, 0, 10, "The ATM charge is 1.5% with a minimum of BDT 10."),
        "b#2": chunk("b", 2, 0, 10, "PIN locks after three attempts."),
    }
    good = Answer("The ATM charge is 1.5% [a#1]", citations=["a#1"])
    mixed = Answer(
        "The ATM charge is 1.5% [a#1] Loans cost 9% yearly [b#2]", citations=["a#1", "b#2"]
    )
    assert m.faithfulness(good, ctx) == 1.0
    assert m.faithfulness(mixed, ctx) == 0.5
    assert m.faithfulness(Answer("x", abstained=True), ctx) is None
    assert m.faithfulness(mixed, ctx, support=lambda claim, ev: True) == 1.0


def test_mean_and_percentile():
    assert m.mean([1.0, None, 0.0]) == 0.5
    assert m.mean([None]) is None
    assert m.percentile([5, 1, 3, 2, 4], 50) == 3
    assert m.percentile([], 95) == 0.0
