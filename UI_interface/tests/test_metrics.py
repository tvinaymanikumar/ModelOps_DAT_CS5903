import pytest

from evaluation.evaluator import aggregate, evaluate_document, evaluate_transcription
from evaluation.key_matching import match_keys
from evaluation.metrics import cer, nls, wer


def test_key_matching_exact_then_fuzzy():
    m = match_keys(["Payee Name", "amount_in_words"], ["payee_name", "amount in word"])
    assert m == {"Payee Name": "payee_name", "amount_in_words": "amount in word"}


def test_anls_threshold():
    assert nls("abcd", "abcd") == 1.0
    assert nls("abcd", "abce") == pytest.approx(0.75)
    assert nls("abcd", "wxyz") == 0.0


def test_counting_deepseek_worst_case():
    gt = {"payee_name": "Kishore Raj Shukla", "amount_in_words": "Seventy Three Lakhs", "amount": 7300000,
          "year": "2016", "account_no": "630801551452"}
    pred = {"payee_name": "kishore das shukla", "amount_in_words": "twenty three lakh", "amount": 23000000,
            "year": "2016", "account_no": "630801551452"}
    d = evaluate_document("c", gt, pred)
    assert (d.tp, d.fp, d.fn) == (2, 3, 0)
    s = d.summary()
    assert s["fla"] == pytest.approx(0.4) and s["exact"] is False


def test_missing_and_extra():
    d = evaluate_document("x", {"a": "1", "b": "2"}, {"a": "1", "c": "3"}, fuzzy_threshold=0.95)
    assert (d.tp, d.fp, d.fn) == (1, 1, 1)


def test_aggregate_micro_and_doc_em():
    d1 = evaluate_document("1", {"a": "1"}, {"a": "1"})
    d2 = evaluate_document("2", {"a": "1", "b": "2"}, {"a": "1", "b": "9"})
    agg = aggregate([d1, d2])
    assert agg["fla"] == pytest.approx(2 / 3)
    assert agg["document_exact_match"] == pytest.approx(0.5)
    assert agg["semantic_similarity"] is None


def test_cer_wer_and_transcription():
    assert cer("abc", "abd") == pytest.approx(1 / 3)
    assert wer("a b c", "a x c") == pytest.approx(1 / 3)
    d = evaluate_transcription("t", "hello world\nsecond line", "hello world\nsecond lime")
    agg = aggregate([d], "transcription")
    assert agg["cer"] > 0 and agg["line_exact_match"] == pytest.approx(0.5)
