from evaluation.kv_metrics import aggregate_rows, evaluate_pair, field_rows


def test_exact_and_normalized_values_count_as_tp():
    r = evaluate_pair({"patient_name": "Anita", "age": "12 years", "sex": "F", "amount": "Rs. 22,23,000"},
                      {"name_of_patient": "anita", "age": "12", "gender": "female", "amount": "2223000"},
                      mode="schemafree")
    assert (r["tp"], r["fp"], r["fn"]) == (4, 0, 0)
    assert r["combined_kv_f1_pct"] == 100.0
    assert r["strict_form_correct"] is False  # strict check uses no domain normalization


def test_wrong_missing_extra():
    r = evaluate_pair({"a": "1", "b": "2", "c": "3"}, {"a": "1", "b": "9", "z": "x"}, mode="schema")
    assert (r["tp"], r["fp"], r["fn"]) == (1, 2, 2)
    outcomes = sorted(x["outcome"] for x in field_rows({"a": "1", "b": "2", "c": "3"}, {"a": "1", "b": "9", "z": "x"},
                                                       mode="schema"))
    assert outcomes == ["correct", "extra", "missing", "wrong"]


def test_missing_prediction_and_aggregate():
    rows = [evaluate_pair({"a": "1"}, {"a": "1"}), evaluate_pair({"a": "1", "b": "2"}, None)]
    agg = aggregate_rows(rows)
    assert agg["documents_evaluated"] == 2 and agg["strict_form_accuracy_pct"] == 50.0
    assert (agg["combined_kv_tp"], agg["combined_kv_fp"], agg["combined_kv_fn"]) == (1, 0, 2)
