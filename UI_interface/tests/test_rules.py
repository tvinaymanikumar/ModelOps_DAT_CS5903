from validation.rules import run_rules


def test_rules_for_task():
    from validation.rules import rules_for_task
    fields = [{"key": "a", "type": "date", "required": True}, {"key": "b", "type": "string", "required": False}]
    specs = rules_for_task("schema_based", fields)
    assert "required_not_empty:a" in specs and "date_parseable:a" in specs and "required_not_empty:b" not in specs
    assert rules_for_task("transcription") == ["transcription_not_empty"]
    assert "schema_free_min_fields:3" in rules_for_task("schema_free")


def test_required_not_empty():
    assert run_rules({"a": ""}, ["required_not_empty:a"])
    assert run_rules({"a": "x"}, ["required_not_empty:a"]) == []


def test_other_rules():
    assert run_rules({"account_no": "6308O1"}, ["digits_only:account_no"])
    assert run_rules({"year": "2061"}, ["year_range:year:1990:2035"])
    assert run_rules({"date": "19/10/2018"}, ["date_parseable:date"]) == []
    assert run_rules({"total": "RM 60.30"}, ["numeric:total"]) == []
    assert run_rules({"name": "", "age": "", "sex": ""}, ["non_empty_ratio:0.67"])
    assert run_rules({"x": 1}, ["no_such_rule"])[0]["rule"] == "unknown_rule"
