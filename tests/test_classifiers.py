import csv
from pathlib import Path

import pytest

from wofc import keywords
from wofc.llm import LLMClassifier, parse, prompt, validate
from wofc.taxonomy import COMPONENTS, FAILURES, PAIRS, VALID, is_valid

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    with open(ROOT / "data" / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_taxonomy_pairs_are_consistent():
    assert all(f in FAILURES for fs in VALID.values() for f in fs)
    assert set(VALID) == set(COMPONENTS) and len(PAIRS) == 26
    assert is_valid("seal", "leak") and not is_valid("sensor", "leak")


def test_data_is_labeled_with_valid_pairs_and_split_by_wording():
    train, test = load("train.csv"), load("test.csv")
    assert len(train) == 240 and len(test) == 120
    assert all(is_valid(r["component"], r["failure"]) for r in train + test)
    assert {r["wording"] for r in train} == {"seen"}
    assert sum(r["wording"] == "new" for r in test) == 60
    assert {(r["component"], r["failure"]) for r in train} == set(PAIRS)  # every pair is in training


def test_data_generation_is_repeatable(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_data", ROOT / "scripts" / "make_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "OUT", tmp_path)
    module.main()
    assert (tmp_path / "test.csv").read_text() == (ROOT / "data" / "test.csv").read_text()


def test_keywords_match_or_stay_silent():
    assert keywords.classify("hyd hose on P-101 leaking, replaced") == {"component": "hydraulic hose",
                                                                         "failure": "leak"}
    assert keywords.classify("pressure line weeping") == {"component": None, "failure": None}


def test_tfidf_learns_the_seen_wording():
    pytest.importorskip("sklearn")
    from wofc.ml import TfidfClassifier

    train = load("train.csv")
    model = TfidfClassifier().fit([r["text"] for r in train], [r["component"] for r in train],
                                  [r["failure"] for r in train])
    p = model.classify("brg on CV-3 running hot, monitoring")
    assert (p["component"], p["failure"]) == ("bearing", "overheating")
    strict = TfidfClassifier(min_confidence=0.99).fit([r["text"] for r in train], [r["component"] for r in train],
                                                     [r["failure"] for r in train])
    assert strict.classify("spider insert wobbling")["component"] is None  # unsure means review


def test_prompt_lists_the_whole_taxonomy():
    text = prompt('belt "snapped" on CV-5')
    assert all(c in text for c in COMPONENTS) and all(f in text for f in FAILURES)
    assert "belt 'snapped' on CV-5" in text


def test_parse_takes_the_first_json_object():
    assert parse('Sure! {"component": "seal", "failure": "leak", "evidence": "x"} done') == {
        "component": "seal", "failure": "leak", "evidence": "x"}
    assert parse("no json here") is None


@pytest.mark.parametrize("reply, problem", [
    ({"component": "valve", "failure": "leak", "evidence": "weeping"}, "unknown component"),
    ({"component": "sensor", "failure": "leak", "evidence": "weeping"}, "cannot have"),
    ({"component": "seal", "failure": "leak", "evidence": "dripping badly"}, "evidence is not quoted"),
    (None, "not JSON"),
])
def test_validation_sends_bad_answers_to_review(reply, problem):
    labels, problems = validate("O-ring on P-101 weeping, replaced", reply)
    assert labels == {"component": None, "failure": None}
    assert any(problem in p for p in problems)


def test_a_valid_answer_with_quoted_evidence_is_used():
    labels, problems = validate("O-ring on P-101 weeping, replaced",
                                {"component": "Seal", "failure": "leak", "evidence": "O-ring on P-101  weeping"})
    assert labels == {"component": "seal", "failure": "leak"} and problems == []


def test_llm_classifier_with_a_scripted_model():
    clf = LLMClassifier(lambda p: '{"component": "coupling", "failure": "loose", "evidence": "spider insert wobbling"}')
    assert clf.classify("spider insert wobbling on M-4")["component"] == "coupling"


def test_cascade_uses_rules_first_and_the_model_only_for_the_rest():
    from wofc.llm import cascade

    calls = []
    model = LLMClassifier(lambda p: calls.append(p) or
                          '{"component": "seal", "failure": "leak", "evidence": "o-ring on p-101 weeping"}')
    classify = cascade(keywords.classify, model.classify)
    assert classify("hyd hose on P-101 leaking")["by"] == "rules" and calls == []
    assert classify("O-ring on P-101 weeping")["component"] == "seal" and len(calls) == 1
    classify("O-ring on P-101 weeping")
    assert len(calls) == 1  # cached
