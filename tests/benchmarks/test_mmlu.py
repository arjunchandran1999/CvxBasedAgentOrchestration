from src.swarm.benchmarks.mmlu import _normalize_choice


def test_normalize_choice_simple() -> None:
    assert _normalize_choice("A") == "A"
    assert _normalize_choice("b") == "B"


def test_normalize_choice_with_text() -> None:
    assert _normalize_choice("Final: C") == "C"
    assert _normalize_choice("Answer is (d).") == "D"


def test_normalize_choice_missing() -> None:
    assert _normalize_choice("") is None
    assert _normalize_choice("I think the answer is 42") is None

