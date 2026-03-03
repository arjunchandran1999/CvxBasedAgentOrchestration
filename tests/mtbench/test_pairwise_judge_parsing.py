from src.swarm.mtbench_judge import _coerce_json


def test_coerce_json_plain() -> None:
    assert _coerce_json('{"winner":"a","rationale":"x"}') == {"winner": "a", "rationale": "x"}


def test_coerce_json_fenced() -> None:
    assert _coerce_json('```json\n{"winner":"tie"}\n```') == {"winner": "tie"}

