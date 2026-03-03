from pathlib import Path

from src.swarm.router_training.preference_data import PreferenceExample, read_preferences_jsonl, write_preferences_jsonl


def test_preference_jsonl_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "prefs.jsonl"
    rows = [
        PreferenceExample(prompt="q", answer_a="a", answer_b="b", label="a", meta={"bench": "x"}),
        PreferenceExample(prompt="q2", answer_a="a2", answer_b="b2", label="tie", meta={}),
    ]
    write_preferences_jsonl(p, rows)
    out = read_preferences_jsonl(p)
    assert out == rows

