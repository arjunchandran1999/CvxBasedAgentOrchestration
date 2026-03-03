import importlib

import pytest


def test_train_tfidf_logreg_trains_if_sklearn_available() -> None:
    sklearn = importlib.util.find_spec("sklearn")
    if sklearn is None:
        pytest.skip("scikit-learn not installed (install extras: .[router])")

    from src.swarm.router_training.train_router import train_tfidf_logreg

    prompts = ["easy question", "hard reasoning", "easy yes no", "hard math proof"]
    labels = [0, 1, 0, 1]  # strong wins on hard
    model = train_tfidf_logreg(prompts=prompts, labels_strong_wins=labels, strong_model="S", weak_model="W")
    p = model.predict_proba_strong("hard new problem")
    assert 0.0 <= p <= 1.0


def test_train_tfidf_logreg_pickle_roundtrip_if_sklearn_available(tmp_path: "Path") -> None:
    import importlib
    from pathlib import Path

    sklearn = importlib.util.find_spec("sklearn")
    if sklearn is None:
        pytest.skip("scikit-learn not installed (install extras: .[router])")

    from src.swarm.router_training.train_router import TrainedRouterModel, train_tfidf_logreg

    prompts = ["easy", "hard"]
    labels = [0, 1]
    m = train_tfidf_logreg(prompts=prompts, labels_strong_wins=labels, strong_model="S", weak_model="W")
    p = tmp_path / "m.pkl"
    m.save(p)
    m2 = TrainedRouterModel.load(p)
    assert 0.0 <= m2.predict_proba_strong("hard") <= 1.0

