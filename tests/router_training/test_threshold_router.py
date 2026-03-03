from src.swarm.router_training.router_models import ThresholdRouter


def test_threshold_router_routes_strong() -> None:
    r = ThresholdRouter(strong_model="S", weak_model="W", tau=0.7, scorer=lambda _: 0.9)
    assert r.route("x") == "S"


def test_threshold_router_routes_weak() -> None:
    r = ThresholdRouter(strong_model="S", weak_model="W", tau=0.7, scorer=lambda _: 0.2)
    assert r.route("x") == "W"

