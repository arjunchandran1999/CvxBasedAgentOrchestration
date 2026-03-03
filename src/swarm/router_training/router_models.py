from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ThresholdRouter:
    """
    A minimal RouteLLM-style router:
    - compute p_strong = P(strong wins) from a scorer
    - route to strong if p_strong >= tau
    """

    strong_model: str
    weak_model: str
    tau: float
    scorer: Callable[[str], float]  # prompt -> probability strong wins

    def route(self, prompt: str) -> str:
        p = float(self.scorer(prompt))
        return self.strong_model if p >= float(self.tau) else self.weak_model

