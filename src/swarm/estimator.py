from __future__ import annotations

from dataclasses import dataclass, field

from .agents import Agent, SubtaskType
from .tasks import Subtask


@dataclass
class QualityEstimator:
    # (agent_name, subtask_type) -> quality estimate in 0..1
    quality: dict[tuple[str, SubtaskType], float] = field(default_factory=dict)
    ema_alpha: float = 0.2

    def ensure_priors(self, agents: list[Agent]) -> None:
        for a in agents:
            for t, q in a.capabilities.items():
                self.quality.setdefault((a.name, t), float(q))

    def predict(self, agent: Agent, subtask: Subtask) -> float:
        base = float(self.quality.get((agent.name, subtask.task_type), 0.5))
        # Difficulty penalty: treat difficulty ~ 1..5, with linear decay.
        penalty = 0.06 * max(0.0, float(subtask.difficulty) - 1.0)
        return float(max(0.0, min(1.0, base - penalty)))

    def update(self, *, agent: Agent, subtask_type: SubtaskType, observed_score: float) -> None:
        key = (agent.name, subtask_type)
        prev = float(self.quality.get(key, 0.5))
        obs = float(max(0.0, min(1.0, observed_score)))
        self.quality[key] = (1.0 - self.ema_alpha) * prev + self.ema_alpha * obs


def estimate_token_cost(agent: Agent, subtask: Subtask) -> float:
    return float(agent.cost_per_token) * float(subtask.estimated_tokens)


def normalized_switch_cost(agent: Agent, *, t_scale_ms: float = 1500.0) -> float:
    # Normalize into the same rough magnitude range as token-cost terms.
    return float(agent.load_time_ms) / float(t_scale_ms)

