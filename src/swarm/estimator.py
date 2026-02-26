from __future__ import annotations

from dataclasses import dataclass, field

from .agents import Agent, SubtaskType
from .tasks import Subtask


@dataclass
class QualityEstimator:
    # (agent_name, subtask_type) -> quality estimate in 0..1
    quality: dict[tuple[str, SubtaskType], float] = field(default_factory=dict)
    ema_alpha: float = 0.2
    # agent_name -> multiplier to map estimated_tokens -> expected actual tokens
    token_multiplier: dict[str, float] = field(default_factory=dict)
    token_ema_alpha: float = 0.2

    def ensure_priors(self, agents: list[Agent]) -> None:
        for a in agents:
            for t, q in a.capabilities.items():
                self.quality.setdefault((a.name, t), float(q))
            self.token_multiplier.setdefault(a.name, 1.0)

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

    def update_token_multiplier(self, *, agent: Agent, estimated_tokens: int, actual_tokens: int) -> None:
        """
        Update per-agent multiplier using observed token usage.

        This is a lightweight "cost estimator training": it calibrates subtask.estimated_tokens
        to the observed tokens the model actually used.
        """
        est = float(max(1, int(estimated_tokens)))
        act = float(max(0, int(actual_tokens)))
        ratio = act / est
        # Keep ratios in a sane range to avoid single-task blowups.
        ratio = float(min(10.0, max(0.1, ratio)))
        prev = float(self.token_multiplier.get(agent.name, 1.0))
        self.token_multiplier[agent.name] = (1.0 - self.token_ema_alpha) * prev + self.token_ema_alpha * ratio

    def estimate_token_cost(self, agent: Agent, subtask: Subtask) -> float:
        mult = float(self.token_multiplier.get(agent.name, 1.0))
        return float(agent.cost_per_token) * float(subtask.estimated_tokens) * mult

    def to_dict(self) -> dict:
        return {
            "ema_alpha": float(self.ema_alpha),
            "token_ema_alpha": float(self.token_ema_alpha),
            "quality": [
                {"agent": a, "subtask_type": t.value, "quality": float(q)}
                for (a, t), q in sorted(self.quality.items(), key=lambda x: (x[0][0], x[0][1].value))
            ],
            "token_multiplier": {str(k): float(v) for k, v in sorted(self.token_multiplier.items())},
        }

    @staticmethod
    def from_dict(d: dict) -> "QualityEstimator":
        est = QualityEstimator()
        if isinstance(d.get("ema_alpha"), (int, float)):
            est.ema_alpha = float(d["ema_alpha"])
        if isinstance(d.get("token_ema_alpha"), (int, float)):
            est.token_ema_alpha = float(d["token_ema_alpha"])

        q_items = d.get("quality") or []
        if isinstance(q_items, list):
            for it in q_items:
                if not isinstance(it, dict):
                    continue
                agent = it.get("agent")
                st = it.get("subtask_type")
                qv = it.get("quality")
                if not isinstance(agent, str) or not isinstance(st, str) or not isinstance(qv, (int, float)):
                    continue
                try:
                    t = SubtaskType(st)
                except Exception:
                    continue
                est.quality[(agent, t)] = float(qv)

        tm = d.get("token_multiplier") or {}
        if isinstance(tm, dict):
            for k, v in tm.items():
                if isinstance(k, str) and isinstance(v, (int, float)):
                    est.token_multiplier[k] = float(v)
        return est

    def save_json(self, path: str) -> None:
        import json
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load_json(path: str) -> "QualityEstimator":
        import json
        from pathlib import Path

        p = Path(path)
        d = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            raise ValueError("Estimator state JSON must be an object")
        return QualityEstimator.from_dict(d)


def estimate_token_cost(agent: Agent, subtask: Subtask) -> float:
    return float(agent.cost_per_token) * float(subtask.estimated_tokens)


def normalized_switch_cost(agent: Agent, *, t_scale_ms: float = 1500.0) -> float:
    # Normalize into the same rough magnitude range as token-cost terms. 
    return float(agent.load_time_ms) / float(t_scale_ms)

