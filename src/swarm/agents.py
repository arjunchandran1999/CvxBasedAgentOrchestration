from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class SubtaskType(str, Enum):
    CODE = "code"
    REASONING = "reasoning"
    MATH = "math"
    CREATIVE = "creative"
    EXTRACTION = "extraction"
    SUMMARIZE = "summarize"


@dataclass(frozen=True)
class Agent:
    name: str  # Ollama model tag
    cost_per_token: float  # relative cost scalar
    max_ctx: int
    capabilities: Mapping[SubtaskType, float]  # prior quality per type, 0..1
    vram_gb: float
    load_time_ms: float


def default_agents() -> list[Agent]:
    # MVP constants: rough priors (tune later).
    return [
        Agent(
            name="qwen2.5-coder:7b",
            cost_per_token=1.2,
            max_ctx=32768,
            capabilities={
                SubtaskType.CODE: 0.85,
                SubtaskType.REASONING: 0.65,
                SubtaskType.MATH: 0.65,
                SubtaskType.CREATIVE: 0.55,
                SubtaskType.EXTRACTION: 0.6,
                SubtaskType.SUMMARIZE: 0.6,
            },
            vram_gb=6.5,
            load_time_ms=1200,
        ),
        Agent(
            name="llama3.1:8b",
            cost_per_token=1.4,
            max_ctx=8192,
            capabilities={
                SubtaskType.CODE: 0.7,
                SubtaskType.REASONING: 0.8,
                SubtaskType.MATH: 0.72,
                SubtaskType.CREATIVE: 0.7,
                SubtaskType.EXTRACTION: 0.7,
                SubtaskType.SUMMARIZE: 0.75,
            },
            vram_gb=8.0,
            load_time_ms=1600,
        ),
        Agent(
            name="phi4:14b",
            cost_per_token=2.0,
            max_ctx=8192,
            capabilities={
                SubtaskType.CODE: 0.75,
                SubtaskType.REASONING: 0.88,
                SubtaskType.MATH: 0.9,
                SubtaskType.CREATIVE: 0.55,
                SubtaskType.EXTRACTION: 0.72,
                SubtaskType.SUMMARIZE: 0.7,
            },
            vram_gb=12.5,
            load_time_ms=2400,
        ),
        Agent(
            name="gemma2:2b",
            cost_per_token=0.4,
            max_ctx=8192,
            capabilities={
                SubtaskType.CODE: 0.45,
                SubtaskType.REASONING: 0.5,
                SubtaskType.MATH: 0.45,
                SubtaskType.CREATIVE: 0.55,
                SubtaskType.EXTRACTION: 0.55,
                SubtaskType.SUMMARIZE: 0.55,
            },
            vram_gb=2.0,
            load_time_ms=600,
        ),
    ]

