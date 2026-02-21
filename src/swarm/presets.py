from __future__ import annotations

from .agents import Agent, SubtaskType
from .model_manager import OllamaModel


def preset_fast_tiny() -> list[Agent]:
    """
    Tiny/fast preset intended for 8GB VRAM POC.

    Note: model tags must exist in your local Ollama. Use `swarm models --ensure ...`
    or adjust tags to match your installed quantizations.
    """
    return [
        Agent(
            name="tinyllama:latest",
            cost_per_token=0.25,
            max_ctx=4096,
            capabilities={
                SubtaskType.CODE: 0.35,
                SubtaskType.REASONING: 0.45,
                SubtaskType.MATH: 0.35,
                SubtaskType.CREATIVE: 0.5,
                SubtaskType.EXTRACTION: 0.55,
                SubtaskType.SUMMARIZE: 0.55,
            },
            vram_gb=1.2,
            load_time_ms=450,
        ),
        Agent(
            name="gemma2:2b",
            cost_per_token=0.35,
            max_ctx=8192,
            capabilities={
                SubtaskType.CODE: 0.45,
                SubtaskType.REASONING: 0.5,
                SubtaskType.MATH: 0.45,
                SubtaskType.CREATIVE: 0.55,
                SubtaskType.EXTRACTION: 0.6,
                SubtaskType.SUMMARIZE: 0.6,
            },
            vram_gb=2.0,
            load_time_ms=600,
        ),
        Agent(
            name="phi3:mini",
            cost_per_token=0.6,
            max_ctx=8192,
            capabilities={
                SubtaskType.CODE: 0.5,
                SubtaskType.REASONING: 0.65,
                SubtaskType.MATH: 0.65,
                SubtaskType.CREATIVE: 0.5,
                SubtaskType.EXTRACTION: 0.6,
                SubtaskType.SUMMARIZE: 0.6,
            },
            vram_gb=3.0,
            load_time_ms=800,
        ),
    ]


def preset_heavy() -> list[Agent]:
    """
    Heavier preset intended for stronger compute.
    """
    return [
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
    ]


def preset_auto_fast(installed: list[OllamaModel], *, k: int = 3) -> list[Agent]:
    """
    Auto-build a 'fast' agent set from whatever is already installed in Ollama.

    Chooses k smallest models by reported size. Priors are generic.
    """
    models = [m for m in installed if m.size_bytes is not None]
    models.sort(key=lambda m: int(m.size_bytes or 0))
    chosen = models[: max(1, k)]

    # Rough VRAM/load-time estimates from model size (very approximate).
    agents: list[Agent] = []
    for m in chosen:
        size_gb = float(m.size_bytes or 0) / (1024.0**3)
        agents.append(
            Agent(
                name=m.name,
                cost_per_token=max(0.3, min(2.5, 0.4 + 0.15 * size_gb)),
                max_ctx=8192,
                capabilities={
                    SubtaskType.CODE: 0.55,
                    SubtaskType.REASONING: 0.6,
                    SubtaskType.MATH: 0.6,
                    SubtaskType.CREATIVE: 0.55,
                    SubtaskType.EXTRACTION: 0.6,
                    SubtaskType.SUMMARIZE: 0.6,
                },
                vram_gb=max(0.8, min(7.5, size_gb * 1.2)),
                load_time_ms=max(400.0, min(2500.0, size_gb * 250.0 + 400.0)),
            )
        )
    return agents

