from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol


@dataclass(frozen=True)
class BenchmarkExample:
    benchmark: str
    example_id: str
    query: str
    reference: dict


class Benchmark(Protocol):
    name: str

    def load(self, *, data_dir: str, limit: int, seed: int) -> list[BenchmarkExample]: ...

    def score(self, *, example: BenchmarkExample, final_answer: str | None) -> dict: ...

    # Optional hooks for multi-subtask or custom scoring.
    def make_subtasks(self, *, example: BenchmarkExample) -> list | None: ...  # list[Subtask] at runtime

    def score_artifact(self, *, example: BenchmarkExample, artifact: dict) -> dict: ...


_REGISTRY: dict[str, Benchmark] = {}


def register(bench: Benchmark) -> None:
    _REGISTRY[bench.name] = bench


def get(name: str) -> Benchmark:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown benchmark: {name}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_benchmarks() -> list[str]:
    return sorted(_REGISTRY.keys())


def ensure_default_benchmarks_loaded() -> None:
    # Import side effects to populate registry.
    from . import workflowbench  # noqa: F401
    from . import code_math_mix  # noqa: F401
    from . import agentic_bench  # noqa: F401

