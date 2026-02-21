from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .agents import SubtaskType
from .ollama_client import OllamaClient, OllamaChatResult
from .routing import Assignment
from .tasks import Subtask


@dataclass(frozen=True)
class SubtaskResult:
    subtask_id: str
    agent: str
    output: str
    latency_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    success: bool
    error: str | None = None
    swapped_in_for_job: bool = False


def _system_for(task_type: SubtaskType) -> str:
    if task_type == SubtaskType.CODE:
        return "You are a senior software engineer. Produce correct, runnable code and explain briefly."
    if task_type == SubtaskType.MATH:
        return "You are a meticulous mathematician. Show steps and final answer."
    if task_type == SubtaskType.EXTRACTION:
        return "You extract structured information precisely. Be concise."
    if task_type == SubtaskType.SUMMARIZE:
        return "You summarize faithfully and concisely."
    if task_type == SubtaskType.CREATIVE:
        return "You are a creative writer. Follow constraints and be engaging."
    return "You are a careful reasoner. Be correct and concise."


async def execute(
    *,
    query: str,
    subtasks: list[Subtask],
    assignments: list[Assignment],
    ollama: OllamaClient,
    loaded_models_before: set[str],
    max_concurrency: int = 4,
) -> list[SubtaskResult]:
    by_id = {s.id: s for s in subtasks}
    sem = asyncio.Semaphore(max_concurrency)

    async def _run(a: Assignment) -> SubtaskResult:
        s = by_id.get(a.subtask_id)
        if s is None:
            return SubtaskResult(
                subtask_id=a.subtask_id,
                agent=a.agent,
                output="",
                latency_ms=None,
                prompt_tokens=None,
                completion_tokens=None,
                success=False,
                error="Unknown subtask_id",
                swapped_in_for_job=a.agent not in loaded_models_before,
            )

        system = _system_for(s.task_type)
        user = (
            f"Overall job query:\n{query}\n\n"
            f"Subtask ({s.task_type.value}, difficulty={s.difficulty}):\n{s.description}\n"
        )

        async with sem:
            try:
                r: OllamaChatResult = await ollama.chat_with_usage(
                    model=a.agent,
                    system=system,
                    user=user,
                    temperature=0.2,
                )
                return SubtaskResult(
                    subtask_id=s.id,
                    agent=a.agent,
                    output=r.text,
                    latency_ms=r.latency_ms,
                    prompt_tokens=r.prompt_tokens,
                    completion_tokens=r.completion_tokens,
                    success=True,
                    swapped_in_for_job=a.agent not in loaded_models_before,
                )
            except Exception as e:
                return SubtaskResult(
                    subtask_id=s.id,
                    agent=a.agent,
                    output="",
                    latency_ms=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                    success=False,
                    error=str(e),
                    swapped_in_for_job=a.agent not in loaded_models_before,
                )

    results = await asyncio.gather(*[_run(a) for a in assignments])
    # Stable order by input assignments
    return list(results)

