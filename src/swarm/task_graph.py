"""Task DAG support for DAG-aware orchestration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterator

from .agents import SubtaskType
from .tasks import Subtask


@dataclass(frozen=True)
class TaskNode:
    id: str
    task_type: SubtaskType
    description: str
    estimated_tokens: int
    difficulty: float
    depends_on: list[str]  # parent node IDs


@dataclass
class TaskDAG:
    job_id: str
    nodes: dict[str, TaskNode]  # id -> node


def node_to_subtask(node: TaskNode) -> Subtask:
    return Subtask(
        id=node.id,
        task_type=node.task_type,
        description=node.description,
        estimated_tokens=node.estimated_tokens,
        difficulty=node.difficulty,
    )


def topological_sort(nodes: dict[str, TaskNode]) -> list[str]:
    """Return node IDs in topological order via Kahn's algorithm."""
    in_degree: dict[str, int] = {nid: 0 for nid in nodes}
    for n in nodes.values():
        for child_id in _children(nodes, n.id):
            in_degree[child_id] = in_degree.get(child_id, 0) + 1
    # Actually: depends_on = parents. So for edge (p -> c), c's in_degree increases.
    in_degree = {nid: 0 for nid in nodes}
    for n in nodes.values():
        for pid in n.depends_on:
            if pid in nodes:
                in_degree[n.id] = in_degree.get(n.id, 0) + 1

    q: deque[str] = deque(nid for nid, d in in_degree.items() if d == 0)
    order: list[str] = []
    while q:
        nid = q.popleft()
        order.append(nid)
        for cid in _children(nodes, nid):
            in_degree[cid] -= 1
            if in_degree[cid] == 0:
                q.append(cid)
    return order


def _children(nodes: dict[str, TaskNode], parent_id: str) -> Iterator[str]:
    for n in nodes.values():
        if parent_id in n.depends_on:
            yield n.id


def get_layers(nodes: dict[str, TaskNode], topo_order: list[str]) -> list[list[str]]:
    """Group by layer: layer 0 = roots; layer k = nodes whose deps are all in layers 0..k-1."""
    layer_of: dict[str, int] = {}
    for nid in topo_order:
        n = nodes.get(nid)
        if n is None:
            continue
        if not n.depends_on:
            layer_of[nid] = 0
        else:
            layer_of[nid] = max(layer_of.get(p, 0) + 1 for p in n.depends_on if p in nodes)
    max_layer = max(layer_of.values()) if layer_of else -1
    layers: list[list[str]] = [[] for _ in range(max_layer + 1)]
    for nid, L in layer_of.items():
        layers[L].append(nid)
    return layers


def get_horizon_nodes(
    completed: set[str],
    dag: TaskDAG,
    horizon_depth: int,
) -> list[str]:
    """Return node IDs for horizon: all currently ready nodes plus successors up to horizon_depth."""
    topo = topological_sort(dag.nodes)
    layer_map: dict[str, int] = {}
    for nid in topo:
        n = dag.nodes.get(nid)
        if n is None:
            continue
        if not n.depends_on:
            layer_map[nid] = 0
        else:
            layer_map[nid] = max(layer_map.get(p, 0) + 1 for p in n.depends_on if p in dag.nodes)

    ready = [nid for nid in topo if set(dag.nodes[nid].depends_on).issubset(completed)]
    horizon: set[str] = set(ready)
    for nid in ready:
        frontier = [nid]
        for _ in range(horizon_depth):
            next_frontier = []
            for pid in frontier:
                for cid in _children(dag.nodes, pid):
                    if cid not in horizon and cid not in completed:
                        horizon.add(cid)
                        next_frontier.append(cid)
            frontier = next_frontier
            if not frontier:
                break
    return [nid for nid in topo if nid in horizon]
