from __future__ import annotations

from typing import Any

from python.encoders.position_constraints import PositionVariables
from python.utils.mtx_parser import ParsedMtxGraph


def add_distance_layer(
    solver: Any, graph: ParsedMtxGraph, variables: PositionVariables, k: int, *, guard: int | None = None
) -> int:
    """Add C6 for one decreasing Chebyshev threshold K."""

    if not isinstance(k, int) or isinstance(k, bool) or not 0 <= k <= variables.side - 1:
        raise ValueError(f"K must be an integer in 0..{variables.side - 1}")
    if guard is not None and (not isinstance(guard, int) or isinstance(guard, bool) or guard <= 0):
        raise ValueError("guard must be a positive variable identifier")
    count = 0
    max_i = variables.side - k - 1
    for u, v in graph.edges:
        for i in range(1, max_i + 1):
            for clause in (
                [-variables.lx[u][i], variables.lx[v][i + k]],
                [-variables.lx[v][i], variables.lx[u][i + k]],
                [-variables.ly[u][i], variables.ly[v][i + k]],
                [-variables.ly[v][i], variables.ly[u][i + k]],
            ):
                solver.add_clause(clause if guard is None else [-guard, *clause])
                count += 1
    return count


def add_guarded_distance_layer(
    solver: Any, graph: ParsedMtxGraph, variables: PositionVariables, k: int, selector: int
) -> int:
    """Add C6 for ``k`` guarded by a positive assumption selector."""
    return add_distance_layer(solver, graph, variables, k, guard=selector)
