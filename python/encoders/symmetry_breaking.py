from __future__ import annotations

from typing import Mapping

from python.utils.mtx_parser import ParsedMtxGraph

SYMMETRY_MODES = frozenset({"none", "q1", "triangle"})


def select_anchor_vertex(graph: ParsedMtxGraph) -> int:
    adjacency = graph.adjacency()
    degrees = {vertex: len(neighbors) for vertex, neighbors in adjacency.items()}
    return min(
        adjacency,
        key=lambda vertex: (-degrees[vertex], -sum(degrees[neighbor] for neighbor in adjacency[vertex]), vertex),
    )


def calculate_symmetry_boundary(side: int) -> int:
    if not isinstance(side, int) or isinstance(side, bool) or side <= 0:
        raise ValueError("host side must be a positive integer")
    return (side + 1) // 2


def encode_symmetry_breaking_constraints(
    graph: ParsedMtxGraph,
    side: int,
    lx_vars: Mapping[int, Mapping[int, int]],
    ly_vars: Mapping[int, Mapping[int, int]],
    mode: str,
) -> tuple[int, list[list[int]]]:
    if mode not in SYMMETRY_MODES:
        raise ValueError(f"unsupported symmetry mode {mode!r}")
    anchor = select_anchor_vertex(graph)
    if mode == "none":
        return anchor, []
    if side == 1:
        return anchor, []
    q = calculate_symmetry_boundary(side)
    clauses = [[lx_vars[anchor][q]]]
    if mode == "q1":
        clauses.append([ly_vars[anchor][q]])
    else:
        clauses.extend([-lx_vars[anchor][i], ly_vars[anchor][i]] for i in range(1, q + 1))
    return anchor, clauses


def verify_symmetry(
    positions: Mapping[int, tuple[int, int]], *, side: int, anchor: int, mode: str
) -> bool:
    if mode not in SYMMETRY_MODES:
        return False
    if mode == "none":
        return True
    if anchor not in positions:
        return False
    x, y = positions[anchor]
    q = calculate_symmetry_boundary(side)
    return x <= q and (y <= q if mode == "q1" else y <= x)
