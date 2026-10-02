from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from python.encoders.symmetry_breaking import verify_symmetry
from python.utils.host_grid import HostGrid
from python.utils.mtx_parser import Edge, normalize_edges


@dataclass(frozen=True, slots=True)
class VerificationResult:
    valid: bool
    bandwidth: int | None
    error: str | None = None


def compute_bandwidth(positions: Mapping[int, tuple[int, int]], edges: Sequence[Edge]) -> int:
    return max(
        (
            max(abs(positions[u][0] - positions[v][0]), abs(positions[u][1] - positions[v][1]))
            for u, v in edges
        ),
        default=0,
    )


def verify_placement(
    n_vertices: int,
    edges: Sequence[Edge],
    host: HostGrid,
    positions: Mapping[int, tuple[int, int]],
    *,
    tested_k: int | None = None,
    anchor: int | None = None,
    symmetry: str | None = None,
) -> VerificationResult:
    expected = set(range(1, n_vertices + 1))
    if set(positions) != expected:
        return VerificationResult(False, None, "vertex set is incomplete or contains extras")
    occupied: set[tuple[int, int]] = set()
    for vertex in expected:
        coordinate = positions[vertex]
        if not isinstance(coordinate, tuple) or len(coordinate) != 2:
            return VerificationResult(False, None, f"invalid coordinate for vertex {vertex}")
        row, col = coordinate
        if not isinstance(row, int) or isinstance(row, bool) or not isinstance(col, int) or isinstance(col, bool):
            return VerificationResult(False, None, f"non-integer coordinate for vertex {vertex}")
        if not 1 <= row <= host.side or not 1 <= col <= host.side:
            return VerificationResult(False, None, f"vertex {vertex} is outside host")
        if coordinate in occupied:
            return VerificationResult(False, None, f"duplicate occupied cell: {coordinate}")
        occupied.add(coordinate)
    if symmetry is not None:
        if anchor is None or not verify_symmetry(positions, side=host.side, anchor=anchor, mode=symmetry):
            return VerificationResult(False, None, "symmetry restriction is violated")
    try:
        bandwidth = compute_bandwidth(positions, normalize_edges(n_vertices, edges))
    except (KeyError, ValueError) as exc:
        return VerificationResult(False, None, str(exc))
    if tested_k is not None and bandwidth > tested_k:
        return VerificationResult(False, bandwidth, f"bandwidth {bandwidth} exceeds K={tested_k}")
    return VerificationResult(True, bandwidth)
