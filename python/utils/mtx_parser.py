from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

Edge = tuple[int, int]
SUPPORTED_FIELDS = frozenset({"pattern", "integer", "real", "complex"})
SUPPORTED_SYMMETRIES = frozenset({"general", "symmetric", "skew-symmetric", "hermitian"})


def normalize_edges(n_vertices: int, edges: Iterable[Sequence[int]]) -> tuple[Edge, ...]:
    if not isinstance(n_vertices, int) or isinstance(n_vertices, bool) or n_vertices <= 0:
        raise ValueError("n_vertices must be a positive integer")
    normalized: set[Edge] = set()
    for edge in edges:
        if len(edge) != 2:
            raise ValueError(f"edge must have two endpoints: {edge!r}")
        u, v = edge
        if not isinstance(u, int) or isinstance(u, bool) or not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"edge endpoints must be integers: {edge!r}")
        if not 1 <= u <= n_vertices or not 1 <= v <= n_vertices:
            raise ValueError(f"edge {edge!r} is outside 1..{n_vertices}")
        if u != v:
            normalized.add((min(u, v), max(u, v)))
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class ParsedMtxGraph:
    n_vertices: int
    edges: tuple[Edge, ...]
    rows: int
    cols: int
    declared_entries: int
    field: str
    symmetry: str
    interpretation: str = "square-matrix-market-v1"

    def __post_init__(self) -> None:
        if self.n_vertices <= 0 or self.rows != self.n_vertices or self.cols != self.n_vertices:
            raise ValueError("graph metadata must describe a positive square matrix")
        object.__setattr__(self, "edges", normalize_edges(self.n_vertices, self.edges))

    @classmethod
    def from_edges(cls, n_vertices: int, edges: Iterable[Sequence[int]]) -> "ParsedMtxGraph":
        normalized = normalize_edges(n_vertices, edges)
        return cls(
            n_vertices=n_vertices,
            edges=normalized,
            rows=n_vertices,
            cols=n_vertices,
            declared_entries=len(normalized),
            field="pattern",
            symmetry="general",
            interpretation="in-memory-edge-list-v1",
        )

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def adjacency(self) -> dict[int, tuple[int, ...]]:
        result: dict[int, list[int]] = {vertex: [] for vertex in range(1, self.n_vertices + 1)}
        for u, v in self.edges:
            result[u].append(v)
            result[v].append(u)
        return {vertex: tuple(neighbors) for vertex, neighbors in result.items()}


def _parse_banner(line: str) -> tuple[str, str]:
    parts = line.split()
    if len(parts) != 5 or parts[0].lower() != "%%matrixmarket":
        raise ValueError("expected a Matrix Market coordinate banner")
    if parts[1].lower() != "matrix" or parts[2].lower() != "coordinate":
        raise ValueError("only Matrix Market coordinate matrices are supported")
    field, symmetry = parts[3].lower(), parts[4].lower()
    if field not in SUPPORTED_FIELDS:
        raise ValueError(f"unsupported Matrix Market field: {field}")
    if symmetry not in SUPPORTED_SYMMETRIES:
        raise ValueError(f"unsupported Matrix Market symmetry: {symmetry}")
    return field, symmetry


def _parse_nonzero(parts: list[str], field: str, line_number: int) -> tuple[int, int, bool]:
    expected = {"pattern": 2, "integer": 3, "real": 3, "complex": 4}[field]
    if len(parts) != expected:
        raise ValueError(f"invalid {field} entry at line {line_number}")
    try:
        row, col = int(parts[0]), int(parts[1])
        if field == "pattern":
            return row, col, True
        if field == "integer":
            return row, col, int(parts[2]) != 0
        if field == "real":
            value = float(parts[2])
            if not math.isfinite(value):
                raise ValueError
            return row, col, value != 0.0
        real, imaginary = float(parts[2]), float(parts[3])
        if not math.isfinite(real) or not math.isfinite(imaginary):
            raise ValueError
        return row, col, real != 0.0 or imaginary != 0.0
    except ValueError as exc:
        raise ValueError(f"invalid {field} value at line {line_number}") from exc


def parse_mtx_graph(path: str | Path) -> ParsedMtxGraph:
    """Parse only strict square Matrix Market coordinate graph inputs.

    Rectangular matrices and legacy edge lists are intentionally unsupported.
    Nonzero off-diagonal entries define a simple undirected edge set.
    """

    source = Path(path)
    field: str | None = None
    symmetry: str | None = None
    banner_seen = header_seen = False
    rows = cols = declared_entries = parsed_entries = 0
    edges: list[Edge] = []
    with source.open("r", encoding="utf-8", errors="strict") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.lower().startswith("%%matrixmarket"):
                if banner_seen:
                    raise ValueError(f"duplicate Matrix Market banner at line {line_number}")
                field, symmetry = _parse_banner(line)
                banner_seen = True
                continue
            if line.startswith("%"):
                continue
            if not banner_seen:
                raise ValueError("Matrix Market banner must precede the size line")
            if not header_seen:
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError("Matrix Market size line must contain rows, cols, and entries")
                try:
                    rows, cols, declared_entries = (int(part) for part in parts)
                except ValueError as exc:
                    raise ValueError("Matrix Market size line must contain integers") from exc
                if rows <= 0 or cols <= 0 or declared_entries < 0:
                    raise ValueError("Matrix Market dimensions and entry count are invalid")
                if rows != cols:
                    raise ValueError(f"only square Matrix Market inputs are supported, got {rows} x {cols}")
                header_seen = True
                continue
            assert field is not None
            parsed_entries += 1
            row, col, nonzero = _parse_nonzero(line.split(), field, line_number)
            if not 1 <= row <= rows or not 1 <= col <= cols:
                raise ValueError(f"Matrix Market entry ({row}, {col}) is outside 1..{rows} x 1..{cols}")
            if nonzero and row != col:
                edges.append((row, col))
    if not banner_seen:
        raise ValueError("file has no Matrix Market banner")
    if not header_seen:
        raise ValueError("Matrix Market file has no size line")
    if parsed_entries != declared_entries:
        raise ValueError(f"Matrix Market entry count mismatch: declared {declared_entries}, parsed {parsed_entries}")
    assert field is not None and symmetry is not None
    return ParsedMtxGraph(rows, tuple(edges), rows, cols, declared_entries, field, symmetry)


def parse_mtx_file(path: str | Path) -> tuple[int, tuple[Edge, ...]]:
    graph = parse_mtx_graph(path)
    return graph.n_vertices, graph.edges


def write_mtx_graph(path: str | Path, graph: ParsedMtxGraph, *, symmetry: str = "symmetric") -> None:
    if symmetry not in {"general", "symmetric"}:
        raise ValueError("fixture writer supports general or symmetric output")
    entries = list(graph.edges)
    if symmetry == "general":
        entries = [entry for edge in entries for entry in (edge, (edge[1], edge[0]))]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            [
                f"%%MatrixMarket matrix coordinate pattern {symmetry}",
                "% generated classical fixture",
                f"{graph.n_vertices} {graph.n_vertices} {len(entries)}",
                *(f"{u} {v}" for u, v in entries),
                "",
            ]
        ),
        encoding="utf-8",
    )
