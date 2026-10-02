from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HostGrid:
    side: int
    n_vertices: int
    policy: str = "ceil-sqrt-square"

    def __post_init__(self) -> None:
        if self.side <= 0 or self.n_vertices <= 0 or self.side * self.side < self.n_vertices:
            raise ValueError("invalid square host")

    @classmethod
    def square(cls, n_vertices: int) -> "HostGrid":
        if not isinstance(n_vertices, int) or isinstance(n_vertices, bool) or n_vertices <= 0:
            raise ValueError("n_vertices must be a positive integer")
        side = math.isqrt(n_vertices)
        if side * side < n_vertices:
            side += 1
        return cls(side, n_vertices)

    @property
    def rows(self) -> int:
        return self.side

    @property
    def cols(self) -> int:
        return self.side

    @property
    def capacity(self) -> int:
        return self.side * self.side

    @property
    def diameter(self) -> int:
        return self.side - 1


def host_cells(host: HostGrid) -> tuple[tuple[int, int], ...]:
    return tuple((row, col) for row in range(1, host.side + 1) for col in range(1, host.side + 1))


def deterministic_placement(host: HostGrid, *, anchor: int) -> dict[int, tuple[int, int]]:
    if not 1 <= anchor <= host.n_vertices:
        raise ValueError("anchor is outside graph range")
    vertices = [anchor, *(vertex for vertex in range(1, host.n_vertices + 1) if vertex != anchor)]
    return dict(zip(vertices, host_cells(host)[: host.n_vertices], strict=True))
