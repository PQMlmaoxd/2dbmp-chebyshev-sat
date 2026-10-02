from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pysat.card import CardEnc, EncType
from pysat.formula import IDPool


@dataclass(slots=True)
class PositionVariables:
    n_vertices: int
    side: int
    pool: IDPool
    x: dict[int, dict[int, int]]
    y: dict[int, dict[int, int]]
    lx: dict[int, dict[int, int]]
    ly: dict[int, dict[int, int]]
    a: dict[int, dict[int, dict[int, int]]] | None

    @classmethod
    def create(
        cls,
        n_vertices: int,
        side: int,
        pool: IDPool,
        *,
        include_cell_indicators: bool = True,
    ) -> "PositionVariables":
        if side <= 0 or side * side < n_vertices:
            raise ValueError("host cannot hold all vertices")
        x: dict[int, dict[int, int]] = {}
        y: dict[int, dict[int, int]] = {}
        lx: dict[int, dict[int, int]] = {}
        ly: dict[int, dict[int, int]] = {}
        a: dict[int, dict[int, dict[int, int]]] | None = {} if include_cell_indicators else None
        for vertex in range(1, n_vertices + 1):
            x[vertex] = {i: pool.id(("X", vertex, i)) for i in range(1, side + 1)}
            y[vertex] = {i: pool.id(("Y", vertex, i)) for i in range(1, side + 1)}
            lx[vertex] = {i: pool.id(("LX", vertex, i)) for i in range(1, side)}
            ly[vertex] = {i: pool.id(("LY", vertex, i)) for i in range(1, side)}
            if a is not None:
                a[vertex] = {
                    i: {j: pool.id(("A", vertex, i, j)) for j in range(1, side + 1)}
                    for i in range(1, side + 1)
                }
        return cls(n_vertices, side, pool, x, y, lx, ly, a)

    def extract_positions(self, model: list[int]) -> dict[int, tuple[int, int]]:
        positive = {literal for literal in model if literal > 0}
        result: dict[int, tuple[int, int]] = {}
        for vertex in range(1, self.n_vertices + 1):
            xs = [i for i, literal in self.x[vertex].items() if literal in positive]
            ys = [i for i, literal in self.y[vertex].items() if literal in positive]
            if len(xs) != 1 or len(ys) != 1:
                raise RuntimeError(f"model has invalid coordinate assignment for vertex {vertex}")
            result[vertex] = (xs[0], ys[0])
        return result


def _add(solver: Any, clauses: list[list[int]]) -> int:
    for clause in clauses:
        solver.add_clause(clause)
    return len(clauses)


def _add_axis_channel(solver: Any, exact: dict[int, int], prefix: dict[int, int], side: int) -> int:
    if side == 1:
        return _add(solver, [[exact[1]]])
    count = _add(solver, [[-exact[1], prefix[1]], [exact[1], -prefix[1]]])
    for i in range(2, side):
        count += _add(
            solver,
            [
                [-exact[i], -prefix[i - 1]],
                [-exact[i], prefix[i]],
                [exact[i], prefix[i - 1], -prefix[i]],
            ],
        )
    return count + _add(
        solver,
        [[-exact[side], -prefix[side - 1]], [exact[side], prefix[side - 1]]],
    )


def encode_position_constraints(
    solver: Any,
    variables: PositionVariables,
    *,
    include_cell_channeling: bool | None = None,
    include_injectivity: bool | None = None,
) -> dict[str, int]:
    """Add position constraints, optionally omitting the A/C4/C5 layer."""

    if include_cell_channeling is None:
        include_cell_channeling = variables.a is not None
    if include_injectivity is None:
        include_injectivity = include_cell_channeling
    if include_injectivity and variables.a is None:
        raise ValueError("injectivity encoding requires cell indicators")
    if include_cell_channeling and variables.a is None:
        raise ValueError("cell channeling requires cell indicators")

    order = channel = cell = injective = 0
    for vertex in range(1, variables.n_vertices + 1):
        for i in range(1, variables.side - 1):
            order += _add(solver, [[-variables.lx[vertex][i], variables.lx[vertex][i + 1]]])
            order += _add(solver, [[-variables.ly[vertex][i], variables.ly[vertex][i + 1]]])
        channel += _add_axis_channel(solver, variables.x[vertex], variables.lx[vertex], variables.side)
        channel += _add_axis_channel(solver, variables.y[vertex], variables.ly[vertex], variables.side)
        if include_cell_channeling:
            assert variables.a is not None
            for i in range(1, variables.side + 1):
                for j in range(1, variables.side + 1):
                    indicator = variables.a[vertex][i][j]
                    cell += _add(
                        solver,
                        [
                            [-indicator, variables.x[vertex][i]],
                            [-indicator, variables.y[vertex][j]],
                            [indicator, -variables.x[vertex][i], -variables.y[vertex][j]],
                        ],
                    )
    if include_injectivity:
        assert variables.a is not None
        for i in range(1, variables.side + 1):
            for j in range(1, variables.side + 1):
                formula = CardEnc.atmost(
                    lits=[variables.a[vertex][i][j] for vertex in range(1, variables.n_vertices + 1)],
                    bound=1,
                    vpool=variables.pool,
                    encoding=EncType.seqcounter,
                )
                injective += _add(solver, formula.clauses)
    return {"order": order, "channel": channel, "cell": cell, "injective": injective}
