from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Mapping

from pysat.formula import IDPool

from python.encoders.distance_encoder_cutoff import add_distance_layer
from python.encoders.position_constraints import PositionVariables, encode_position_constraints
from python.encoders.symmetry_breaking import encode_symmetry_breaking_constraints, select_anchor_vertex
from python.solvers.solver_factory import create_sat_solver
from python.utils.host_grid import HostGrid
from python.utils.mtx_parser import ParsedMtxGraph
from python.utils.solution_verifier import verify_placement


class ClauseRecorder:
    """Minimal SAT-like sink used to build a reusable immutable base formula."""

    def __init__(self) -> None:
        self.clauses: list[list[int]] = []

    def add_clause(self, clause: Any) -> None:
        self.clauses.append([int(literal) for literal in clause])


@dataclass(frozen=True, slots=True)
class FixedKTemplate:
    graph: ParsedMtxGraph
    host: HostGrid
    symmetry: str
    anchor: int
    pool: IDPool
    variables: PositionVariables
    base_clauses: tuple[tuple[int, ...], ...]
    variables_primary: int
    base_variables: int
    base_clauses_count: int
    symmetry_clauses: int
    position_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class FixedKDecision:
    k: int
    result: str
    solver: str
    elapsed_s: float
    encoding: Mapping[str, object]
    positions: Mapping[int, tuple[int, int]] | None = None
    bandwidth: int | None = None
    error: str | None = None
    solver_stats: Mapping[str, object] | None = None


def build_fixed_k_template(graph: ParsedMtxGraph, *, symmetry: str = "triangle") -> FixedKTemplate:
    host = HostGrid.square(graph.n_vertices)
    pool = IDPool()
    variables = PositionVariables.create(graph.n_vertices, host.side, pool)
    variables_primary = pool.top
    recorder = ClauseRecorder()
    anchor, symmetry_clauses = encode_symmetry_breaking_constraints(
        graph, host.side, variables.lx, variables.ly, symmetry
    )
    for clause in symmetry_clauses:
        recorder.add_clause(clause)
    position_counts = encode_position_constraints(recorder, variables)
    return FixedKTemplate(
        graph=graph,
        host=host,
        symmetry=symmetry,
        anchor=anchor,
        pool=pool,
        variables=variables,
        base_clauses=tuple(tuple(clause) for clause in recorder.clauses),
        variables_primary=variables_primary,
        base_variables=pool.top,
        base_clauses_count=len(recorder.clauses),
        symmetry_clauses=len(symmetry_clauses),
        position_counts=dict(position_counts),
    )


def _set_affinity(cpu: int | None) -> None:
    if cpu is None or not hasattr(os, "sched_setaffinity"):
        return
    os.sched_setaffinity(0, {cpu})


def decide_fixed_k(
    template: FixedKTemplate,
    k: int,
    solver_name: str,
    *,
    cpu: int | None = None,
    task_id: int | None = None,
) -> FixedKDecision:
    if not isinstance(k, int) or isinstance(k, bool) or not 0 <= k <= template.host.diameter:
        raise ValueError(f"K must be an integer in 0..{template.host.diameter}")
    started = time.perf_counter()
    _set_affinity(cpu)
    solver = None
    distance_count = 0
    result = "ERROR"
    positions: Mapping[int, tuple[int, int]] | None = None
    bandwidth: int | None = None
    error: str | None = None
    stats: Mapping[str, object] | None = None
    try:
        solver = create_sat_solver(solver_name, one_shot=True)
        solver.append_formula(template.base_clauses)
        distance_count = add_distance_layer(solver, template.graph, template.variables, k)
        satisfiable = solver.solve()
        if satisfiable is None:
            error = "solver returned UNKNOWN"
        elif not satisfiable:
            result = "UNSAT"
        else:
            model = solver.get_model()
            if model is None:
                error = "SAT solver returned no model"
            else:
                extracted = template.variables.extract_positions(model)
                verified = verify_placement(
                    template.graph.n_vertices,
                    template.graph.edges,
                    template.host,
                    extracted,
                    tested_k=k,
                    anchor=template.anchor,
                    symmetry=template.symmetry,
                )
                if not verified.valid or verified.bandwidth is None:
                    error = f"invalid SAT witness: {verified.error}"
                else:
                    result = "SAT"
                    positions = dict(extracted)
                    bandwidth = verified.bandwidth
        try:
            stats = dict(solver.accum_stats())
        except (AttributeError, RuntimeError, TypeError):
            stats = None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if solver is not None:
            solver.delete()
    elapsed = time.perf_counter() - started
    encoding = {
        "strategy": "fixed-k-one-shot",
        "task_id": task_id,
        "k": k,
        "solver": solver_name,
        "variables_primary": template.variables_primary,
        "base_variables": template.base_variables,
        "base_clauses": template.base_clauses_count,
        "distance_clauses": distance_count,
        "total_variables": template.base_variables,
        "total_clauses": template.base_clauses_count + distance_count,
        "symmetry_clauses": template.symmetry_clauses,
        **template.position_counts,
    }
    return FixedKDecision(
        k=k,
        result=result if error is None else "ERROR",
        solver=solver_name,
        elapsed_s=elapsed,
        encoding=encoding,
        positions=positions,
        bandwidth=bandwidth,
        error=error,
        solver_stats=stats,
    )
