from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Mapping

from pysat.formula import IDPool

from python.encoders.distance_encoder_cutoff import add_distance_layer
from python.encoders.position_constraints import PositionVariables, encode_position_constraints
from python.encoders.symmetry_breaking import encode_symmetry_breaking_constraints, select_anchor_vertex
from python.solvers.solver_factory import DEFAULT_SOLVER, create_sat_solver
from python.utils.host_grid import HostGrid, deterministic_placement
from python.utils.lower_bound_calculator import initial_upper_bound
from python.utils.mtx_parser import ParsedMtxGraph
from python.utils.solution_verifier import verify_placement
from python.utils.verification_io import build_verification_document, write_json_atomic


@dataclass(frozen=True, slots=True)
class Incumbent:
    constraint_k: int
    bandwidth: int
    positions: Mapping[int, tuple[int, int]]
    elapsed_s: float
    solve_count: int


@dataclass(frozen=True, slots=True)
class SolveResult:
    optimal_k: int
    positions: Mapping[int, tuple[int, int]]
    status: str
    termination: str
    termination_k: int
    initial_upper_bound: int
    solver: str
    symmetry: str
    anchor: int
    elapsed_s: float
    solve_count: int
    sat_count: int
    unsat_count: int
    encoding: Mapping[str, object]


class IncrementalBandwidthSolverCutoffSymmetry:
    """Exact persistent SAT solver for the ceil-sqrt square L-infinity host."""

    def __init__(
        self,
        graph: ParsedMtxGraph,
        *,
        symmetry: str = "triangle",
        solver_name: str = DEFAULT_SOLVER,
        on_incumbent: Callable[[Incumbent], None] | None = None,
        on_progress: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        self.graph = graph
        self.host = HostGrid.square(graph.n_vertices)
        self.symmetry = symmetry
        self.solver_name = solver_name
        self.on_incumbent = on_incumbent
        self.on_progress = on_progress

    @staticmethod
    def _snapshot(variables: int, clauses: int, *, complete: bool | None = None) -> dict[str, object]:
        snapshot: dict[str, object] = {"variables": variables, "clauses": clauses}
        if complete is not None:
            snapshot["complete"] = complete
        return snapshot

    def _publish_progress(self, encoding: Mapping[str, object]) -> None:
        if self.on_progress:
            self.on_progress(copy.deepcopy(dict(encoding)))

    @staticmethod
    def _mark_final(encoding: dict[str, object], *, complete: bool) -> None:
        phases = encoding["phases"]
        assert isinstance(phases, dict)
        current = phases["current"]
        assert isinstance(current, dict)
        phases["final"] = {**current, "complete": complete}

    def solve(self) -> SolveResult:
        started = time.perf_counter()
        anchor = select_anchor_vertex(self.graph)
        upper = initial_upper_bound(self.graph.n_vertices)
        self._trace({"event": "search-start", "symmetry": self.symmetry, "upper_bound": upper})
        if not self.graph.edges:
            zero = self._snapshot(0, 0, complete=True)
            encoding: dict[str, object] = {
                "strategy": "incremental",
                "variables_primary": 0,
                "distance_by_k": {},
                "phases": {"initial": zero, "before_first_k": zero, "after_first_k": zero, "current": zero, "final": zero},
                "work": {"generated_variables": 0, "generated_clauses": 0},
            }
            self._publish_progress(encoding)
            positions = deterministic_placement(self.host, anchor=anchor)
            verified = verify_placement(
                self.graph.n_vertices, self.graph.edges, self.host, positions, tested_k=0, anchor=anchor, symmetry=self.symmetry
            )
            if not verified.valid:
                raise RuntimeError(f"seed witness is invalid: {verified.error}")
            incumbent = Incumbent(0, 0, dict(positions), time.perf_counter() - started, 0)
            if self.on_incumbent:
                self.on_incumbent(incumbent)
            return SolveResult(0, dict(positions), "OPTIMAL_BY_BOUND", "zero-edge-graph", 0, upper, self.solver_name, self.symmetry, anchor, time.perf_counter() - started, 0, 0, 0, encoding)

        pool = IDPool()
        variables = PositionVariables.create(self.graph.n_vertices, self.host.side, pool)
        solver = create_sat_solver(self.solver_name)
        added_layers: list[int] = []
        incumbent: Incumbent | None = None
        solve_count = sat_count = unsat_count = 0
        initial = self._snapshot(pool.top, 0)
        encoding: dict[str, object] = {
            "strategy": "incremental",
            "variables_primary": pool.top,
            "distance_by_k": {},
            "phases": {"initial": initial, "current": initial, "final": self._snapshot(pool.top, 0, complete=False)},
            "work": {"generated_variables": pool.top, "generated_clauses": 0},
        }
        self._publish_progress(encoding)
        try:
            anchor, clauses = encode_symmetry_breaking_constraints(
                self.graph, self.host.side, variables.lx, variables.ly, self.symmetry
            )
            for clause in clauses:
                solver.add_clause(clause)
            encoding["symmetry"] = len(clauses)
            position_counts = encode_position_constraints(solver, variables)
            encoding.update(position_counts)
            clause_total = len(clauses) + sum(position_counts.values())
            phases = encoding["phases"]
            assert isinstance(phases, dict)
            before_first_k = self._snapshot(pool.top, clause_total)
            phases["before_first_k"] = before_first_k
            phases["current"] = before_first_k
            self._mark_final(encoding, complete=False)
            encoding["work"] = {"generated_variables": pool.top, "generated_clauses": clause_total}
            self._publish_progress(encoding)
            current_k = upper
            first_layer = True
            while current_k >= 1:
                if added_layers and current_k >= added_layers[-1]:
                    raise RuntimeError("incremental distance layers must be strictly decreasing")
                distance_count = add_distance_layer(solver, self.graph, variables, current_k)
                added_layers.append(current_k)
                distance_by_k = encoding["distance_by_k"]
                assert isinstance(distance_by_k, dict)
                distance_by_k[current_k] = distance_count
                clause_total += distance_count
                current = self._snapshot(pool.top, clause_total)
                phases["current"] = current
                if first_layer:
                    phases["after_first_k"] = current
                    first_layer = False
                self._mark_final(encoding, complete=False)
                encoding["work"] = {"generated_variables": pool.top, "generated_clauses": clause_total}
                self._publish_progress(encoding)
                solve_count += 1
                self._trace({"event": "attempt-start", "k": current_k, "solve_count": solve_count})
                solve_started = time.perf_counter()
                satisfiable = solver.solve()
                self._trace(
                    {
                        "event": "attempt-result",
                        "k": current_k,
                        "result": "SAT" if satisfiable else "UNSAT",
                        "elapsed_s": time.perf_counter() - solve_started,
                    }
                )
                if satisfiable:
                    sat_count += 1
                    positions = variables.extract_positions(solver.get_model())
                    verified = verify_placement(
                        self.graph.n_vertices,
                        self.graph.edges,
                        self.host,
                        positions,
                        tested_k=current_k,
                        anchor=anchor,
                        symmetry=self.symmetry,
                    )
                    if not verified.valid or verified.bandwidth is None:
                        raise RuntimeError(f"SAT witness is invalid: {verified.error}")
                    if verified.bandwidth < 1:
                        raise RuntimeError("graph with edges has zero bandwidth")
                    incumbent = Incumbent(current_k, verified.bandwidth, dict(positions), time.perf_counter() - started, solve_count)
                    if self.on_incumbent:
                        self.on_incumbent(incumbent)
                    if verified.bandwidth == 1:
                        self._mark_final(encoding, complete=True)
                        return self._result(incumbent, "OPTIMAL_BY_BOUND", "minimum-positive-bandwidth", 1, upper, anchor, started, solve_count, sat_count, unsat_count, encoding)
                    current_k = verified.bandwidth - 1
                else:
                    unsat_count += 1
                    if incumbent is None:
                        raise RuntimeError(f"initial upper bound K={current_k} is unexpectedly UNSAT")
                    self._mark_final(encoding, complete=True)
                    return self._result(incumbent, "OPTIMAL_BY_UNSAT", "unsat-below-verified-incumbent", current_k, upper, anchor, started, solve_count, sat_count, unsat_count, encoding)
            raise RuntimeError("search ended without an optimal witness")
        finally:
            solver.delete()

    @staticmethod
    def _trace(event: Mapping[str, object]) -> None:
        print(json.dumps(event, sort_keys=True), file=sys.stdout, flush=True)

    def _result(
        self,
        incumbent: Incumbent,
        status: str,
        termination: str,
        termination_k: int,
        upper: int,
        anchor: int,
        started: float,
        solve_count: int,
        sat_count: int,
        unsat_count: int,
        encoding: Mapping[str, object],
    ) -> SolveResult:
        return SolveResult(
            incumbent.bandwidth,
            dict(incumbent.positions),
            status,
            termination,
            termination_k,
            upper,
            self.solver_name,
            self.symmetry,
            anchor,
            time.perf_counter() - started,
            solve_count,
            sat_count,
            unsat_count,
            dict(encoding),
        )


def run_worker(
    instance: str | Path,
    checkpoint: str | Path,
    *,
    symmetry: str,
    method_id: str,
    solver_name: str,
    timeout_s: float,
    solver_class=IncrementalBandwidthSolverCutoffSymmetry,
    solver_script: str | None = None,
) -> dict[str, object]:
    """Worker protocol shared by the primary Triangle and comparison methods."""

    from python.utils.lower_bound_calculator import initial_upper_bound
    from python.utils.mtx_parser import parse_mtx_graph

    input_path, checkpoint_path = Path(instance).resolve(), Path(checkpoint)
    script = solver_script or __name__
    graph = parse_mtx_graph(input_path)
    host = HostGrid.square(graph.n_vertices)
    anchor = select_anchor_vertex(graph)

    latest_encoding: dict[str, object] = {}

    def publish_progress(encoding: Mapping[str, object]) -> None:
        nonlocal latest_encoding
        latest_encoding = copy.deepcopy(dict(encoding))
        if checkpoint_path.is_file():
            document = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            document["encoding"] = latest_encoding
            write_json_atomic(checkpoint_path, document)

    def publish(incumbent: Incumbent) -> None:
        write_json_atomic(
            checkpoint_path,
            build_verification_document(
                instance_file=input_path,
                graph=graph,
                host=host,
                method_id=method_id,
                solver=solver_name,
                solver_script=script,
                symmetry=symmetry,
                anchor=anchor,
                status="TIMEOUT_FEASIBLE",
                termination="in-progress",
                termination_k=None,
                elapsed_s=incumbent.elapsed_s,
                timeout_s=timeout_s,
                initial_upper_bound=initial_upper_bound(graph.n_vertices),
                certified_lower=None,
                final_verified=incumbent.bandwidth,
                optimal=None,
                constraint_k=incumbent.constraint_k,
                positions=dict(incumbent.positions),
                execution={"solve_count": incumbent.solve_count},
                encoding=latest_encoding,
            ),
        )

    solver = solver_class(graph, solver_name=solver_name, on_incumbent=publish, on_progress=publish_progress)
    if solver.symmetry != symmetry:
        raise RuntimeError(f"worker symmetry mismatch: solver={solver.symmetry}, declared={symmetry}")
    result = solver.solve()
    write_json_atomic(
        checkpoint_path,
        build_verification_document(
            instance_file=input_path,
            graph=graph,
            host=host,
            method_id=method_id,
            solver=solver_name,
            solver_script=script,
            symmetry=symmetry,
            anchor=result.anchor,
            status=result.status,
            termination=result.termination,
            termination_k=result.termination_k,
            elapsed_s=result.elapsed_s,
            timeout_s=timeout_s,
            initial_upper_bound=result.initial_upper_bound,
            certified_lower=result.optimal_k,
            final_verified=result.optimal_k,
            optimal=result.optimal_k,
            constraint_k=result.optimal_k,
            positions=dict(result.positions),
            execution={"solve_count": result.solve_count, "sat_count": result.sat_count, "unsat_count": result.unsat_count},
            encoding=result.encoding,
        ),
    )
    return {"status": result.status, "optimal_k": result.optimal_k}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exact L-infinity Triangle worker")
    parser.add_argument("instance", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--solver", default=DEFAULT_SOLVER)
    parser.add_argument("--timeout", type=float, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_worker(args.instance, args.checkpoint, symmetry="triangle", method_id="exact-linf-triangle", solver_name=args.solver, timeout_s=args.timeout, solver_script="python.solvers.incremental_bandwidth_solver_cutoff_symmetry_nodrsa"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
