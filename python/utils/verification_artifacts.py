from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from python.encoders.symmetry_breaking import SYMMETRY_MODES, select_anchor_vertex
from python.solvers.bound_portfolio import BoundState, portfolio_summary
from python.solvers.entrypoints import METHODS
from python.solvers.solver_factory import SUPPORTED_SOLVERS, validate_solver_for_method
from python.utils.host_grid import HostGrid, deterministic_placement
from python.utils.lower_bound_calculator import initial_upper_bound, theoretical_lower_bound
from python.utils.mtx_parser import parse_mtx_graph
from python.utils.solution_verifier import verify_placement
from python.utils.verification_io import SCHEMA, sha256_file, write_json_atomic

MANIFEST_SCHEMA = "2dbmp-linf-verification-manifest-v3"
TERMINAL_STATUSES = frozenset({"OPTIMAL_BY_UNSAT", "OPTIMAL_BY_BOUND", "OPTIMAL_BY_THEORETICAL_BOUND"})
FEASIBLE_STATUSES = frozenset({"TIMEOUT_FEASIBLE", "INTERRUPTED_FEASIBLE"})
FIXED_K_STATUSES = frozenset({"FIXED_K_SAT", "FIXED_K_TIMEOUT"})


def _require_int(value: Any, name: str, *, nullable: bool = False) -> int | None:
    if nullable and value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _canonical_script(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("method.solver_script must be a string")
    return value.removesuffix(".py").replace("/", ".")


def _method_entry(method: Mapping[str, Any], artifact: Path) -> Any:
    method_id = method.get("id")
    entry = next((candidate for candidate in METHODS.values() if candidate.method_id == method_id), None)
    if entry is None:
        raise ValueError(f"{artifact}: unknown method id")
    if _canonical_script(method.get("solver_script")) != _canonical_script(entry.script):
        raise ValueError(f"{artifact}: solver script does not match method registry")
    if method.get("symmetry") != entry.symmetry:
        raise ValueError(f"{artifact}: method symmetry does not match method registry")
    solver = method.get("solver")
    if solver not in SUPPORTED_SOLVERS:
        raise ValueError(f"{artifact}: unsupported solver metadata")
    try:
        validate_solver_for_method(solver, portfolio=entry.portfolio)
    except ValueError as exc:
        raise ValueError(f"{artifact}: {exc}") from exc
    return entry


def _verify_parallel_portfolio(
    artifact: Path,
    payload: Mapping[str, Any],
    *,
    graph: Any,
    host: HostGrid,
    anchor: int,
    status: str,
    verified_bandwidth: int,
    termination_k: int | None,
) -> None:
    encoding = payload.get("encoding")
    if not isinstance(encoding, dict) or encoding.get("strategy") != "parallel-bounds":
        raise ValueError(f"{artifact}: parallel portfolio encoding is missing")
    portfolio = encoding.get("portfolio")
    if not isinstance(portfolio, dict):
        raise ValueError(f"{artifact}: parallel portfolio metadata is missing")
    if portfolio.get("trust_model") != "sat-backend-unsat":
        raise ValueError(f"{artifact}: parallel portfolio trust model is missing")
    workers_requested = _require_int(portfolio.get("workers_requested"), "portfolio.workers_requested")
    if workers_requested is None or workers_requested <= 0:
        raise ValueError(f"{artifact}: invalid requested worker count")
    workers_effective = portfolio.get("workers_effective")
    if workers_effective is not None:
        effective = _require_int(workers_effective, "portfolio.workers_effective")
        if effective is None or effective <= 0 or effective > workers_requested:
            raise ValueError(f"{artifact}: invalid effective worker count")
    if status in TERMINAL_STATUSES and portfolio.get("status") != status:
        raise ValueError(f"{artifact}: portfolio status mismatch")
    if status not in TERMINAL_STATUSES and portfolio.get("status") not in {"running", status}:
        raise ValueError(f"{artifact}: invalid portfolio status")

    raw_bounds = portfolio.get("bounds")
    if not isinstance(raw_bounds, dict):
        raise ValueError(f"{artifact}: portfolio bounds are missing")
    lower = _require_int(raw_bounds.get("lower_bound"), "portfolio.bounds.lower_bound")
    upper = _require_int(raw_bounds.get("upper_bound"), "portfolio.bounds.upper_bound")
    assert lower is not None and upper is not None
    if upper < lower:
        raise ValueError(f"{artifact}: portfolio bounds are invalid")
    closed = raw_bounds.get("closed")
    if not isinstance(closed, bool) or closed != (lower == upper):
        raise ValueError(f"{artifact}: portfolio closed flag mismatch")
    decisive = raw_bounds.get("decisive")
    if not isinstance(decisive, bool) or decisive != (upper - lower == 1):
        raise ValueError(f"{artifact}: portfolio decisive flag mismatch")
    tested_k = raw_bounds.get("tested_k")
    if not isinstance(tested_k, list) or any(not isinstance(k, int) or isinstance(k, bool) for k in tested_k):
        raise ValueError(f"{artifact}: portfolio tested K ledger is invalid")
    if tested_k != sorted(set(tested_k)):
        raise ValueError(f"{artifact}: portfolio tested K ledger is not canonical")

    initial_lower = _require_int(portfolio.get("lower_bound_initial"), "portfolio.lower_bound_initial")
    initial_upper = _require_int(portfolio.get("upper_bound_initial"), "portfolio.upper_bound_initial")
    host_upper = _require_int(portfolio.get("host_upper_bound"), "portfolio.host_upper_bound")
    if initial_lower != theoretical_lower_bound(graph):
        raise ValueError(f"{artifact}: portfolio initial lower bound mismatch")
    if host_upper != initial_upper_bound(graph.n_vertices):
        raise ValueError(f"{artifact}: portfolio host upper bound mismatch")
    if initial_upper is None or not initial_lower <= initial_upper <= host_upper:
        raise ValueError(f"{artifact}: portfolio initial bounds are invalid")
    seed_positions = deterministic_placement(host, anchor=anchor)
    seed_verified = verify_placement(
        graph.n_vertices,
        graph.edges,
        host,
        seed_positions,
        tested_k=host.diameter,
        anchor=anchor,
        symmetry="triangle",
    )
    if not seed_verified.valid or seed_verified.bandwidth != initial_upper:
        raise ValueError(f"{artifact}: deterministic seed provenance mismatch")

    tasks = portfolio.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError(f"{artifact}: portfolio task ledger is missing")
    records: list[dict[str, Any]] = []
    expected_task_id = 1
    try:
        task_items = sorted(tasks.items(), key=lambda item: int(item[0]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{artifact}: portfolio task IDs are invalid") from exc
    for key, raw_record in task_items:
        if not isinstance(raw_record, dict):
            raise ValueError(f"{artifact}: malformed portfolio task record")
        task_id = _require_int(raw_record.get("task_id"), "portfolio task_id")
        if str(task_id) != str(key) or task_id != expected_task_id:
            raise ValueError(f"{artifact}: portfolio task ID mismatch")
        expected_task_id += 1
        task_status = raw_record.get("status")
        if task_status not in {"RUNNING", "SAT", "UNSAT", "CANCELLED", "ERROR"}:
            raise ValueError(f"{artifact}: invalid portfolio task status")
        if task_status in {"SAT", "UNSAT"}:
            if not isinstance(raw_record.get("accepted"), bool):
                raise ValueError(f"{artifact}: portfolio decision lacks acceptance state")
            if task_status == "SAT" and (
                not isinstance(raw_record.get("bandwidth"), int) or isinstance(raw_record.get("bandwidth"), bool)
            ):
                raise ValueError(f"{artifact}: portfolio SAT task lacks bandwidth")
            task_k = _require_int(raw_record.get("k"), "portfolio task K")
            if task_k is None or task_k < 0 or task_k > host.diameter:
                raise ValueError(f"{artifact}: portfolio task K is outside the host domain")
            decision_order = _require_int(raw_record.get("decision_order"), "portfolio decision order")
            if decision_order is None or decision_order <= 0:
                raise ValueError(f"{artifact}: portfolio decision order is invalid")
        elif raw_record.get("accepted") not in {None, False}:
            raise ValueError(f"{artifact}: non-decision task is marked accepted")
        if status in TERMINAL_STATUSES and task_status == "RUNNING":
            raise ValueError(f"{artifact}: completed portfolio contains a running task")
        records.append(raw_record)

    accepted_records = [record for record in records if record.get("accepted") is True]
    try:
        accepted_records.sort(
            key=lambda record: (
                _require_int(record.get("decision_order"), "accepted decision order"),
                _require_int(record.get("task_id"), "accepted task_id"),
            )
        )
    except ValueError as exc:
        raise ValueError(f"{artifact}: accepted task ordering is invalid") from exc
    decision_orders = [record["decision_order"] for record in accepted_records]
    if len(decision_orders) != len(set(decision_orders)):
        raise ValueError(f"{artifact}: accepted task ordering is not unique")
    replay = BoundState(initial_lower, initial_upper)
    accepted_k: list[int] = []
    for record in accepted_records:
        if record.get("accepted") is not True:
            continue
        k = _require_int(record.get("k"), "accepted portfolio task K")
        if k is None:
            raise ValueError(f"{artifact}: accepted task lacks K")
        try:
            if record.get("status") == "SAT":
                bandwidth = _require_int(record.get("bandwidth"), "accepted SAT bandwidth")
                if k in replay.tested:
                    raise ValueError(f"bound K={k} was already processed")
                if replay.lower <= k < replay.upper:
                    replay = replay.apply(k=k, result="SAT", bandwidth=bandwidth)
                elif k < replay.lower:
                    raise ValueError(f"SAT K={k} contradicts lower bound {replay.lower}")
                elif bandwidth is None or bandwidth >= replay.upper:
                    raise ValueError(f"stale SAT K={k} was not an improving witness")
                else:
                    replay = replay.record_sat_witness(k=k, bandwidth=bandwidth)
            elif record.get("status") == "UNSAT":
                replay = replay.apply(k=k, result="UNSAT")
            else:
                raise ValueError("accepted task is not a SAT/UNSAT decision")
        except ValueError as exc:
            raise ValueError(f"{artifact}: portfolio ledger replay failed: {exc}") from exc
        accepted_k.append(k)
    if sorted(set(accepted_k)) != tested_k:
        raise ValueError(f"{artifact}: portfolio tested K does not match accepted task ledger")
    if dict(portfolio_summary(replay)) != raw_bounds:
        raise ValueError(f"{artifact}: portfolio bounds do not match accepted task replay")
    if upper != verified_bandwidth:
        raise ValueError(f"{artifact}: portfolio upper bound does not match the witness")
    running_ids = sorted(record["task_id"] for record in records if record.get("status") == "RUNNING")
    active_tasks = portfolio.get("active_tasks")
    if active_tasks != running_ids:
        raise ValueError(f"{artifact}: portfolio active task list mismatch")

    if status in TERMINAL_STATUSES:
        if not closed or lower != upper or upper != verified_bandwidth:
            raise ValueError(f"{artifact}: terminal portfolio bounds do not match the witness")
    if status == "OPTIMAL_BY_UNSAT":
        matching = [
            record
            for record in records
            if record.get("status") == "UNSAT"
            and record.get("accepted") is True
            and record.get("k") == termination_k
        ]
        if not matching or raw_bounds.get("unsat_boundary") != termination_k:
            raise ValueError(f"{artifact}: UNSAT certificate task is missing")
    winning_solver = portfolio.get("winning_solver")
    winning_k = _require_int(portfolio.get("winning_k"), "portfolio.winning_k")
    if winning_solver == "deterministic-seed":
        if winning_k != verified_bandwidth:
            raise ValueError(f"{artifact}: deterministic seed provenance mismatch")
    else:
        if not any(
            record.get("status") == "SAT"
            and record.get("accepted") is True
            and record.get("k") == winning_k
            and record.get("bandwidth") == verified_bandwidth
            and record.get("solver") == winning_solver
            for record in records
        ):
            raise ValueError(f"{artifact}: winning SAT task is missing")


def verify_document(path: str | Path, payload: Mapping[str, Any]) -> str:
    artifact = Path(path)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{artifact}: unsupported schema")
    instance, method, outcome, bounds, witness = (
        payload.get("instance"), payload.get("method"), payload.get("outcome"), payload.get("bounds"), payload.get("witness")
    )
    if not all(isinstance(value, dict) for value in (instance, method, outcome, bounds, witness)):
        raise ValueError(f"{artifact}: missing required object")
    input_path = Path(instance.get("file", ""))
    if not input_path.is_file():
        raise ValueError(f"{artifact}: input does not exist")
    if input_path.resolve() == artifact.resolve():
        raise ValueError(f"{artifact}: artifact cannot overwrite its input")
    if instance.get("sha256") != sha256_file(input_path):
        raise ValueError(f"{artifact}: input hash mismatch")
    graph = parse_mtx_graph(input_path)
    method_entry = _method_entry(method, artifact)
    matrix = instance.get("matrix") or {}
    declared = (
        instance.get("vertices"), instance.get("graph_edges"), instance.get("graph_interpretation"),
        matrix.get("rows"), matrix.get("cols"), matrix.get("declared_entries"), matrix.get("field"), matrix.get("symmetry"),
    )
    expected = (
        graph.n_vertices, graph.n_edges, graph.interpretation, graph.rows, graph.cols,
        graph.declared_entries, graph.field, graph.symmetry,
    )
    if declared != expected:
        raise ValueError(f"{artifact}: graph metadata mismatch")
    grid = method.get("grid") or {}
    side = _require_int(grid.get("rows"), "grid.rows")
    if grid.get("cols") != side or grid.get("policy") != "ceil-sqrt-square":
        raise ValueError(f"{artifact}: invalid host metadata")
    host = HostGrid(side, graph.n_vertices)
    if side != initial_upper_bound(graph.n_vertices) + 1:
        raise ValueError(f"{artifact}: host side mismatch")
    symmetry = method.get("symmetry")
    if symmetry not in SYMMETRY_MODES:
        raise ValueError(f"{artifact}: invalid symmetry")
    anchor = _require_int(method.get("anchor"), "anchor")
    if anchor != select_anchor_vertex(graph):
        raise ValueError(f"{artifact}: anchor does not match method registry rule")
    raw_positions = witness.get("positions")
    if not isinstance(raw_positions, dict):
        raise ValueError(f"{artifact}: positions must be an object")
    positions: dict[int, tuple[int, int]] = {}
    for key, coordinate in raw_positions.items():
        if not isinstance(coordinate, list) or len(coordinate) != 2 or any(not isinstance(value, int) or isinstance(value, bool) for value in coordinate):
            raise ValueError(f"{artifact}: malformed position")
        vertex = int(key)
        if vertex in positions:
            raise ValueError(f"{artifact}: duplicate vertex key")
        positions[vertex] = (coordinate[0], coordinate[1])
    tested_k = _require_int(witness.get("constraint_k"), "constraint_k")
    verified = verify_placement(graph.n_vertices, graph.edges, host, positions, tested_k=tested_k, anchor=anchor, symmetry=symmetry)
    if not verified.valid or verified.bandwidth is None:
        raise ValueError(f"{artifact}: invalid witness: {verified.error}")
    final_verified = _require_int(bounds.get("final_verified"), "final_verified")
    if final_verified != verified.bandwidth or witness.get("actual_bandwidth") != verified.bandwidth:
        raise ValueError(f"{artifact}: bandwidth mismatch")
    if bounds.get("initial_upper_bound") != initial_upper_bound(graph.n_vertices):
        raise ValueError(f"{artifact}: initial upper bound mismatch")
    status = outcome.get("status")
    if status not in TERMINAL_STATUSES | FEASIBLE_STATUSES | FIXED_K_STATUSES:
        raise ValueError(f"{artifact}: invalid status")
    optimal = _require_int(bounds.get("optimal"), "optimal", nullable=True)
    lower = _require_int(bounds.get("certified_lower"), "certified_lower", nullable=True)
    termination_k = _require_int(outcome.get("termination_k"), "termination_k", nullable=True)
    encoding = payload.get("encoding")
    if method_entry.portfolio and (not isinstance(encoding, dict) or encoding.get("strategy") != "parallel-bounds"):
        raise ValueError(f"{artifact}: portfolio method lacks portfolio encoding")
    if not method_entry.portfolio and isinstance(encoding, dict) and encoding.get("strategy") == "parallel-bounds":
        raise ValueError(f"{artifact}: non-portfolio method contains portfolio encoding")
    if status == "OPTIMAL_BY_BOUND":
        if verified.bandwidth not in {0, 1} or optimal != verified.bandwidth or lower != verified.bandwidth or termination_k != verified.bandwidth:
            raise ValueError(f"{artifact}: invalid bound optimality claim")
    elif status == "OPTIMAL_BY_UNSAT":
        if optimal != verified.bandwidth or lower != verified.bandwidth or termination_k != verified.bandwidth - 1:
            raise ValueError(f"{artifact}: invalid UNSAT optimality metadata")
    elif status == "OPTIMAL_BY_THEORETICAL_BOUND":
        if (
            optimal != verified.bandwidth
            or lower != verified.bandwidth
            or termination_k != verified.bandwidth
            or theoretical_lower_bound(graph) != verified.bandwidth
        ):
            raise ValueError(f"{artifact}: invalid theoretical-bound optimality claim")
    elif status == "FIXED_K_SAT":
        if not isinstance(encoding, dict) or _require_int(encoding.get("target_k"), "encoding.target_k") != tested_k:
            raise ValueError(f"{artifact}: fixed-K target metadata mismatch")
        if not 0 <= tested_k <= host.diameter:
            raise ValueError(f"{artifact}: fixed-K target is outside the host domain")
        if verified.bandwidth > tested_k or optimal is not None or lower is not None or termination_k is not None:
            raise ValueError(f"{artifact}: invalid fixed-K SAT metadata")
    elif status == "FIXED_K_TIMEOUT":
        target = _require_int(encoding.get("target_k"), "encoding.target_k") if isinstance(encoding, dict) else None
        if not isinstance(encoding, dict) or target is None:
            raise ValueError(f"{artifact}: fixed-K timeout lacks a target")
        if not 0 <= target <= host.diameter:
            raise ValueError(f"{artifact}: fixed-K target is outside the host domain")
        if optimal is not None or lower is not None or termination_k is not None:
            raise ValueError(f"{artifact}: fixed-K timeout claims optimality")
    elif optimal is not None or lower is not None or termination_k is not None:
        raise ValueError(f"{artifact}: feasible result claims optimality")
    if method_entry.portfolio:
        _verify_parallel_portfolio(
            artifact,
            payload,
            graph=graph,
            host=host,
            anchor=anchor,
            status=status,
            verified_bandwidth=verified.bandwidth,
            termination_k=termination_k,
        )
    return status


def verify_artifact(path: str | Path) -> str:
    artifact = Path(path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact}: JSON root must be an object")
    return verify_document(artifact, payload)


def verify_tree(root: str | Path) -> Counter[str]:
    directory = Path(root).resolve()
    manifest_path = directory / "manifest.json"
    artifacts = sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*.json")
        if path.name != "manifest.json"
        and ".checkpoint." not in path.name
        and not any(part == "portfolio" or part.endswith(".portfolio") for part in path.relative_to(directory).parts)
    )
    if not artifacts:
        raise ValueError(f"no artifacts under {directory}")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("artifacts") != artifacts:
            raise ValueError("verification manifest does not match artifact tree")
    counts: Counter[str] = Counter()
    for relative in artifacts:
        counts[verify_artifact(directory / relative)] += 1
    return counts


def write_manifest(root: str | Path) -> Path:
    directory = Path(root).resolve()
    artifacts = sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*.json")
        if path.name != "manifest.json"
        and ".checkpoint." not in path.name
        and not any(part == "portfolio" or part.endswith(".portfolio") for part in path.relative_to(directory).parts)
    )
    if not artifacts:
        raise ValueError(f"no artifacts under {directory}")
    path = directory / "manifest.json"
    write_json_atomic(path, {"schema": MANIFEST_SCHEMA, "artifacts": artifacts})
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify L-infinity artifacts")
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    group = verify.add_mutually_exclusive_group(required=True)
    group.add_argument("--artifact", type=Path)
    group.add_argument("--root", type=Path)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "verify":
        if args.artifact:
            print(verify_artifact(args.artifact))
        else:
            print(dict(sorted(verify_tree(args.root).items())))
    else:
        print(write_manifest(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
