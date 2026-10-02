from __future__ import annotations

import argparse
import copy
import json
import math
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from python.encoders.symmetry_breaking import select_anchor_vertex
from python.benchmarks.mdo_best_k import mdo_best_k
from python.solvers.entrypoints import METHODS, get_method, worker_path
from python.solvers.solver_factory import (
    DEFAULT_SOLVER,
    PORTFOLIO_DEFAULT_SOLVER,
    SUPPORTED_SOLVERS,
    validate_solver_for_method,
)
from python.utils.host_grid import HostGrid, deterministic_placement
from python.utils.lower_bound_calculator import initial_upper_bound, theoretical_lower_bound
from python.utils.mtx_parser import parse_mtx_graph
from python.utils.process_supervisor import ProcessResult, run_supervised
from python.utils.solution_verifier import compute_bandwidth
from python.utils.verification_artifacts import TERMINAL_STATUSES, verify_document
from python.utils.verification_io import SCHEMA, build_verification_document, write_json_atomic

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SKIP_SCHEMA = "2dbmp-linf-run-skip-v1"


def _error(instance: Path, method: str, solver: str, error: str, process: ProcessResult | None = None) -> dict[str, Any]:
    return {
        "schema": "2dbmp-linf-run-error-v2",
        "instance": str(instance),
        "method": method,
        "solver": solver,
        "error": error,
        "execution": None if process is None else {
            "returncode": process.returncode,
            "elapsed_s": process.elapsed_s,
            "timed_out": process.timed_out,
            "termination_signal": process.termination_signal,
        },
    }


def _skip(instance: Path, method: str, solver: str) -> dict[str, Any]:
    return {
        "schema": SKIP_SCHEMA,
        "instance": str(instance),
        "method": method,
        "solver": solver,
        "status": "SKIPPED_NO_MDO_K",
        "reason": "no published M_D0 best K is available",
    }


def _finalize_feasible(payload: dict[str, Any], status: str, termination: str, process: ProcessResult) -> dict[str, Any]:
    updated = copy.deepcopy(payload)
    if updated.get("schema") != SCHEMA or not isinstance(updated.get("outcome"), dict):
        raise ValueError("checkpoint is not a verification document")
    portfolio = updated.get("encoding", {}).get("portfolio") if isinstance(updated.get("encoding"), dict) else None
    current_status = updated["outcome"].get("status")
    if current_status in TERMINAL_STATUSES:
        updated["execution"].update(
            {"returncode": process.returncode, "timed_out": process.timed_out, "termination_signal": process.termination_signal}
        )
        return updated
    if isinstance(portfolio, dict) and portfolio.get("status") in TERMINAL_STATUSES:
        updated["outcome"].update(
            {"status": portfolio["status"], "termination": termination, "termination_k": updated["outcome"].get("termination_k")}
        )
        updated["execution"].update(
            {"returncode": process.returncode, "timed_out": process.timed_out, "termination_signal": process.termination_signal}
        )
        return updated
    updated["outcome"].update({"status": status, "termination": termination, "termination_k": None, "elapsed_s": process.elapsed_s})
    updated["bounds"].update({"certified_lower": None, "optimal": None})
    updated["execution"].update({"returncode": process.returncode, "timed_out": process.timed_out, "termination_signal": process.termination_signal})
    if isinstance(portfolio, dict):
        portfolio["status"] = status
    return updated


def run(
    instance: str | Path,
    *,
    method_key: str,
    solver_name: str,
    timeout_s: float,
    output: str | Path,
    target_k: int | None = None,
    stdout_path: str | Path | None = None,
    stderr_path: str | Path | None = None,
    portfolio_log_dir: str | Path | None = None,
    workers: int = 4,
    fallback_solver: str = "cadical300",
    cpu_policy: str = "physical",
    manager_core: str = "separate",
) -> dict[str, Any]:
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout must be a finite positive number")
    input_path, output_path = Path(instance).resolve(), Path(output).resolve()
    if input_path == output_path:
        raise ValueError("output path must differ from input path")
    method = get_method(method_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if solver_name not in SUPPORTED_SOLVERS:
        payload = _error(input_path, method_key, solver_name, f"unsupported solver {solver_name!r}")
        write_json_atomic(output_path, payload)
        return payload
    try:
        validate_solver_for_method(solver_name, portfolio=method.portfolio)
    except ValueError as exc:
        payload = _error(input_path, method_key, solver_name, str(exc))
        write_json_atomic(output_path, payload)
        return payload
    if method.portfolio:
        if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
            payload = _error(input_path, method_key, solver_name, "workers must be a positive integer")
            write_json_atomic(output_path, payload)
            return payload
        if fallback_solver not in SUPPORTED_SOLVERS:
            payload = _error(input_path, method_key, solver_name, f"unsupported fallback solver {fallback_solver!r}")
            write_json_atomic(output_path, payload)
            return payload
        if cpu_policy not in {"physical", "logical", "none"}:
            payload = _error(input_path, method_key, solver_name, f"unsupported cpu policy {cpu_policy!r}")
            write_json_atomic(output_path, payload)
            return payload
        if manager_core not in {"separate", "none"}:
            payload = _error(input_path, method_key, solver_name, f"unsupported manager core policy {manager_core!r}")
            write_json_atomic(output_path, payload)
            return payload
    if target_k is not None and not method.fixed_k_mdo:
        raise ValueError("target_k is supported only by the fixed-k-mdo method")
    resolved_target_k = target_k if target_k is not None else (mdo_best_k(input_path) if method.fixed_k_mdo else None)
    target_source = "explicit" if target_k is not None else "mdo"
    if method.fixed_k_mdo and resolved_target_k is None:
        payload = _skip(input_path, method_key, solver_name)
        write_json_atomic(output_path, payload)
        return payload
    stdout = Path(stdout_path).resolve() if stdout_path is not None else output_path.with_suffix(output_path.suffix + ".stdout.log")
    stderr = Path(stderr_path).resolve() if stderr_path is not None else output_path.with_suffix(output_path.suffix + ".stderr.log")
    if stdout == output_path or stderr == output_path or stdout == stderr:
        raise ValueError("log paths must differ from each other and from output path")
    if stdout == input_path or stderr == input_path:
        raise ValueError("log paths must differ from the input path")
    selected_portfolio_log_dir = (
        Path(portfolio_log_dir).resolve()
        if portfolio_log_dir is not None
        else output_path.parent / f"{output_path.stem}.portfolio"
    )
    checkpoint = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.checkpoint.json")
    run_started = time.perf_counter()
    try:
        graph = parse_mtx_graph(input_path)
        host = HostGrid.square(graph.n_vertices)
        if method.fixed_k_mdo and (
            not isinstance(resolved_target_k, int)
            or isinstance(resolved_target_k, bool)
            or not 0 <= resolved_target_k <= host.diameter
        ):
            raise ValueError(f"target_k must be an integer in 0..{host.diameter}")
        anchor = select_anchor_vertex(graph)
        seed_positions = deterministic_placement(host, anchor=anchor)
        seed_bandwidth = compute_bandwidth(seed_positions, graph.edges)
        write_json_atomic(
            checkpoint,
            build_verification_document(
                instance_file=input_path,
                graph=graph,
                host=host,
                method_id=method.method_id,
                solver=solver_name,
                solver_script=method.script,
                symmetry=method.symmetry,
                anchor=anchor,
                status="TIMEOUT_FEASIBLE",
                termination="supervisor-seed",
                termination_k=None,
                elapsed_s=0.0,
                timeout_s=timeout_s,
                initial_upper_bound=initial_upper_bound(graph.n_vertices),
                certified_lower=None,
                final_verified=seed_bandwidth,
                optimal=None,
                constraint_k=initial_upper_bound(graph.n_vertices),
                positions=seed_positions,
                encoding=(
                    {
                        "strategy": "parallel-bounds",
                        "workers": workers,
                        "fallback_solver": fallback_solver,
                        "cpu_policy": cpu_policy,
                        "manager_core": manager_core,
                        "portfolio": {
                            "strategy": "parallel-bounds",
                            "lower_bound_initial": theoretical_lower_bound(graph),
                            "upper_bound_initial": seed_bandwidth,
                            "host_upper_bound": initial_upper_bound(graph.n_vertices),
                            "workers_requested": workers,
                            "workers_effective": None,
                            "cpu_policy": cpu_policy,
                            "manager_core": manager_core,
                            "winning_solver": "deterministic-seed",
                            "winning_k": seed_bandwidth,
                            "status": "running",
                            "bounds": {
                                "lower_bound": theoretical_lower_bound(graph),
                                "upper_bound": seed_bandwidth,
                                "decisive": seed_bandwidth - theoretical_lower_bound(graph) == 1,
                                "closed": seed_bandwidth == theoretical_lower_bound(graph),
                                "tested_k": [],
                                "unsat_boundary": None,
                            },
                            "active_tasks": [],
                            "failed_k": [],
                            "fallback_k": [],
                            "tasks": {},
                            "trust_model": "sat-backend-unsat",
                        },
                    }
                    if method.portfolio
                    else ({} if resolved_target_k is None else {"strategy": "fixed-k-mdo", "target_k": resolved_target_k})
                ),
            ),
        )
        worker_path(SOURCE_ROOT, method)
        module = method.script.removesuffix(".py").replace("/", ".")
        command = [
                sys.executable,
                "-m",
                module,
                str(input_path),
                str(checkpoint),
                *([] if resolved_target_k is None else ["--target-k", str(resolved_target_k), "--target-source", target_source]),
                *([] if not method.portfolio else [
                    "--workers", str(workers),
                    "--fallback-solver", fallback_solver,
                    "--cpu-policy", cpu_policy,
                    "--manager-core", manager_core,
                    "--portfolio-log-dir", str(selected_portfolio_log_dir),
                ]),
                "--solver",
                solver_name,
                "--timeout",
                str(timeout_s),
            ]
        process = run_supervised(
            command,
            cwd=SOURCE_ROOT,
            # Portfolio workers use the complete requested budget; reserve
            # bounded cleanup time for coordinator cancellation and checkpoint
            # publication. The grace scales with the requested worker pool.
            timeout_s=timeout_s + (min(10.0, 1.0 + 0.5 * workers) if method.portfolio else 0.0),
            stdout_path=stdout,
            stderr_path=stderr,
        )
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        if process.timed_out:
            payload = _finalize_feasible(payload, "FIXED_K_TIMEOUT" if resolved_target_k is not None else "TIMEOUT_FEASIBLE", "wall-timeout", process)
        elif process.returncode != 0:
            worker_payload = payload
            error_code = worker_payload.get("error_code")
            payload = _error(input_path, method_key, solver_name, str(worker_payload.get("error", "worker exited unsuccessfully")), process)
            if worker_payload.get("schema") == SCHEMA:
                payload["encoding"] = worker_payload.get("encoding", {})
                payload["last_checkpoint_status"] = (worker_payload.get("outcome") or {}).get("status")
            if error_code in {"ERROR_MDO_K_UNSAT", "ERROR_FIXED_K_UNSAT"}:
                payload["error_code"] = error_code
                payload["target_k"] = resolved_target_k
                payload["encoding"] = worker_payload.get("encoding", {})
            write_json_atomic(output_path, payload)
            return payload
        else:
            payload["execution"].update({"returncode": process.returncode, "timed_out": False, "termination_signal": process.termination_signal})
        verify_document(output_path, payload)
        write_json_atomic(output_path, payload)
        return payload
    except Exception as exc:
        payload = _error(input_path, method_key, solver_name, str(exc))
        write_json_atomic(output_path, payload)
        return payload
    except KeyboardInterrupt:
        if checkpoint.is_file():
            try:
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                payload = _finalize_feasible(
                    payload,
                    "INTERRUPTED_FEASIBLE",
                    "interrupted",
                    ProcessResult(-signal.SIGINT, time.perf_counter() - run_started, False, signal.SIGINT),
                )
                verify_document(output_path, payload)
                write_json_atomic(output_path, payload)
                return payload
            except Exception as exc:
                payload = _error(input_path, method_key, solver_name, str(exc))
                write_json_atomic(output_path, payload)
                return payload
        raise
    finally:
        checkpoint.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run exact L-infinity 2D-BMP methods")
    parser.add_argument("instance", type=Path)
    parser.add_argument("--method", choices=sorted(METHODS), default="triangle")
    parser.add_argument("--solver")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fallback-solver", default="cadical300")
    parser.add_argument("--cpu-policy", choices=("physical", "logical", "none"), default="physical")
    parser.add_argument("--manager-core", choices=("separate", "none"), default="separate")
    args = parser.parse_args(argv)
    solver_name = args.solver or (PORTFOLIO_DEFAULT_SOLVER if args.method == "parallel-bounds" else DEFAULT_SOLVER)
    payload = run(
        args.instance,
        method_key=args.method,
        solver_name=solver_name,
        timeout_s=args.timeout,
        output=args.output,
        workers=args.workers,
        fallback_solver=args.fallback_solver,
        cpu_policy=args.cpu_policy,
        manager_core=args.manager_core,
    )
    print(json.dumps({"output": str(args.output), "schema": payload["schema"]}, sort_keys=True))
    return 0 if payload["schema"] == SCHEMA else 2


if __name__ == "__main__":
    raise SystemExit(main())
