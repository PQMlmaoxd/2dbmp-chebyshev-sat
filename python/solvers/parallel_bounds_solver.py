from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import signal
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any, Callable, Mapping

from python.solvers.bound_portfolio import BoundState, choose_bound_targets, portfolio_summary
from python.solvers.fixed_k_solver import FixedKDecision, FixedKTemplate, build_fixed_k_template, decide_fixed_k
from python.solvers.incremental_bandwidth_solver_cutoff_symmetry_nodrsa import Incumbent, SolveResult
from python.encoders.symmetry_breaking import select_anchor_vertex
from python.solvers.solver_factory import PORTFOLIO_DEFAULT_SOLVER, SUPPORTED_SOLVERS
from python.utils.cpu_affinity import allowed_cpus, choose_cpu_slots
from python.utils.host_grid import HostGrid, deterministic_placement
from python.utils.lower_bound_calculator import initial_upper_bound, theoretical_lower_bound
from python.utils.mtx_parser import ParsedMtxGraph, parse_mtx_graph
from python.utils.solution_verifier import verify_placement
from python.utils.verification_io import build_verification_document, write_json_atomic


_TEMPLATE: FixedKTemplate | None = None
_SOLVE_LOCK = threading.Lock()


def _pin(cpu: int | None) -> None:
    if cpu is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})


def _set_parent_death_signal() -> None:
    if not sys_platform_linux():
        return
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL) != 0:
            return
        if os.getppid() == 1:
            os._exit(1)
    except (AttributeError, OSError, RuntimeError):
        return


def sys_platform_linux() -> bool:
    return os.name == "posix" and Path("/proc/sys/kernel").is_dir()


def _child_loop(connection: Any, slot_id: int, cpu: int | None, generation: int, inherited_parent_fd: int) -> None:
    try:
        with suppress(OSError):
            os.close(inherited_parent_fd)
        _set_parent_death_signal()
        _pin(cpu)
        connection.send(
            {"type": "ready", "slot_id": slot_id, "generation": generation, "pid": os.getpid(), "cpu": cpu}
        )
        while True:
            command = connection.recv()
            if command.get("type") == "shutdown":
                return
            if command.get("type") != "solve":
                connection.send({"type": "error", "error": "unknown worker command", "slot_id": slot_id})
                continue
            template = _TEMPLATE
            if template is None:
                raise RuntimeError("fixed-K template was not inherited by child")
            task_id = int(command["task_id"])
            decision = decide_fixed_k(
                template,
                int(command["k"]),
                str(command["solver"]),
                cpu=cpu,
                task_id=task_id,
            )
            connection.send(
                {"type": "decision", "slot_id": slot_id, "generation": generation, **_decision_payload(decision)}
            )
    except EOFError:
        return
    except BaseException as exc:
        try:
            connection.send(
                {
                    "type": "error",
                    "slot_id": slot_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        connection.close()


def _decision_payload(decision: FixedKDecision) -> dict[str, object]:
    return {
        "task_id": decision.encoding.get("task_id"),
        "k": decision.k,
        "result": decision.result,
        "solver": decision.solver,
        "elapsed_s": decision.elapsed_s,
        "encoding": dict(decision.encoding),
        "positions": None if decision.positions is None else {str(vertex): list(position) for vertex, position in decision.positions.items()},
        "bandwidth": decision.bandwidth,
        "error": decision.error,
        "solver_stats": None if decision.solver_stats is None else dict(decision.solver_stats),
    }


@dataclass
class _Task:
    task_id: int
    k: int
    slot_id: int
    generation: int
    solver: str
    dispatched_at: float
    retry: bool = False


@dataclass
class _Slot:
    slot_id: int
    cpu: int | None
    process: Any
    connection: Any
    generation: int
    started_at: float
    ready: bool = False
    stop_reason: str | None = None
    task: _Task | None = None


class ParallelBoundsSolver:
    """Coordinator for independent fixed-K SAT decisions.

    The coordinator owns all mutable proof state and publications. Child
    processes only execute one fixed-K query and return an immutable message.
    """

    def __init__(
        self,
        graph: ParsedMtxGraph,
        *,
        solver_name: str = PORTFOLIO_DEFAULT_SOLVER,
        fallback_solver: str = "cadical300",
        workers: int = 4,
        timeout_s: float = 3600.0,
        symmetry: str = "triangle",
        cpu_policy: str = "physical",
        manager_core: str = "separate",
        on_incumbent: Callable[[Incumbent], None] | None = None,
        on_progress: Callable[[Mapping[str, object]], None] | None = None,
        log_dir: str | Path | None = None,
    ) -> None:
        if solver_name not in SUPPORTED_SOLVERS:
            raise ValueError(f"unsupported primary solver {solver_name!r}")
        if fallback_solver not in SUPPORTED_SOLVERS:
            raise ValueError(f"unsupported fallback solver {fallback_solver!r}")
        if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
            raise ValueError("workers must be a positive integer")
        if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be a finite positive number")
        if symmetry != "triangle":
            raise ValueError("parallel-bounds currently requires triangle symmetry")
        self.graph = graph
        self.host = HostGrid.square(graph.n_vertices)
        self.solver_name = solver_name
        self.fallback_solver = fallback_solver
        self.requested_workers = workers
        self.timeout_s = timeout_s
        self.symmetry = symmetry
        self.cpu_policy = cpu_policy
        self.manager_core = manager_core
        self.on_incumbent = on_incumbent
        self.on_progress = on_progress
        self.log_dir = None if log_dir is None else Path(log_dir)
        self._ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
        if self._ctx is None:
            raise RuntimeError("parallel-bounds requires a POSIX fork multiprocessing context")
        self._slots: dict[int, _Slot] = {}
        self._slot_generations: dict[int, int] = {}
        self._next_task_id = 1
        self._tasks: dict[int, dict[str, object]] = {}
        self._fallback_for_k: set[int] = set()
        self._failed_k: set[int] = set()
        self._decision_order = 0
        self._event_seq = 0
        self._stdout_broken = False
        self._original_affinity: set[int] | None = None
        self._defer_checkpoint = False

    @staticmethod
    def _snapshot(variables: int, clauses: int, *, complete: bool | None = None) -> dict[str, object]:
        snapshot: dict[str, object] = {"variables": variables, "clauses": clauses}
        if complete is not None:
            snapshot["complete"] = complete
        return snapshot

    def _emit(self, event: Mapping[str, object], *, slot_id: int | None = None) -> None:
        self._event_seq += 1
        payload = {"seq": self._event_seq, "elapsed_s": time.perf_counter() - self._started, **dict(event)}
        if slot_id is not None:
            payload["slot_id"] = slot_id
        if not self._stdout_broken:
            try:
                print(json.dumps(payload, sort_keys=True), flush=True)
            except (BrokenPipeError, OSError):
                self._stdout_broken = True
        if self.log_dir is not None and slot_id is not None:
            try:
                worker_dir = self.log_dir / "workers"
                worker_dir.mkdir(parents=True, exist_ok=True)
                with (worker_dir / f"slot-{slot_id}.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(payload, sort_keys=True) + "\n")
            except OSError:
                pass

    def _publish_progress(self, encoding: dict[str, object]) -> None:
        if self.on_progress:
            self.on_progress(copy.deepcopy(encoding))

    def _write_task_artifact(self, task_id: int, record: Mapping[str, object]) -> None:
        if self.log_dir is None:
            return
        try:
            decisions = self.log_dir / "decisions"
            decisions.mkdir(parents=True, exist_ok=True)
            write_json_atomic(decisions / f"task-{task_id:06d}.json", record)
        except OSError:
            pass

    def _write_summary(self, encoding: Mapping[str, object], *, complete: bool = False) -> None:
        if self.log_dir is None:
            return
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(self.log_dir / "summary.json", {"complete": complete, "encoding": dict(encoding)})
        except OSError:
            pass

    def _new_encoding(
        self, template: FixedKTemplate, state: BoundState, *, upper_seed: int, seed_bandwidth: int
    ) -> dict[str, object]:
        initial = self._snapshot(template.variables_primary, 0)
        before = self._snapshot(template.base_variables, template.base_clauses_count)
        portfolio = {
            "strategy": "parallel-bounds",
            "lower_bound_initial": state.lower,
            "upper_bound_initial": state.upper,
            "host_upper_bound": upper_seed,
            "workers_requested": self.requested_workers,
            "workers_effective": None,
            "workers_active_peak": 0,
            "cpu_policy": self.cpu_policy,
            "manager_core": self.manager_core,
            "tasks": {},
            "lower_bound_certificate": "maximum-degree",
            "winning_solver": "deterministic-seed",
            "winning_k": seed_bandwidth,
            "status": "running",
            "bounds": dict(portfolio_summary(state)),
            "active_tasks": [],
            "failed_k": [],
            "fallback_k": [],
            "trust_model": "sat-backend-unsat",
        }
        return {
            "strategy": "parallel-bounds",
            "variables_primary": template.variables_primary,
            "base_variables": template.base_variables,
            "base_clauses": template.base_clauses_count,
            "symmetry": template.symmetry_clauses,
            **template.position_counts,
            "distance_by_k": {},
            "phases": {
                "initial": initial,
                "before_first_k": before,
                "after_first_k": before,
                "current": before,
                "final": {**before, "complete": False},
            },
            "work": {"generated_variables": template.base_variables, "generated_clauses": template.base_clauses_count},
            "portfolio": portfolio,
        }

    def _update_encoding(self, encoding: dict[str, object], state: BoundState, *, complete: bool = False) -> None:
        portfolio = encoding["portfolio"]
        assert isinstance(portfolio, dict)
        portfolio["bounds"] = dict(portfolio_summary(state))
        portfolio["active_tasks"] = sorted(
            task_id for task_id, record in self._tasks.items() if record.get("status") == "RUNNING"
        )
        portfolio["failed_k"] = sorted(self._failed_k)
        portfolio["fallback_k"] = sorted(self._fallback_for_k)
        portfolio["tasks"] = copy.deepcopy(self._tasks)
        portfolio["workers_active_peak"] = max(
            int(portfolio.get("workers_active_peak", 0)),
            sum(1 for slot in self._slots.values() if slot.task is not None),
        )
        phases = encoding["phases"]
        assert isinstance(phases, dict)
        phases["final"] = {**phases["current"], "complete": complete}
        self._publish_progress(encoding)
        self._write_summary(encoding, complete=complete)

    def _start_slot(self, slot_id: int, cpu: int | None) -> _Slot:
        generation = self._slot_generations.get(slot_id, 0) + 1
        self._slot_generations[slot_id] = generation
        if slot_id in self._slots:
            self._terminate_slot(self._slots[slot_id], "respawn")
        parent, child = self._ctx.Pipe(duplex=True)
        process = self._ctx.Process(
            target=_child_loop,
            args=(child, slot_id, cpu, generation, parent.fileno()),
            daemon=False,
        )
        process.start()
        child.close()
        slot = _Slot(slot_id, cpu, process, parent, generation, time.perf_counter(), False)
        self._slots[slot_id] = slot
        self._emit(
            {"event": "worker-spawn", "pid": process.pid, "cpu": cpu, "generation": generation},
            slot_id=slot_id,
        )
        return slot

    def _terminate_slot(self, slot: _Slot, reason: str) -> None:
        slot.stop_reason = reason
        process = slot.process
        if process.is_alive():
            with suppress(BaseException):
                self._emit(
                    {
                        "event": "cancel-requested",
                        "task_id": None if slot.task is None else slot.task.task_id,
                        "k": None if slot.task is None else slot.task.k,
                        "reason": reason,
                    },
                    slot_id=slot.slot_id,
                )
            process.terminate()
            process.join(timeout=0.25)
            if process.is_alive():
                try:
                    os.kill(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.join(timeout=1.0)
        try:
            slot.connection.close()
        except OSError:
            pass
        slot.task = None

    def _shutdown_slots(self) -> None:
        for slot in list(self._slots.values()):
            if slot.process.is_alive() and slot.task is None:
                try:
                    slot.connection.send({"type": "shutdown"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
                slot.process.join(timeout=0.5)
            if slot.process.is_alive():
                self._terminate_slot(slot, "portfolio-shutdown")
            slot.process.join(timeout=0.5)
            try:
                slot.connection.close()
            except OSError:
                pass
        self._slots.clear()

    def _dispatch(self, slot: _Slot, k: int, encoding: dict[str, object], now: float) -> bool:
        task_id = self._next_task_id
        self._next_task_id += 1
        use_fallback = k in self._fallback_for_k
        solver = self.fallback_solver if use_fallback else self.solver_name
        task = _Task(task_id, k, slot.slot_id, slot.generation, solver, now, use_fallback)
        slot.task = task
        c6 = 4 * self.graph.n_edges * max(self.host.side - k - 1, 0)
        self._tasks[task_id] = {
            "task_id": task_id,
            "slot_id": slot.slot_id,
            "cpu": slot.cpu,
            "generation": slot.generation,
            "k": k,
            "solver": solver,
            "retry": use_fallback,
            "status": "RUNNING",
            "accepted": False,
            "dispatched_at": now - self._started,
            "c6_clauses": c6,
        }
        work = encoding["work"]
        assert isinstance(work, dict)
        work["generated_variables"] = int(work["generated_variables"]) + int(encoding["base_variables"])
        work["generated_clauses"] = int(work["generated_clauses"]) + int(encoding["base_clauses"]) + c6
        distance_by_k = encoding["distance_by_k"]
        assert isinstance(distance_by_k, dict)
        distance_by_k.setdefault(k, c6)
        phases = encoding["phases"]
        assert isinstance(phases, dict)
        if phases["after_first_k"] == phases["before_first_k"]:
            phases["after_first_k"] = self._snapshot(int(encoding["base_variables"]), int(encoding["base_clauses"]) + c6)
        phases["current"] = self._snapshot(int(encoding["base_variables"]), int(encoding["base_clauses"]) + c6)
        self._emit({"event": "attempt-start", "task_id": task_id, "k": k, "solver": solver}, slot_id=slot.slot_id)
        try:
            slot.connection.send(
                {
                    "type": "solve",
                    "task_id": task_id,
                    "generation": slot.generation,
                    "k": k,
                    "solver": solver,
                }
            )
        except (BrokenPipeError, EOFError, OSError) as exc:
            record = self._tasks[task_id]
            record.update(
                {
                    "status": "ERROR",
                    "accepted": False,
                    "finished_at": time.perf_counter() - self._started,
                    "error": f"dispatch failed: {type(exc).__name__}: {exc}",
                }
            )
            self._write_task_artifact(task_id, record)
            if not task.retry and self.fallback_solver != self.solver_name:
                self._fallback_for_k.add(k)
            else:
                self._failed_k.add(k)
            slot.task = None
            self._terminate_slot(slot, "dispatch-failed")
            self._slots.pop(slot.slot_id, None)
            return False
        return True

    def _process_decision(
        self,
        *,
        slot: _Slot,
        task: _Task,
        decision: Mapping[str, object],
        state: BoundState,
        best_positions: dict[int, tuple[int, int]],
        encoding: dict[str, object],
        counts: dict[str, int],
        anchor: int,
    ) -> tuple[BoundState, dict[int, tuple[int, int]]]:
        if not slot.process.is_alive() and slot.stop_reason is None:
            slot.stop_reason = "result-drained"
        task_record = self._tasks[task.task_id]
        self._decision_order += 1
        result = decision.get("result")
        task_record.update(
            {
                "status": result if isinstance(result, str) else "ERROR",
                "finished_at": time.perf_counter() - self._started,
                "elapsed_s": decision.get("elapsed_s"),
                "encoding": decision.get("encoding", {}),
                "solver_stats": decision.get("solver_stats"),
                "error": decision.get("error"),
                "accepted": False,
                "decision_order": self._decision_order,
            }
        )
        slot.task = None
        counts["solve_count"] += 1
        k = task.k

        if result == "ERROR":
            self._write_task_artifact(task.task_id, task_record)
            self._emit(
                {
                    "event": "attempt-result",
                    "task_id": task.task_id,
                    "k": k,
                    "result": "ERROR",
                    "accepted": False,
                    "error": decision.get("error"),
                },
                slot_id=slot.slot_id,
            )
            if not task.retry and self.fallback_solver != self.solver_name:
                self._fallback_for_k.add(k)
            else:
                self._failed_k.add(k)
            self._emit(
                {"event": "worker-error", "task_id": task.task_id, "k": k, "error": decision.get("error")},
                slot_id=slot.slot_id,
            )
            return state, best_positions

        if result not in {"SAT", "UNSAT"}:
            self._write_task_artifact(task.task_id, task_record)
            raise RuntimeError(f"unknown fixed-K result {result!r}")

        positions = decision.get("positions")
        bandwidth = decision.get("bandwidth")
        if result == "SAT":
            if not isinstance(positions, dict) or not isinstance(bandwidth, int) or isinstance(bandwidth, bool):
                self._write_task_artifact(task.task_id, task_record)
                raise RuntimeError("SAT result lacks a witness")
            try:
                normalized = {
                    int(vertex): (int(value[0]), int(value[1]))
                    for vertex, value in positions.items()
                }
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                self._write_task_artifact(task.task_id, task_record)
                raise RuntimeError(f"SAT result has malformed positions: {exc}") from exc
            verified = verify_placement(
                self.graph.n_vertices,
                self.graph.edges,
                self.host,
                normalized,
                tested_k=k,
                anchor=anchor,
                symmetry=self.symmetry,
            )
            if not verified.valid or verified.bandwidth != bandwidth:
                self._write_task_artifact(task.task_id, task_record)
                raise RuntimeError(f"coordinator rejected SAT witness: {verified.error}")
            counts["sat_count"] += 1
            task_record["bandwidth"] = bandwidth
            current = state.lower <= k < state.upper and k not in state.tested
            if k < state.lower:
                task_record["stale_reason"] = "contradicts-lower-bound"
                self._write_task_artifact(task.task_id, task_record)
                raise RuntimeError(f"SAT result at K={k} contradicts lower bound {state.lower}")
            if current:
                next_state = state.apply(k=k, result="SAT", bandwidth=bandwidth)
                accepted = True
            elif bandwidth < state.upper:
                next_state = state.record_sat_witness(k=k, bandwidth=bandwidth)
                task_record["accepted_reason"] = "better-stale-witness"
                accepted = True
            else:
                next_state = state
                task_record["stale_reason"] = "bounds-dominated"
                accepted = False
            task_record["accepted"] = accepted
            self._write_task_artifact(task.task_id, task_record)
            event: dict[str, object] = {
                "event": "attempt-result",
                "task_id": task.task_id,
                "k": k,
                "result": "SAT",
                "bandwidth": bandwidth,
                "accepted": accepted,
            }
            if not accepted:
                event["stale"] = True
            else:
                event.update({"lower_bound": next_state.lower, "upper_bound": next_state.upper})
            self._emit(event, slot_id=slot.slot_id)
            if accepted and bandwidth < state.upper:
                portfolio = encoding["portfolio"]
                assert isinstance(portfolio, dict)
                portfolio["winning_solver"] = task.solver
                portfolio["winning_k"] = k
                best_positions = normalized
                self._emit(
                    {
                        "event": "bound-update",
                        "kind": "SAT",
                        "k": k,
                        "bandwidth": bandwidth,
                        "lower_bound": next_state.lower,
                        "upper_bound": next_state.upper,
                    },
                    slot_id=slot.slot_id,
                )
                if self.on_incumbent:
                    self._defer_checkpoint = True
                    try:
                        self._update_encoding(encoding, next_state)
                    finally:
                        self._defer_checkpoint = False
                    self.on_incumbent(
                        Incumbent(
                            k,
                            bandwidth,
                            dict(normalized),
                            time.perf_counter() - self._started,
                            counts["solve_count"],
                        )
                    )
                else:
                    self._update_encoding(encoding, next_state)
            return next_state, best_positions

        counts["unsat_count"] += 1
        current = state.lower <= k < state.upper and k not in state.tested
        if k >= state.upper:
            task_record["stale_reason"] = "contradicts-upper-witness"
            self._write_task_artifact(task.task_id, task_record)
            raise RuntimeError(f"UNSAT result at K={k} contradicts upper bound {state.upper}")
        if current:
            next_state = state.apply(k=k, result="UNSAT")
            accepted = True
        else:
            next_state = state
            task_record["stale_reason"] = "bounds-dominated"
            accepted = False
        task_record["accepted"] = accepted
        self._write_task_artifact(task.task_id, task_record)
        event = {
            "event": "attempt-result",
            "task_id": task.task_id,
            "k": k,
            "result": "UNSAT",
            "accepted": accepted,
        }
        if not accepted:
            event["stale"] = True
        else:
            event.update({"lower_bound": next_state.lower, "upper_bound": next_state.upper})
        self._emit(event, slot_id=slot.slot_id)
        if accepted:
            self._emit(
                {
                    "event": "bound-update",
                    "kind": "UNSAT",
                    "k": k,
                    "lower_bound": next_state.lower,
                    "upper_bound": next_state.upper,
                },
                slot_id=slot.slot_id,
            )
        return next_state, best_positions

    def _drain_ready_decisions(
        self,
        *,
        state: BoundState,
        best_positions: dict[int, tuple[int, int]],
        encoding: dict[str, object],
        counts: dict[str, int],
        anchor: int,
    ) -> tuple[BoundState, dict[int, tuple[int, int]]]:
        for slot in list(self._slots.values()):
            while True:
                try:
                    if not slot.connection.poll():
                        break
                    message = slot.connection.recv()
                except (EOFError, OSError):
                    break
                if message.get("type") == "ready":
                    if message.get("generation") == slot.generation:
                        slot.ready = True
                    continue
                if message.get("type") != "decision":
                    continue
                task = slot.task
                if (
                    task is None
                    or task.task_id != message.get("task_id")
                    or task.generation != message.get("generation")
                ):
                    continue
                state, best_positions = self._process_decision(
                    slot=slot,
                    task=task,
                    decision=message,
                    state=state,
                    best_positions=best_positions,
                    encoding=encoding,
                    counts=counts,
                    anchor=anchor,
                )
                self._update_encoding(encoding, state)
        return state, best_positions

    def _reap_dead_slots(self, encoding: dict[str, object], now: float) -> None:
        for slot in list(self._slots.values()):
            if slot.process.is_alive() and not slot.ready and now - slot.started_at > 5.0:
                self._terminate_slot(slot, "worker-ready-timeout")
                self._slots.pop(slot.slot_id, None)
                continue
            if slot.process.is_alive():
                continue
            try:
                if slot.connection.poll():
                    continue
            except (EOFError, OSError):
                pass
            task = slot.task
            if task is not None:
                record = self._tasks[task.task_id]
                record.update({"status": "ERROR", "finished_at": now - self._started, "error": "worker exited without a result"})
                record["accepted"] = False
                self._write_task_artifact(task.task_id, record)
                if not task.retry:
                    self._fallback_for_k.add(task.k)
                else:
                    self._failed_k.add(task.k)
                slot.task = None
            slot.process.join()
            try:
                slot.connection.close()
            except OSError:
                pass
            del self._slots[slot.slot_id]
            if slot.stop_reason is None:
                self._emit({"event": "worker-error", "error": "worker exited without a result"}, slot_id=slot.slot_id)

    def _cancel_dominated(self, state: BoundState, encoding: dict[str, object]) -> None:
        for slot in list(self._slots.values()):
            task = slot.task
            if task is not None and (task.k < state.lower or task.k >= state.upper):
                with suppress(EOFError, OSError):
                    if slot.connection.poll():
                        continue
                self._tasks[task.task_id].update({"status": "CANCELLED", "cancel_reason": "bounds-dominated", "accepted": False})
                self._write_task_artifact(task.task_id, self._tasks[task.task_id])
                self._terminate_slot(slot, "bounds-dominated")

    def _cancel_active_tasks(self, reason: str) -> None:
        for slot in list(self._slots.values()):
            task = slot.task
            if task is None:
                continue
            record = self._tasks[task.task_id]
            record.update({"status": "CANCELLED", "cancel_reason": reason, "finished_at": time.perf_counter() - self._started, "accepted": False})
            self._write_task_artifact(task.task_id, record)
            self._terminate_slot(slot, reason)

    def _force_cleanup(self) -> None:
        with suppress(BaseException):
            self._cancel_active_tasks("portfolio-aborted")
        with suppress(BaseException):
            self._shutdown_slots()
        global _TEMPLATE
        _TEMPLATE = None
        if self._original_affinity is not None and hasattr(os, "sched_setaffinity"):
            with suppress(OSError):
                os.sched_setaffinity(0, self._original_affinity)

    def _available_slots(self, desired: int, cpus: tuple[int | None, ...]) -> list[_Slot]:
        slots = [
            slot
            for slot in self._slots.values()
            if slot.task is None and slot.ready and slot.process.is_alive()
        ]
        next_id = 0
        while len(self._slots) < desired:
            while next_id in self._slots:
                next_id += 1
            self._start_slot(next_id, cpus[next_id] if next_id < len(cpus) else None)
            next_id += 1
        return slots

    def _retire_idle_slots(self, desired: int) -> None:
        live = [slot for slot in self._slots.values() if slot.process.is_alive()]
        idle = [slot for slot in live if slot.task is None]
        for slot in sorted(idle, key=lambda item: item.slot_id, reverse=True):
            if len(live) <= desired:
                break
            self._terminate_slot(slot, "worker-count-reduced")
            self._slots.pop(slot.slot_id, None)
            live.remove(slot)

    def _closed_certificate(self, state: BoundState) -> tuple[str, str, int]:
        if state.upper == 0:
            return "OPTIMAL_BY_BOUND", "zero-edge-graph", state.upper
        if state.upper == 1:
            return "OPTIMAL_BY_BOUND", "minimum-positive-bandwidth", state.upper
        if state.lower == theoretical_lower_bound(self.graph) and state.lower == state.upper:
            return "OPTIMAL_BY_THEORETICAL_BOUND", "theoretical-lower-bound", state.upper
        if state.unsat_boundary == state.upper - 1:
            return "OPTIMAL_BY_UNSAT", "unsat-below-verified-incumbent", state.upper - 1
        raise RuntimeError("closed bounds lack a valid optimality certificate")

    def _final_result(
        self,
        *,
        state: BoundState,
        positions: Mapping[int, tuple[int, int]],
        started: float,
        upper_seed: int,
        counts: Mapping[str, int],
        encoding: dict[str, object],
        status: str,
        termination: str,
        termination_k: int | None,
    ) -> SolveResult:
        portfolio = encoding["portfolio"]
        assert isinstance(portfolio, dict)
        portfolio["status"] = status
        portfolio["finished_at_s"] = time.perf_counter() - started
        self._cancel_active_tasks("portfolio-complete")
        self._defer_checkpoint = True
        try:
            self._update_encoding(encoding, state, complete=True)
        finally:
            self._defer_checkpoint = False
        return SolveResult(
            state.upper,
            dict(positions),
            status,
            termination,
            termination_k,
            upper_seed,
            self.solver_name,
            self.symmetry,
            select_anchor_vertex(self.graph),
            time.perf_counter() - started,
            counts["solve_count"],
            counts["sat_count"],
            counts["unsat_count"],
            copy.deepcopy(encoding),
        )

    def _timeout_result(
        self,
        *,
        state: BoundState,
        positions: Mapping[int, tuple[int, int]],
        started: float,
        upper_seed: int,
        counts: Mapping[str, int],
        encoding: dict[str, object],
    ) -> SolveResult:
        portfolio = encoding["portfolio"]
        assert isinstance(portfolio, dict)
        portfolio["status"] = "TIMEOUT_FEASIBLE"
        self._cancel_active_tasks("portfolio-timeout")
        self._update_encoding(encoding, state, complete=False)
        return SolveResult(
            state.upper,
            dict(positions),
            "TIMEOUT_FEASIBLE",
            "portfolio-soft-timeout",
            None,
            upper_seed,
            self.solver_name,
            self.symmetry,
            select_anchor_vertex(self.graph),
            time.perf_counter() - started,
            counts["solve_count"],
            counts["sat_count"],
            counts["unsat_count"],
            copy.deepcopy(encoding),
        )

    def solve(self) -> SolveResult:
        if not _SOLVE_LOCK.acquire(blocking=False):
            raise RuntimeError("parallel-bounds does not support concurrent solves in one process")
        try:
            return self._solve_locked()
        finally:
            self._force_cleanup()
            _SOLVE_LOCK.release()

    def _solve_locked(self) -> SolveResult:
        global _TEMPLATE
        self._started = time.perf_counter()
        original_affinity = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        self._original_affinity = original_affinity
        template = build_fixed_k_template(self.graph, symmetry=self.symmetry)
        _TEMPLATE = template
        anchor = template.anchor
        upper_seed = initial_upper_bound(self.graph.n_vertices)
        seed_positions = deterministic_placement(self.host, anchor=anchor)
        seed_verified = verify_placement(
            self.graph.n_vertices,
            self.graph.edges,
            self.host,
            seed_positions,
            tested_k=upper_seed,
            anchor=anchor,
            symmetry=self.symmetry,
        )
        if not seed_verified.valid or seed_verified.bandwidth is None:
            raise RuntimeError(f"deterministic seed witness is invalid: {seed_verified.error}")
        state = BoundState(theoretical_lower_bound(self.graph), seed_verified.bandwidth)
        encoding = self._new_encoding(template, state, upper_seed=upper_seed, seed_bandwidth=seed_verified.bandwidth)
        counts = {"solve_count": 0, "sat_count": 0, "unsat_count": 0}
        best_positions = dict(seed_positions)
        self._publish_progress(encoding)
        if self.on_incumbent:
            self.on_incumbent(Incumbent(upper_seed, seed_verified.bandwidth, best_positions, 0.0, 0))
        self._emit(
            {
                "event": "portfolio-start",
                "lower_bound": state.lower,
                "upper_bound": state.upper,
                "host_upper_bound": upper_seed,
                "workers": self.requested_workers,
                "solver": self.solver_name,
                "fallback_solver": self.fallback_solver,
            }
        )
        manager_core: int | None = None
        try:
            try:
                cpu_count = len(allowed_cpus())
                reserved = 1 if self.manager_core == "separate" and self.cpu_policy != "none" else 0
                effective_workers = min(self.requested_workers, max(1, cpu_count - reserved))
                cpus, manager_core = choose_cpu_slots(effective_workers, policy=self.cpu_policy, manager_core=self.manager_core)
            except ValueError:
                effective_workers = min(self.requested_workers, max(1, len(allowed_cpus())))
                cpus, manager_core = choose_cpu_slots(effective_workers, policy="none", manager_core="none")
                encoding["portfolio"]["cpu_fallback"] = "none"
            encoding["portfolio"]["workers_effective"] = effective_workers
            if manager_core is not None and hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, {manager_core})
            if state.closed:
                status = "OPTIMAL_BY_BOUND" if state.upper in {0, 1} else "OPTIMAL_BY_THEORETICAL_BOUND"
                termination = "minimum-positive-bandwidth" if state.upper in {0, 1} else "theoretical-lower-bound"
                return self._final_result(
                    state=state,
                    positions=best_positions,
                    started=self._started,
                    upper_seed=upper_seed,
                    counts=counts,
                    encoding=encoding,
                    status=status,
                    termination=termination,
                    termination_k=state.upper,
                )
            deadline = self._started + self.timeout_s
            while not state.closed:
                now = time.perf_counter()
                self._reap_dead_slots(encoding, now)
                self._cancel_dominated(state, encoding)
                desired = min(effective_workers, max(1, state.upper - state.lower))
                self._retire_idle_slots(desired)
                planning = BoundState(state.lower, state.upper, state.tested | frozenset(task.k for slot in self._slots.values() if slot.task for task in [slot.task]))
                targets = list(choose_bound_targets(planning, desired, exclude=frozenset(self._failed_k)))
                free_slots = self._available_slots(desired, cpus)
                active_k = {slot.task.k for slot in self._slots.values() if slot.task is not None}
                for k in targets:
                    if k in active_k:
                        continue
                    idle = next((slot for slot in free_slots if slot.task is None and slot.process.is_alive()), None)
                    if idle is None:
                        break
                    if self._dispatch(idle, k, encoding, now):
                        active_k.add(k)
                self._update_encoding(encoding, state)
                if time.perf_counter() >= deadline:
                    state, best_positions = self._drain_ready_decisions(
                        state=state,
                        best_positions=best_positions,
                        encoding=encoding,
                        counts=counts,
                        anchor=anchor,
                    )
                    if state.closed:
                        status, termination, termination_k = self._closed_certificate(state)
                        return self._final_result(
                            state=state,
                            positions=best_positions,
                            started=self._started,
                            upper_seed=upper_seed,
                            counts=counts,
                            encoding=encoding,
                            status=status,
                            termination=termination,
                            termination_k=termination_k,
                        )
                    self._emit({"event": "portfolio-timeout", "lower_bound": state.lower, "upper_bound": state.upper})
                    return self._timeout_result(
                        state=state,
                        positions=best_positions,
                        started=self._started,
                        upper_seed=upper_seed,
                        counts=counts,
                        encoding=encoding,
                    )
                readers = []
                for slot in self._slots.values():
                    if slot.process.is_alive():
                        if slot.task is not None or not slot.ready:
                            readers.append(slot.connection)
                        continue
                    with suppress(EOFError, OSError):
                        if slot.connection.poll():
                            readers.append(slot.connection)
                if not readers:
                    if not targets:
                        unresolved = {
                            k for k in range(state.lower, state.upper) if k not in state.tested
                        }
                        if not unresolved or unresolved.issubset(self._failed_k):
                            encoding["portfolio"]["status"] = "ERROR"
                            self._update_encoding(encoding, state)
                            raise RuntimeError("all unresolved fixed-K targets failed on primary and fallback solvers")
                        time.sleep(0.01)
                        continue
                    time.sleep(0.001)
                    continue
                ready = wait(readers, timeout=min(0.1, max(0.0, deadline - time.perf_counter())))
                for connection in ready:
                    try:
                        message = connection.recv()
                    except (EOFError, OSError):
                        slot = next((candidate for candidate in self._slots.values() if candidate.connection is connection), None)
                        if slot is not None and slot.process.is_alive():
                            task = slot.task
                            if task is not None:
                                record = self._tasks[task.task_id]
                                record.update(
                                    {
                                        "status": "ERROR",
                                        "finished_at": time.perf_counter() - self._started,
                                        "error": "worker IPC closed before a result",
                                    }
                                )
                                self._write_task_artifact(task.task_id, record)
                                if not task.retry and self.fallback_solver != self.solver_name:
                                    self._fallback_for_k.add(task.k)
                                else:
                                    self._failed_k.add(task.k)
                            self._terminate_slot(slot, "ipc-closed")
                            self._slots.pop(slot.slot_id, None)
                        continue
                    slot = next((candidate for candidate in self._slots.values() if candidate.connection is connection), None)
                    if slot is None:
                        continue
                    if message.get("type") == "ready":
                        if message.get("generation") != slot.generation:
                            self._emit(
                                {"event": "stale-ready", "generation": message.get("generation")},
                                slot_id=slot.slot_id,
                            )
                            continue
                        slot.ready = True
                        self._emit({"event": "worker-ready", "pid": message.get("pid"), "cpu": message.get("cpu")}, slot_id=slot.slot_id)
                        continue
                    if message.get("type") != "decision":
                        self._emit({"event": "worker-error", "error": message.get("error", "unknown child message")}, slot_id=slot.slot_id)
                        continue
                    task = slot.task
                    if (
                        task is None
                        or task.task_id != message.get("task_id")
                        or task.generation != message.get("generation")
                    ):
                        self._emit({"event": "stale-result", "task_id": message.get("task_id")}, slot_id=slot.slot_id)
                        continue
                    state, best_positions = self._process_decision(
                        slot=slot,
                        task=task,
                        decision=message,
                        state=state,
                        best_positions=best_positions,
                        encoding=encoding,
                        counts=counts,
                        anchor=anchor,
                    )
                    self._update_encoding(encoding, state)
            state, best_positions = self._drain_ready_decisions(
                state=state,
                best_positions=best_positions,
                encoding=encoding,
                counts=counts,
                anchor=anchor,
            )
            status, termination, termination_k = self._closed_certificate(state)
            return self._final_result(
                state=state,
                positions=best_positions,
                started=self._started,
                upper_seed=upper_seed,
                counts=counts,
                encoding=encoding,
                status=status,
                termination=termination,
                termination_k=termination_k,
            )
        finally:
            self._force_cleanup()


def run_parallel_worker(
    instance: str | Path,
    checkpoint: str | Path,
    *,
    solver_name: str,
    fallback_solver: str,
    timeout_s: float,
    workers: int,
    cpu_policy: str,
    manager_core: str,
    method_id: str,
    solver_script: str,
    log_dir: str | Path | None = None,
) -> dict[str, object]:
    input_path = Path(instance).resolve()
    checkpoint_path = Path(checkpoint)
    graph = parse_mtx_graph(input_path)
    host = HostGrid.square(graph.n_vertices)
    anchor = select_anchor_vertex(graph)
    latest_encoding: dict[str, object] = {}

    def publish_progress(encoding: Mapping[str, object]) -> None:
        nonlocal latest_encoding
        latest_encoding = copy.deepcopy(dict(encoding))
        if checkpoint_path.is_file() and not solver._defer_checkpoint:
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            payload["encoding"] = latest_encoding
            write_json_atomic(checkpoint_path, payload)

    def publish_incumbent(incumbent: Incumbent) -> None:
        write_json_atomic(
            checkpoint_path,
            build_verification_document(
                instance_file=input_path,
                graph=graph,
                host=host,
                method_id=method_id,
                solver=solver_name,
                solver_script=solver_script,
                symmetry="triangle",
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

    solver = ParallelBoundsSolver(
        graph,
        solver_name=solver_name,
        fallback_solver=fallback_solver,
        workers=workers,
        timeout_s=timeout_s,
        cpu_policy=cpu_policy,
        manager_core=manager_core,
        on_incumbent=publish_incumbent,
        on_progress=publish_progress,
        log_dir=log_dir,
    )
    result = solver.solve()
    write_json_atomic(
        checkpoint_path,
        build_verification_document(
            instance_file=input_path,
            graph=graph,
            host=host,
            method_id=method_id,
            solver=solver_name,
            solver_script=solver_script,
            symmetry=result.symmetry,
            anchor=result.anchor,
            status=result.status,
            termination=result.termination,
            termination_k=result.termination_k,
            elapsed_s=result.elapsed_s,
            timeout_s=timeout_s,
            initial_upper_bound=result.initial_upper_bound,
            certified_lower=result.optimal_k if result.status.startswith("OPTIMAL") else None,
            final_verified=result.optimal_k,
            optimal=result.optimal_k if result.status.startswith("OPTIMAL") else None,
            constraint_k=result.optimal_k,
            positions=dict(result.positions),
            execution={"solve_count": result.solve_count, "sat_count": result.sat_count, "unsat_count": result.unsat_count},
            encoding=result.encoding,
        ),
    )
    return {"status": result.status, "optimal_k": result.optimal_k}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parallel fixed-K exact L-infinity worker")
    parser.add_argument("instance", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--solver", default=PORTFOLIO_DEFAULT_SOLVER)
    parser.add_argument("--fallback-solver", default="cadical300")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-policy", choices=("physical", "logical", "none"), default="physical")
    parser.add_argument("--manager-core", choices=("separate", "none"), default="separate")
    parser.add_argument("--portfolio-log-dir", type=Path)
    parser.add_argument("--timeout", type=float, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run_parallel_worker(
                args.instance,
                args.checkpoint,
                solver_name=args.solver,
                fallback_solver=args.fallback_solver,
                timeout_s=args.timeout,
                workers=args.workers,
                cpu_policy=args.cpu_policy,
                manager_core=args.manager_core,
                method_id="exact-linf-parallel-bounds",
                solver_script="python/solvers/parallel_bounds_solver.py",
                log_dir=args.portfolio_log_dir,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
