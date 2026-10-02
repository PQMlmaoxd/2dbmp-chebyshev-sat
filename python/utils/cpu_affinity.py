from __future__ import annotations

import os
from pathlib import Path


def _topology_value(cpu: int, name: str) -> str | None:
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology") / name
    try:
        return path.read_text(encoding="ascii").strip()
    except OSError:
        return None


def allowed_cpus() -> tuple[int, ...]:
    if hasattr(os, "sched_getaffinity"):
        return tuple(sorted(os.sched_getaffinity(0)))
    return tuple(range(os.cpu_count() or 1))


def physical_core_representatives(cpus: tuple[int, ...] | None = None) -> tuple[int, ...]:
    available = cpus if cpus is not None else allowed_cpus()
    groups: dict[tuple[str, str], list[int]] = {}
    for cpu in available:
        package = _topology_value(cpu, "physical_package_id") or "0"
        core = _topology_value(cpu, "core_id") or str(cpu)
        groups.setdefault((package, core), []).append(cpu)
    return tuple(min(values) for _, values in sorted(groups.items()))


def choose_cpu_slots(
    worker_count: int,
    *,
    policy: str = "physical",
    manager_core: str = "separate",
) -> tuple[tuple[int | None, ...], int | None]:
    if not isinstance(worker_count, int) or isinstance(worker_count, bool) or worker_count <= 0:
        raise ValueError("worker_count must be a positive integer")
    if policy not in {"physical", "logical", "none"}:
        raise ValueError("cpu policy must be physical, logical, or none")
    if manager_core not in {"separate", "none"}:
        raise ValueError("manager_core must be separate or none")
    if policy == "none":
        return (None,) * worker_count, None
    cpus = allowed_cpus()
    candidates = physical_core_representatives(cpus) if policy == "physical" else cpus
    manager = candidates[0] if manager_core == "separate" and candidates else None
    worker_candidates = tuple(cpu for cpu in candidates if cpu != manager)
    if len(worker_candidates) < worker_count:
        worker_candidates = tuple(cpu for cpu in cpus if cpu != manager)
    if len(worker_candidates) < worker_count:
        raise ValueError(f"not enough CPUs for {worker_count} workers and manager")
    return worker_candidates[:worker_count], manager
