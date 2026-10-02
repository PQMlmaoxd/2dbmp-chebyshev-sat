from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MethodEntrypoint:
    key: str
    script: str
    method_id: str
    symmetry: str
    fixed_k_mdo: bool = False
    portfolio: bool = False


METHODS = {
    "triangle": MethodEntrypoint(
        "triangle",
        "python/solvers/incremental_bandwidth_solver_cutoff_symmetry_nodrsa.py",
        "exact-linf-triangle",
        "triangle",
    ),
    "parallel-bounds": MethodEntrypoint(
        "parallel-bounds",
        "python/solvers/parallel_bounds_solver.py",
        "exact-linf-parallel-bounds",
        "triangle",
        portfolio=True,
    ),
}


def get_method(key: str) -> MethodEntrypoint:
    try:
        return METHODS[key]
    except KeyError as exc:
        raise ValueError(f"unknown method {key!r}; choose one of: {', '.join(sorted(METHODS))}") from exc


def worker_path(root: Path, method: MethodEntrypoint) -> Path:
    path = root / method.script
    if not path.is_file():
        raise FileNotFoundError(f"worker script is missing: {path}")
    return path
