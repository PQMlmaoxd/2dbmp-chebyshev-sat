from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

from python.utils.host_grid import HostGrid
from python.utils.mtx_parser import ParsedMtxGraph

SCHEMA = "2dbmp-linf-verification-v3"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def build_verification_document(
    *,
    instance_file: str | Path,
    graph: ParsedMtxGraph,
    host: HostGrid,
    method_id: str,
    solver: str,
    solver_script: str,
    symmetry: str,
    anchor: int,
    status: str,
    termination: str,
    termination_k: int | None,
    elapsed_s: float | None,
    timeout_s: float,
    initial_upper_bound: int,
    certified_lower: int | None,
    final_verified: int,
    optimal: int | None,
    constraint_k: int,
    positions: Mapping[int, tuple[int, int]],
    execution: Mapping[str, Any] | None = None,
    encoding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "instance": {
            "file": str(Path(instance_file).resolve()),
            "sha256": sha256_file(instance_file),
            "vertices": graph.n_vertices,
            "graph_edges": graph.n_edges,
            "graph_interpretation": graph.interpretation,
            "matrix": {
                "rows": graph.rows,
                "cols": graph.cols,
                "declared_entries": graph.declared_entries,
                "field": graph.field,
                "symmetry": graph.symmetry,
            },
        },
        "method": {
            "id": method_id,
            "solver": solver,
            "solver_script": solver_script,
            "grid": {"rows": host.side, "cols": host.side, "policy": host.policy},
            "symmetry": symmetry,
            "anchor": anchor,
        },
        "outcome": {
            "status": status,
            "termination": termination,
            "termination_k": termination_k,
            "elapsed_s": elapsed_s,
            "timeout_s": timeout_s,
        },
        "bounds": {
            "initial_upper_bound": initial_upper_bound,
            "certified_lower": certified_lower,
            "final_verified": final_verified,
            "optimal": optimal,
        },
        "witness": {
            "valid": True,
            "constraint_k": constraint_k,
            "actual_bandwidth": final_verified,
            "positions": {str(vertex): [row, col] for vertex, (row, col) in sorted(positions.items())},
        },
        "execution": dict(execution or {}),
        "encoding": dict(encoding or {}),
    }
