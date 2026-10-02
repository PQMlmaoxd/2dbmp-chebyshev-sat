from pathlib import Path

from python.solve import run
from python.solvers.entrypoints import METHODS, worker_path
from python.utils.verification_artifacts import TERMINAL_STATUSES, verify_artifact
from python.utils.verification_io import SCHEMA


def test_public_methods_run_and_verify(tmp_path: Path) -> None:
    assert tuple(METHODS) == ("triangle", "parallel-bounds")
    root = Path(__file__).resolve().parents[2]
    for method in METHODS.values():
        assert worker_path(root, method).is_file()

    instance = tmp_path / "path5.mtx"
    instance.write_text(
        "\n".join(
            (
                "%%MatrixMarket matrix coordinate pattern symmetric",
                "% path graph with five vertices",
                "5 5 4",
                "1 2",
                "2 3",
                "3 4",
                "4 5",
                "",
            )
        ),
        encoding="utf-8",
    )
    for method, solver in (("triangle", "cadical300"), ("parallel-bounds", "kissat404")):
        output = tmp_path / f"{method}.json"
        payload = run(
            instance,
            method_key=method,
            solver_name=solver,
            timeout_s=30,
            output=output,
            workers=2,
            fallback_solver="cadical300",
            cpu_policy="none",
            manager_core="none",
        )
        assert payload["schema"] == SCHEMA
        assert verify_artifact(output) in TERMINAL_STATUSES
        assert payload["bounds"]["optimal"] == payload["bounds"]["final_verified"] == 1
        assert payload["method"]["solver"] == solver
        assert payload["execution"]["solve_count"] >= 1
        if method == "parallel-bounds":
            portfolio = payload["encoding"]["portfolio"]
            assert portfolio["upper_bound_initial"] > portfolio["lower_bound_initial"]
            assert portfolio["bounds"]["tested_k"]
            assert any(task["accepted"] for task in portfolio["tasks"].values())
