from __future__ import annotations

DEFAULT_SOLVER = "cadical300"
PORTFOLIO_DEFAULT_SOLVER = "kissat404"
SUPPORTED_SOLVERS = ("cadical195", "cadical300", "glucose42", "kissat404")
INCREMENTAL_SOLVERS = frozenset({"cadical195", "cadical300", "glucose42"})
ONE_SHOT_SOLVERS = frozenset(SUPPORTED_SOLVERS)
CADICAL300_PARAMETERS = {"factor": 0, "ilb": 2}
CADICAL300_ONE_SHOT_PARAMETERS = {"factor": 1, "ilb": 0}


def validate_solver_for_method(solver_name: str, *, portfolio: bool) -> None:
    if solver_name not in SUPPORTED_SOLVERS:
        raise ValueError(f"unsupported solver {solver_name!r}; choose one of: {', '.join(SUPPORTED_SOLVERS)}")
    if not portfolio and solver_name not in INCREMENTAL_SOLVERS:
        raise ValueError(f"solver {solver_name!r} is supported only by the parallel-bounds one-shot portfolio")


def create_sat_solver(solver_name: str = DEFAULT_SOLVER, *, one_shot: bool = False):
    if solver_name == "kissat404" and not one_shot:
        raise ValueError("kissat404 is available only for one-shot parallel-bounds decisions")
    if solver_name == "cadical195":
        from pysat.solvers import Cadical195

        return Cadical195()
    if solver_name == "cadical300":
        try:
            from pysat.solvers import Cadical300
        except ImportError as exc:
            raise RuntimeError("cadical300 requires python-sat>=1.9.dev3") from exc
        solver = Cadical300()
        solver.configure(CADICAL300_ONE_SHOT_PARAMETERS if one_shot else CADICAL300_PARAMETERS)
        return solver
    if solver_name == "glucose42":
        from pysat.solvers import Glucose42

        return Glucose42()
    if solver_name == "kissat404":
        try:
            from pysat.solvers import Kissat404
        except ImportError as exc:
            raise RuntimeError("kissat404 requires python-sat>=1.9.dev15") from exc
        return Kissat404()
    raise ValueError(f"unsupported solver {solver_name!r}; choose one of: {', '.join(SUPPORTED_SOLVERS)}")
