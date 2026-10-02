# 2D-BMP under the Chebyshev norm

This repository contains a small exact SAT solver for the two-dimensional
bandwidth minimization problem on a square host grid. It exposes two methods:
`triangle` and the POSIX-only `parallel-bounds` portfolio.

## Objective

For a graph `G = (V, E)`, let `h = ceil(sqrt(|V|))`. The solver seeks an
injective placement `p(v) = (x_v, y_v)` in the `h x h` grid that minimizes

```text
min over injective p: V -> {1,...,h}^2 of
  max over (u,v) in E of max(abs(x_u - x_v), abs(y_u - y_v)).
```

This is the Chebyshev (`L-infinity`) bandwidth objective. Vertices are numbered
from 1 through `|V|`.

## Input format

Inputs must be square Matrix Market coordinate matrices. Each nonzero
off-diagonal entry represents an undirected edge; diagonal entries are ignored.
The parser accepts the Matrix Market `pattern`, `integer`, `real`, and `complex`
fields and the supported Matrix Market symmetry declarations.

## Environment

Use Linux or another POSIX environment with `fork`. The supervised runner uses
POSIX process groups, and `parallel-bounds` requires a POSIX fork context; this
repository is not a native-Windows target. Python 3.10 or newer is required.

Install the locked development environment with:

```bash
uv sync --extra dev --locked
```

Run the smoke test with:

```bash
uv run pytest
```

## Solve

From the repository root, run the triangle method with CaDiCaL 3.0.0:

```bash
uv run python -m python.solve graph.mtx --method triangle --solver cadical300 --timeout 30 --output results/triangle.json
```

Run the parallel-bounds method with Kissat 4.0.4:

```bash
uv run python -m python.solve graph.mtx --method parallel-bounds --solver kissat404 --fallback-solver cadical300 --workers 2 --cpu-policy none --manager-core none --timeout 30 --output results/parallel-bounds.json
```

Verify an emitted result with:

```bash
uv run python -m python.utils.verification_artifacts verify --artifact results/triangle.json
```

The verifier checks the input hash, graph metadata, method metadata, bounds,
and placement witness. Generated results and logs are not part of this source
repository.
