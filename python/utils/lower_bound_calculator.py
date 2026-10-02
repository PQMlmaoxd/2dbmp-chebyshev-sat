from __future__ import annotations

from .host_grid import HostGrid
from .mtx_parser import ParsedMtxGraph


def initial_upper_bound(n_vertices: int) -> int:
    return HostGrid.square(n_vertices).diameter


def paper_start_bound(n_vertices: int) -> int:
    return HostGrid.square(n_vertices).side


def lower_bound(has_edges: bool) -> int:
    return 1 if has_edges else 0


def degree_lower_bound(graph: ParsedMtxGraph) -> int:
    """Return the paper's grid-neighborhood lower bound.

    A vertex at L-infinity distance at most K has at most 8*K neighboring
    cells. Summing the rings gives 4*K*(K+1) available neighbor cells.
    """

    maximum_degree = max((len(neighbors) for neighbors in graph.adjacency().values()), default=0)
    bound = 0
    covered = 0
    while covered < maximum_degree:
        bound += 1
        covered += 8 * bound
    return bound


def theoretical_lower_bound(graph: ParsedMtxGraph) -> int:
    if not graph.edges:
        return 0
    return max(1, degree_lower_bound(graph))
