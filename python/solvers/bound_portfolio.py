from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True, slots=True)
class BoundState:
    lower: int
    upper: int
    tested: frozenset[int] = field(default_factory=frozenset)
    unsat_boundary: int | None = None

    def __post_init__(self) -> None:
        if self.lower < 0 or self.upper < self.lower:
            raise ValueError("invalid bound interval")

    @property
    def decisive(self) -> bool:
        return self.upper - self.lower == 1

    @property
    def closed(self) -> bool:
        return self.lower == self.upper

    def apply(self, *, k: int, result: str, bandwidth: int | None = None) -> "BoundState":
        if not isinstance(k, int) or isinstance(k, bool):
            raise ValueError("bound K must be an integer")
        if k in self.tested:
            raise ValueError(f"bound K={k} was already processed")
        if not self.lower <= k < self.upper:
            raise ValueError(f"bound K={k} is outside [{self.lower}, {self.upper})")
        tested = self.tested | {k}
        if result == "UNSAT":
            new_lower = max(self.lower, k + 1)
            boundary = k if new_lower > self.lower else self.unsat_boundary
            return BoundState(new_lower, self.upper, frozenset(tested), boundary)
        if result == "SAT":
            return self.record_sat_witness(k=k, bandwidth=bandwidth, require_query_in_interval=True)
        raise ValueError(f"unknown decision result {result!r}")


    def record_sat_witness(
        self, *, k: int, bandwidth: int | None, require_query_in_interval: bool = False
    ) -> "BoundState":
        if not isinstance(k, int) or isinstance(k, bool):
            raise ValueError("bound K must be an integer")
        if k in self.tested:
            raise ValueError(f"bound K={k} was already processed")
        if require_query_in_interval and not self.lower <= k < self.upper:
            raise ValueError(f"bound K={k} is outside [{self.lower}, {self.upper})")
        if not isinstance(bandwidth, int) or isinstance(bandwidth, bool):
            raise ValueError("SAT result requires an integer bandwidth")
        if bandwidth < self.lower or bandwidth > k:
            raise ValueError("SAT witness bandwidth contradicts current bounds")
        return BoundState(self.lower, min(self.upper, bandwidth), frozenset(self.tested | {k}), self.unsat_boundary)


def choose_bound_targets(
    state: BoundState, limit: int, *, exclude: frozenset[int] = frozenset()
) -> tuple[int, ...]:
    """Choose distinct K values from the unresolved interval.

    The endpoints are valuable because they immediately improve one side of
    the interval. Remaining slots split the largest uncovered integer gap.
    """

    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("target limit must be a positive integer")
    if state.closed:
        return ()
    available = [k for k in range(state.lower, state.upper) if k not in state.tested and k not in exclude]
    if not available:
        return ()
    selected: list[int] = []

    def add(k: int) -> None:
        if k in available and k not in selected and len(selected) < limit:
            selected.append(k)

    add(state.lower)
    add(state.upper - 1)
    while len(selected) < limit:
        candidates = [k for k in available if k not in selected]
        if not candidates:
            break
        boundaries = sorted({state.lower - 1, state.upper, *selected})
        gaps: list[tuple[int, int, int]] = []
        for left, right in zip(boundaries, boundaries[1:]):
            gap = [k for k in candidates if left < k < right]
            if gap:
                midpoint = gap[(len(gap) - 1) // 2]
                gaps.append((len(gap), -left, midpoint))
        if not gaps:
            break
        _, _, midpoint = max(gaps)
        add(midpoint)
    return tuple(selected)


def portfolio_summary(state: BoundState) -> Mapping[str, object]:
    return {
        "lower_bound": state.lower,
        "upper_bound": state.upper,
        "decisive": state.decisive,
        "closed": state.closed,
        "tested_k": sorted(state.tested),
        "unsat_boundary": state.unsat_boundary,
    }
