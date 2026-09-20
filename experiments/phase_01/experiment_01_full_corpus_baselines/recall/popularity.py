"""Popularity recall built exclusively from recommendation_train logs."""

from __future__ import annotations

from typing import Sequence

from experiments.common.data import TestRequest


def rank_popularity(clicks: dict[int, int], exposures: dict[int, int], mode: str = "click") -> list[int]:
    notes = set(exposures) | set(clicks)
    if mode == "click":
        key = lambda n: (-clicks.get(n, 0), -exposures.get(n, 0), n)
    elif mode == "exposure":
        key = lambda n: (-exposures.get(n, 0), -clicks.get(n, 0), n)
    elif mode == "ctr":
        # Prepared only as an alternate diagnostic; not used as the main baseline.
        key = lambda n: (-(clicks.get(n, 0) / exposures[n]), -exposures[n], n)
    else:
        raise ValueError(f"Unknown popularity mode: {mode}")
    return sorted(notes, key=key)


def filter_ranking(ranking: Sequence[int], history: Sequence[int], k: int, exclude_history: bool) -> list[int]:
    blocked = set(history) if exclude_history else set()
    return [int(note) for note in ranking if note not in blocked][:k]


class GlobalPopularity:
    def __init__(self, clicks: dict[int, int], exposures: dict[int, int], mode: str = "click"):
        self.mode = mode
        self.ranking = rank_popularity(clicks, exposures, mode=mode)

    def recommend(self, requests: Sequence[TestRequest], k: int = 500, exclude_history: bool = True) -> list[list[int]]:
        scan = self.ranking[: k + 64]
        return [filter_ranking(scan, request.history, k, exclude_history) for request in requests]
