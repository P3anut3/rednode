"""One evaluator shared by every recall baseline."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import pandas as pd

from .data import TestRequest


KS = (10, 50, 100, 200, 500)


def _target_for_segment(
    request: TestRequest,
    segment: str,
    warm_items: set[int],
    warm_users: set[int],
    clicked_items: set[int] | None = None,
) -> set[int] | None:
    truth = set(request.ground_truth)
    if segment == "overall":
        return truth
    if segment == "warm_item":
        target = truth & warm_items
        return target or None
    if segment == "train_clicked":
        target = truth & (clicked_items or set())
        return target or None
    if segment == "train_exposed_never_clicked":
        target = truth & (warm_items - (clicked_items or set()))
        return target or None
    if segment == "cold_item":
        target = truth - warm_items
        return target or None
    if segment == "warm_user":
        return truth if request.user_idx in warm_users else None
    if segment == "cold_user":
        return truth if request.user_idx not in warm_users else None
    raise ValueError(f"Unknown segment: {segment}")


def evaluate_rankings(
    requests: Sequence[TestRequest],
    rankings: Sequence[Sequence[int]],
    warm_items: set[int],
    warm_users: set[int],
    method: str,
    clicked_items: set[int] | None = None,
    deadline: float | None = None,
) -> tuple[dict[str, dict[str, float | int]], pd.DataFrame]:
    if len(requests) != len(rankings):
        raise ValueError("requests and rankings must have equal length")
    segments = ["overall", "warm_item", "cold_item", "warm_user", "cold_user"]
    if clicked_items is not None:
        segments[1:1] = ["train_clicked", "train_exposed_never_clicked"]
    accum = {
        s: {
            "recalls": {k: [] for k in KS},
            "hits": {k: [] for k in KS},
            "rr100": [],
            "rr500": [],
            "ranks": [],
            "misses": 0,
            "eligible": 0,
        }
        for s in segments
    }
    per_request = []
    for request_index, (request, ranking_values) in enumerate(zip(requests, rankings)):
        if (
            request_index % 250 == 0
            and deadline is not None
            and time.monotonic() > deadline
        ):
            raise TimeoutError("ranking evaluation exceeded deadline")
        ranking = list(map(int, ranking_values[:500]))
        # A ranking is already validated as unique by the caller.  Build the
        # inverse once instead of repeatedly materializing set(ranking[:k])
        # for every K and every segment (7 * 5 copies per request in Phase 2).
        rank_by_note = {note: rank for rank, note in enumerate(ranking, 1)}
        overall_truth = set(request.ground_truth)
        overall_hit_ranks = [
            rank_by_note[note] for note in overall_truth if note in rank_by_note
        ]
        overall_hits = {k: sum(rank <= k for rank in overall_hit_ranks) for k in KS}
        first = min(overall_hit_ranks, default=None)
        per_request.append(
            {
                "request_idx": request.request_idx,
                "user_idx": request.user_idx,
                "ground_truth": sorted(overall_truth),
                "retrieved_top500": ranking,
                "first_hit_rank": first,
                **{f"num_hits@{k}": overall_hits[k] for k in KS},
            }
        )
        for segment in segments:
            target = _target_for_segment(
                request, segment, warm_items, warm_users, clicked_items
            )
            if not target:
                continue
            bucket = accum[segment]
            bucket["eligible"] += 1
            target_hit_ranks = [
                rank_by_note[note] for note in target if note in rank_by_note
            ]
            first_hit = min(target_hit_ranks, default=None)
            if first_hit is None:
                bucket["misses"] += 1
            else:
                bucket["ranks"].append(first_hit)
            bucket["rr100"].append(
                1.0 / first_hit if first_hit is not None and first_hit <= 100 else 0.0
            )
            bucket["rr500"].append(
                1.0 / first_hit if first_hit is not None and first_hit <= 500 else 0.0
            )
            for k in KS:
                num_hits = sum(rank <= k for rank in target_hit_ranks)
                bucket["recalls"][k].append(num_hits / len(target))
                bucket["hits"][k].append(float(num_hits > 0))

    metrics: dict[str, dict[str, float | int]] = {}
    for segment, bucket in accum.items():
        ranks = bucket["ranks"]
        eligible = int(bucket["eligible"])
        row: dict[str, float | int] = {
            "method": method,
            "segment": segment,
            "eligible_requests": eligible,
            "overall_miss_rate@500": bucket["misses"] / eligible
            if eligible
            else float("nan"),
            "mean_first_hit_rank_on_hits": float(np.mean(ranks))
            if ranks
            else float("nan"),
            "median_first_hit_rank_on_hits": float(np.median(ranks))
            if ranks
            else float("nan"),
            "MRR@100": float(np.mean(bucket["rr100"])) if eligible else float("nan"),
            "MRR@500": float(np.mean(bucket["rr500"])) if eligible else float("nan"),
        }
        for k in KS:
            row[f"Recall@{k}"] = (
                float(np.mean(bucket["recalls"][k])) if eligible else float("nan")
            )
            row[f"HitRate@{k}"] = (
                float(np.mean(bucket["hits"][k])) if eligible else float("nan")
            )
        metrics[segment] = row
    return metrics, pd.DataFrame(per_request)
