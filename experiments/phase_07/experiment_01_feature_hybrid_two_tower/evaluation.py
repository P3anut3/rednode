"""One-pass formal metrics, Phase-6 status splits, and paired bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.common.metrics import KS


def phase6_status_metrics(
    requests,
    rankings,
    train_targets: set[int],
    train_vocab: set[int],
    temporal_train_users: set[int],
    method: str,
    old_warm_items: set[int],
    old_clicked_items: set[int],
    old_warm_users: set[int],
    deadline: float | None = None,
):
    """Compute legacy and temporal splits in one ranking pass.

    The previous implementation invoked the full five/seven-segment evaluator
    four times.  That rebuilt a 500-item rank dictionary and traversed every
    request repeatedly.  This keeps the exact metric definitions while sharing
    the ranking lookup across all ten required segments.
    """
    import time

    legacy_names = (
        "overall",
        "train_clicked",
        "train_exposed_never_clicked",
        "warm_item",
        "cold_item",
        "warm_user",
        "cold_user",
    )
    temporal_names = (
        "train_target_seen",
        "train_history_only",
        "completely_unseen",
    )

    def empty_bucket():
        return {
            "recalls": {k: [] for k in KS},
            "hits": {k: [] for k in KS},
            "rr100": [],
            "rr500": [],
            "ranks": [],
            "misses": 0,
            "eligible": 0,
        }

    accum = {name: empty_bucket() for name in (*legacy_names, *temporal_names)}
    per_request = []
    history_only_items = train_vocab - train_targets
    exposed_never_clicked = old_warm_items - old_clicked_items

    def update(name: str, truth: set[int], rank_by_note: dict[int, int]) -> None:
        if not truth:
            return
        bucket = accum[name]
        bucket["eligible"] += 1
        hit_ranks = [rank_by_note[note] for note in truth if note in rank_by_note]
        first = min(hit_ranks, default=None)
        if first is None:
            bucket["misses"] += 1
        else:
            bucket["ranks"].append(first)
        bucket["rr100"].append(1.0 / first if first and first <= 100 else 0.0)
        bucket["rr500"].append(1.0 / first if first and first <= 500 else 0.0)
        for k in KS:
            hits = sum(rank <= k for rank in hit_ranks)
            bucket["recalls"][k].append(hits / len(truth))
            bucket["hits"][k].append(float(hits > 0))

    for request_index, (request, ranking_values) in enumerate(zip(requests, rankings)):
        if (
            request_index % 250 == 0
            and deadline is not None
            and time.monotonic() > deadline
        ):
            raise TimeoutError("Phase 7 one-pass evaluation exceeded deadline")
        ranking = list(map(int, ranking_values[:500]))
        truth = set(request.ground_truth)
        rank_by_note = {
            note: rank for rank, note in enumerate(ranking, 1) if note in truth
        }
        overall_hit_ranks = list(rank_by_note.values())
        first = min(overall_hit_ranks, default=None)
        per_request.append(
            {
                "request_idx": request.request_idx,
                "user_idx": request.user_idx,
                "ground_truth": sorted(truth),
                "retrieved_top500": ranking,
                "first_hit_rank": first,
                **{
                    f"num_hits@{k}": sum(rank <= k for rank in overall_hit_ranks)
                    for k in KS
                },
            }
        )
        update("overall", truth, rank_by_note)
        update("train_clicked", truth & old_clicked_items, rank_by_note)
        update(
            "train_exposed_never_clicked",
            truth & exposed_never_clicked,
            rank_by_note,
        )
        update("warm_item", truth & old_warm_items, rank_by_note)
        update("cold_item", truth - old_warm_items, rank_by_note)
        update(
            "warm_user",
            truth if request.user_idx in old_warm_users else set(),
            rank_by_note,
        )
        update(
            "cold_user",
            truth if request.user_idx not in old_warm_users else set(),
            rank_by_note,
        )
        update("train_target_seen", truth & train_targets, rank_by_note)
        update("train_history_only", truth & history_only_items, rank_by_note)
        update("completely_unseen", truth - train_vocab, rank_by_note)

    def finalize(name: str, label: str) -> dict:
        bucket = accum[name]
        eligible = int(bucket["eligible"])
        ranks = bucket["ranks"]
        row = {
            "method": label,
            "segment": "overall" if name in temporal_names else name,
            "eligible_requests": eligible,
            "overall_miss_rate@500": (
                bucket["misses"] / eligible if eligible else float("nan")
            ),
            "mean_first_hit_rank_on_hits": (
                float(np.mean(ranks)) if ranks else float("nan")
            ),
            "median_first_hit_rank_on_hits": (
                float(np.median(ranks)) if ranks else float("nan")
            ),
            "MRR@100": (float(np.mean(bucket["rr100"])) if eligible else float("nan")),
            "MRR@500": (float(np.mean(bucket["rr500"])) if eligible else float("nan")),
        }
        for k in KS:
            row[f"Recall@{k}"] = (
                float(np.mean(bucket["recalls"][k])) if eligible else float("nan")
            )
            row[f"HitRate@{k}"] = (
                float(np.mean(bucket["hits"][k])) if eligible else float("nan")
            )
        return row

    base = {name: finalize(name, method) for name in legacy_names}
    extra = {name: finalize(name, f"{method}_{name}") for name in temporal_names}
    return base, extra, pd.DataFrame(per_request)


def paired_bootstrap(
    base_per_request,
    candidate_per_request,
    requests,
    segment: str,
    warm_items: set[int],
    train_target_items: set[int] | None = None,
    train_vocab_items: set[int] | None = None,
    replicates: int = 2_000,
    seed: int = 42,
    deadline: float | None = None,
) -> dict:
    base = {
        int(row.request_idx): set(row.retrieved_top500)
        for row in base_per_request.itertuples(index=False)
    }
    candidate = {
        int(row.request_idx): set(row.retrieved_top500)
        for row in candidate_per_request.itertuples(index=False)
    }
    values = []
    for request in requests:
        truth = set(request.ground_truth)
        if segment == "warm":
            truth &= warm_items
        elif segment == "cold":
            truth -= warm_items
        elif segment == "train_target_seen":
            if train_target_items is None:
                raise ValueError("train_target_items is required")
            truth &= train_target_items
        elif segment == "train_history_only":
            if train_target_items is None or train_vocab_items is None:
                raise ValueError("temporal train item sets are required")
            truth &= train_vocab_items - train_target_items
        elif segment == "completely_unseen":
            if train_vocab_items is None:
                raise ValueError("train_vocab_items is required")
            truth -= train_vocab_items
        elif segment != "overall":
            raise ValueError(segment)
        if truth:
            values.append(
                len(truth & candidate[request.request_idx]) / len(truth)
                - len(truth & base[request.request_idx]) / len(truth)
            )
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {
            "segment": segment,
            "eligible_requests": 0,
            "point_delta": float("nan"),
            "ci95_lower": float("nan"),
            "ci95_upper": float("nan"),
            "replicates": 0,
        }
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 250):
        if deadline is not None:
            import time

            if time.monotonic() > deadline:
                raise TimeoutError("paired bootstrap exceeded deadline")
        count = min(250, replicates - start)
        sample = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[sample].mean(axis=1)
    return {
        "segment": segment,
        "eligible_requests": len(values),
        "point_delta": float(values.mean()),
        "ci95_lower": float(np.quantile(means, 0.025)),
        "ci95_upper": float(np.quantile(means, 0.975)),
        "replicates": replicates,
    }


def paired_bootstrap_segments(
    base_per_request,
    candidate_per_request,
    requests,
    segments: tuple[str, ...],
    warm_items: set[int],
    train_target_items: set[int],
    train_vocab_items: set[int],
    replicates: int = 2_000,
    seed: int = 42,
    deadline: float | None = None,
) -> dict:
    """Compute several paired CIs while materializing Top500 sets only once."""
    base = {
        int(row.request_idx): set(row.retrieved_top500)
        for row in base_per_request.itertuples(index=False)
    }
    candidate = {
        int(row.request_idx): set(row.retrieved_top500)
        for row in candidate_per_request.itertuples(index=False)
    }
    values = {segment: [] for segment in segments}
    for request in requests:
        truth = set(request.ground_truth)
        truths = {
            "overall": truth,
            "warm": truth & warm_items,
            "cold": truth - warm_items,
            "train_target_seen": truth & train_target_items,
            "train_history_only": truth
            & (train_vocab_items - train_target_items),
            "completely_unseen": truth - train_vocab_items,
        }
        base_top = base[request.request_idx]
        candidate_top = candidate[request.request_idx]
        for segment in segments:
            segment_truth = truths[segment]
            if segment_truth:
                values[segment].append(
                    len(segment_truth & candidate_top) / len(segment_truth)
                    - len(segment_truth & base_top) / len(segment_truth)
                )
    result = {}
    for offset, segment in enumerate(segments):
        array = np.asarray(values[segment], dtype=np.float64)
        if len(array) == 0:
            result[segment] = {
                "segment": segment,
                "eligible_requests": 0,
                "point_delta": float("nan"),
                "ci95_lower": float("nan"),
                "ci95_upper": float("nan"),
                "replicates": 0,
            }
            continue
        rng = np.random.default_rng(seed + offset)
        means = np.empty(replicates, dtype=np.float64)
        for start in range(0, replicates, 250):
            if deadline is not None:
                import time

                if time.monotonic() > deadline:
                    raise TimeoutError("paired bootstrap exceeded deadline")
            count = min(250, replicates - start)
            sample = rng.integers(0, len(array), size=(count, len(array)))
            means[start : start + count] = array[sample].mean(axis=1)
        result[segment] = {
            "segment": segment,
            "eligible_requests": len(array),
            "point_delta": float(array.mean()),
            "ci95_lower": float(np.quantile(means, 0.025)),
            "ci95_upper": float(np.quantile(means, 0.975)),
            "replicates": replicates,
        }
    return result


def route_contribution(
    requests, candidate_rankings, reference_frame: pd.DataFrame
) -> dict:
    reference = {
        int(row.request_idx): set(map(int, row.retrieved_top500))
        for row in reference_frame.itertuples(index=False)
    }
    candidate_hits, reference_hits = set(), set()
    for request, ranking in zip(requests, candidate_rankings):
        truth = set(request.ground_truth)
        candidate_hits.update(
            (request.request_idx, note) for note in truth & set(ranking[:500])
        )
        reference_hits.update(
            (request.request_idx, note)
            for note in truth & reference.get(request.request_idx, set())
        )
    return {
        "candidate_positive_hits": len(candidate_hits),
        "reference_positive_hits": len(reference_hits),
        "candidate_only": len(candidate_hits - reference_hits),
        "reference_only": len(reference_hits - candidate_hits),
        "both": len(candidate_hits & reference_hits),
        "neither": sum(len(r.ground_truth) for r in requests)
        - len(candidate_hits | reference_hits),
    }


def feature_slices(
    requests, rankings, store, target_frequency: dict[int, int]
) -> pd.DataFrame:
    accum = {}

    def add(group: str, value: str, hit: bool):
        cell = accum.setdefault((group, value), [0, 0])
        cell[0] += 1
        cell[1] += int(hit)

    for request, ranking in zip(requests, rankings):
        retrieved = set(ranking[:500])
        profile_row = (
            int(store.user_lookup[request.user_idx])
            if 0 <= request.user_idx < len(store.user_lookup)
            else -1
        )
        for note in request.ground_truth:
            hit = note in retrieved
            add("user_profile", "available" if profile_row >= 0 else "missing", hit)
            item_row = (
                int(store.item_lookup[note])
                if 0 <= note < len(store.item_lookup)
                else -1
            )
            if item_row >= 0:
                category = store.item_categorical[item_row]
                add("note_type", str(int(category[0])), hit)
                add("taxonomy1", str(int(category[1])), hit)
                add("taxonomy2", str(int(category[2])), hit)
            frequency = target_frequency.get(int(note), 0)
            bucket = (
                "0"
                if frequency == 0
                else (
                    "1"
                    if frequency == 1
                    else "2-3"
                    if frequency <= 3
                    else "4-10"
                    if frequency <= 10
                    else "11+"
                )
            )
            add("train_target_frequency", bucket, hit)
    return pd.DataFrame(
        [
            {
                "slice": key[0],
                "value": key[1],
                "positive_count": count,
                "positive_hits@500": hits,
                "positive_recall@500": hits / count,
            }
            for key, (count, hits) in accum.items()
        ]
    )
