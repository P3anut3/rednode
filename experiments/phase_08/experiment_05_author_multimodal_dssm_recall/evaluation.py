"""Reuse shared macro evaluator; no test API or alternate ground-truth definition."""
import numpy as np

from experiments.common.data import TestRequest
from experiments.common.metrics import evaluate_rankings
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.config import LEGACY_EVALUATOR_CACHE
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.evaluation import phase6_status_metrics
from .config import OUT


def evaluate(requests, rankings, assets, label, guard):
    targets = set(map(int, np.load(OUT / 'features/train_target_ids.npy')))
    history = set(map(int, np.load(OUT / 'features/train_history_ids.npy')))
    users = set(map(int, assets.train_users))
    legacy = [set(map(int, np.load(LEGACY_EVALUATOR_CACHE / name))) for name in
              ('train_exposed.npy', 'train_clicked.npy', 'train_users.npy')]
    main, temporal, frame = phase6_status_metrics(requests, rankings, targets, history | targets,
                                                users, label, *legacy, deadline=guard.deadline)
    extra = {}
    for name in ('with_image', 'without_image', 'temporal_seen_user', 'temporal_unseen_user',
                 'history_empty', 'history_1_5', 'history_6_10', 'history_11_20'):
        segment_requests = []
        for r in requests:
            truth = set(r.ground_truth)
            if name in ('with_image', 'without_image'):
                ordered = sorted(truth)
                mapped = assets.lookup(ordered)
                image_ids = {note for note, row in zip(ordered, mapped)
                             if row >= 0 and assets.available[row]}
                truth = image_ids if name == 'with_image' else truth - image_ids
            elif name.startswith('temporal_'):
                truth = truth if ((r.user_idx in users) == (name == 'temporal_seen_user')) else set()
            else:
                size = min(20, sum(int(v) >= 0 for v in r.history))
                low, high = {'history_empty': (0, 0), 'history_1_5': (1, 5),
                             'history_6_10': (6, 10), 'history_11_20': (11, 20)}[name]
                truth = truth if low <= size <= high else set()
            segment_requests.append(TestRequest(r.request_idx, r.user_idx, r.history, frozenset(truth)))
        metrics, _ = evaluate_rankings(segment_requests, rankings, set(), users, label,
                                       deadline=guard.deadline)
        extra[name] = metrics['overall']
    guard.check()
    return {'metrics': main, 'phase6_item_status': temporal, 'segments': extra,
            'top500_coverage': float(np.mean([len(x) == 500 for x in rankings])),
            'empty_history_rate': float(np.mean([not r.history for r in requests])),
            'test_opened': False}, frame


def recall_values(requests, rankings, subset=None):
    values = np.full(len(requests), np.nan)
    for i, (r, ranking) in enumerate(zip(requests, rankings)):
        truth = set(r.ground_truth)
        if subset is not None:
            truth &= subset
        if truth:
            values[i] = len(truth & set(map(int, ranking[:500]))) / len(truth)
    return values


def paired(requests, newer, reference, subset=None, seed=42):
    delta = recall_values(requests, newer, subset) - recall_values(requests, reference, subset)
    return bootstrap_delta(delta, seed)


def bootstrap_delta(delta, seed=42):
    delta = delta[np.isfinite(delta)]
    if not len(delta):
        return {'eligible_requests': 0, 'status': 'N/A', 'delta': None, 'ci95': None}
    rng = np.random.default_rng(seed)
    means = [float(delta[rng.integers(len(delta), size=len(delta))].mean()) for _ in range(2000)]
    return {'eligible_requests': len(delta), 'delta': float(delta.mean()),
            'ci95': list(map(float, np.quantile(means, [.025, .975]))),
            'scope': 'request uncertainty, not random initialization uncertainty'}
