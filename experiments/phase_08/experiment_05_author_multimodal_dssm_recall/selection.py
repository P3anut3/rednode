"""Pure, synthetic-testable three-seed selection contract."""
import numpy as np

SEEDS = (42, 43, 44)


def choose_models(scores):
    """Scores keyed by model, then seed; a one-seed T can never become final D."""
    for name in ('A0', 'A1', 'B0', 'B1'):
        if any(seed not in scores.get(name, {}) for seed in SEEDS):
            raise ValueError(f'{name} requires seeds42/43/44')
    for name in ('T0', 'T1'):
        if 42 not in scores.get(name, {}):
            raise ValueError(f'{name} requires seed42 screening')
    best_b42 = max(scores['B0'][42], scores['B1'][42])
    # T initial screening compares same seed, not a one-seed score against a
    # three-seed mean. A T that ties/beats the best B42 must be replicated first.
    required_t = [name for name in ('T0', 'T1') if scores[name][42] >= best_b42]
    pending = {name: [seed for seed in SEEDS if seed not in scores[name]]
               for name in required_t}
    pending = {name: seeds for name, seeds in pending.items() if seeds}
    means = {name: float(np.mean([scores[name][s] for s in SEEDS]))
             for name in scores if all(s in scores[name] for s in SEEDS)}
    eligible = [name for name in ('B0', 'B1', 'T0', 'T1') if name in means]
    return {
        'selection_ready': not pending, 'required_T_replications': pending,
        'safe_model': max(eligible, key=lambda m: means[m]) if not pending else None,
        'reference_model': max(('A0', 'A1'), key=lambda m: means[m]),
        'three_seed_means': means,
        'three_seed_stds': {name: float(np.std([scores[name][s] for s in SEEDS])) for name in means},
        'eligible_safe_models': eligible, 'seed_rule': 'three-seed mean; same-seed T42 screening only',
        'T_screening_rule': 'T42 >= max(B0_42,B1_42) requires T43/44 before final selection',
    }
