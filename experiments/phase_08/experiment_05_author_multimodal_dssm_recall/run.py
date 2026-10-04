"""Review-gated, validation-only Phase8-05. Default plan does not touch data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'experiments.phase_08.experiment_05_author_multimodal_dssm_recall'

import numpy as np
import pandas as pd
import torch

from experiments.phase_06.experiment_01_id_two_tower_retrieval.data import load_temporal_frames, grouped_requests
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.retrieval import deterministic_proxy_candidates
from .artifacts import (Guard, atomic_json, atomic_parquet, atomic_torch, complete,
                        verified, sha256, stage_lock, cuda_preflight)
from .config import OUT, ROOT, EXPERIMENT, MODELS, PROTOCOL, H2, ITEM_NUMERIC
from .data import Assets, build_features
from .evaluation import evaluate, paired, recall_values, bootstrap_delta
from .models import MultimodalDSSM
from .retrieval import ResidentExact, filtered, validate_rankings
from .trainer import train_epoch, encode_items, encode_queries, validation_loss


def source_hashes():
    return {str(p.relative_to(EXPERIMENT)): sha256(p) for p in
            sorted(EXPERIMENT.glob('*.py')) + [EXPERIMENT / 'source_manifest.json']}


def compatible_source(previous):
    """Only an explicit, exact-hash numerical-runtime patch accepts old assets.

    Old completion markers stay immutable. Model/data/loss/split source changes
    cannot be grandfathered by this migration contract.
    """
    current = source_hashes()
    if previous == current:
        return True
    path = OUT / 'debug/numerical_recovery/runtime_patch.json'
    if not path.exists():
        return False
    patch = json.loads(path.read_text())
    if (patch.get('previous_source_hashes') != previous
            or patch.get('current_source_hashes') != current
            or patch.get('test_opened') is not False):
        return False
    if set(previous) != set(current):
        return False
    changed = {name for name in previous if previous[name] != current[name]}
    if not changed <= {'run.py', 'trainer.py', 'self_check.py', 'reporting.py'}:
        return False
    snapshot = OUT / 'debug/numerical_recovery/source_v1'
    return all((snapshot / name).is_file() and sha256(snapshot / name) == digest
               for name, digest in previous.items())


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def create(model_name, seed, device):
    cfg = MODELS[model_name]
    assets = Assets(cfg['features'], cfg['images'])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = MultimodalDSSM(assets.vocabs, len(ITEM_NUMERIC), cfg['id_dropout'],
                          dropout_seed=900000 + seed).to(device)
    return model, assets


def proxy(assets, valid, count=5000):
    requests = grouped_requests(valid)
    order = np.random.default_rng(42).permutation(len(requests))[:count]
    requests = [requests[int(i)] for i in sorted(order)]
    positives = set().union(*(r.ground_truth for r in requests))
    candidates = deterministic_proxy_candidates(assets.catalog, positives, 100000, 42)
    return requests, candidates


def retrieve(model, assets, requests, candidate_ids, device, guard, vectors=None):
    started = time.monotonic()
    if vectors is None:
        vectors = encode_items(model, assets, assets.lookup(candidate_ids), device, guard)
    encoding = time.monotonic() - started
    query_started = time.monotonic()
    queries = encode_queries(model, assets, requests, device, guard)
    query_encoding = time.monotonic() - query_started
    index = ResidentExact(vectors, device, guard)
    try:
        # Exclude the full original history, not only the input last20.
        overfetch = min(len(candidate_ids), 500 + max((len(set(r.history)) for r in requests), default=0))
        _, raw_rows, elapsed = index.search(queries, overfetch)
        rankings = filtered(raw_rows, candidate_ids, [r.history for r in requests])
        validate_rankings(requests, rankings, assets.catalog)
        timing = {'item_encoding_seconds': encoding, 'index_build_seconds': index.build_seconds,
                  'query_encoding_seconds': query_encoding,
                  'query_search_seconds': elapsed, 'index_bytes': index.index_bytes,
                  'mean_search_latency_seconds': elapsed / max(1, len(requests)),
                  'backend': 'GPU resident FP32 exact IP; no CPU fallback'}
    finally:
        index.close()
    return rankings, timing


def budget_gib():
    """Remaining entire preregistered plan, not just next temporary output."""
    # Three model-only epoch checkpoints plus one vector file; no duplicate best.
    model_bytes = (PROTOCOL['items'] + 1) * 32 * 4 + 12 * 2**20
    remaining = 0
    for m in MODELS:
        for s in ([42, 43, 44] if m[0] in 'AB' else [42]):
            name = f'{m}_seed{s}'
            if not (OUT / 'training' / name / 'complete.json').exists():
                remaining += 3 * model_bytes
            if not (OUT / 'validation' / name / 'item_vectors.f32.npy').exists():
                remaining += PROTOCOL['items'] * 128 * 4
    return remaining / 2**30 + 4


def execute_training(args, smoke=False):
    stage_started = time.monotonic()
    if not smoke and (OUT / 'selection.json').exists():
        raise RuntimeError('final selection frozen; no further training in this experiment')
    device_info = cuda_preflight(args.device)
    model, assets = create(args.model, args.seed, args.device)
    train, valid = load_temporal_frames(ROOT)
    folder = OUT / ('smoke' if smoke else 'training') / f'{args.model}_seed{args.seed}'
    if folder.exists():
        raise FileExistsError(f'no overwriting run: {folder}')
    if not smoke:
        for name in MODELS:
            marker = verified(OUT / 'smoke' / f'{name}_seed42')
            if not compatible_source(marker['source_hashes']) or marker['features_marker_sha256'] != sha256(OUT / 'features/complete.json'):
                raise RuntimeError('all six smoke runs must match code/features before formal training')
        if args.seed != 42:
            for name in MODELS:
                verified(OUT / 'validation' / f'{name}_seed42')
                verified(OUT / 'diagnostics' / f'{name}_seed42')
    guard = Guard(600 if smoke else args.train_seconds, OUT,
                  max_rss_gib=args.max_rss_gib, disk_need_gib=.5 if smoke else budget_gib())
    folder.mkdir(parents=True)
    features_digest = sha256(OUT / 'features/complete.json')
    initial = state_hash(model)
    atomic_json(folder / 'config.json', {'model': args.model, 'seed': args.seed,
                'matrix': MODELS[args.model], 'protocol': PROTOCOL,
                'source_hashes': source_hashes(), 'features_marker_sha256': features_digest,
                'initial_state_sha256': initial, 'parameters': sum(p.numel() for p in model.parameters()),
                'id_parameters': model.i_emb_item.weight.numel() + model.u_emb_user.weight.numel(),
                'device': device_info, 'frozen_embedding_parameters': 0,
                'reference_setting': 'author architecture/local assets/temporal split; not published reproduction'})
    patch_path = OUT / 'debug/numerical_recovery/runtime_patch.json'
    if patch_path.exists():
        atomic_json(folder / 'runtime_patch_binding.json', {'sha256': sha256(patch_path),
                    'protocol': 'same-batch AMP backoff; no loss/model/data change'})
    if smoke:
        train = train.iloc[np.random.default_rng(42).permutation(len(train))[:10000]].reset_index(drop=True)
    requests, candidates = proxy(assets, valid, count=1000 if smoke else 5000)
    np.save(folder / 'proxy_request_ids.npy', np.asarray([r.request_idx for r in requests], np.int64))
    np.save(folder / 'proxy_candidate_ids.npy', candidates)
    checkpoint_dir = folder / 'checkpoints'
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=.5)
    scaler = torch.amp.GradScaler('cuda')
    curves, best, stale = [], -1., 0
    shuffle = np.random.default_rng(args.seed)  # independent of torch dropout draws
    for epoch in range(1, 2 if smoke else 4):
        record = train_epoch(model, assets, train, shuffle.permutation(len(train)), optimizer,
                             scaler, args.device, guard, epoch, checkpoint_dir)
        rankings, timing = retrieve(model, assets, requests, candidates, args.device, guard)
        result, _ = evaluate(requests, rankings, assets, args.model, guard)
        score = result['metrics']['overall']['Recall@500']
        val_loss = validation_loss(model, assets,
                    valid[valid.request_id.isin([r.request_idx for r in requests])], args.device, guard)
        scheduler.step(val_loss)
        record.update(proxy_R500=score, proxy_timing=timing, fixed_proxy_validation_loss=val_loss,
                      lr=optimizer.param_groups[0]['lr'])
        curves.append(record)
        path = checkpoint_dir / f'epoch_{epoch}.pt'
        atomic_torch(path, {'state_dict': model.state_dict(), 'epoch': epoch,
                     'model_name': args.model, 'seed': args.seed,
                     'source_hashes': source_hashes(), 'features_marker_sha256': features_digest,
                     'dropout_rng': model.dropout_generator.get_state()})
        atomic_json(folder / 'training_curves.json', curves)
        if score > best:
            best, stale = score, 0
            atomic_json(folder / 'best.json', {'epoch': epoch, 'proxy_R500': score,
                        'checkpoint': str(path.relative_to(folder)), 'checkpoint_sha256': sha256(path)})
        else:
            stale += 1
        print(f'{args.model} seed{args.seed} epoch{epoch}: loss={record["loss"]:.5f}, '
              f'fixed proxy R500={score:.6f}', flush=True)
        if stale >= 2:
            break
    pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, dict)} for row in curves]).to_csv(folder / 'training_curves.csv', index=False)
    guard.check()
    complete(folder, [p for p in folder.rglob('*') if p.is_file()], {
        'source_hashes': source_hashes(), 'features_marker_sha256': features_digest,
        'initial_state_sha256': initial, 'stage': 'smoke' if smoke else 'training',
        'stage_wall_seconds': time.monotonic() - stage_started,
        'normal_finish': True})


def load_best(model_name, seed, device):
    folder = OUT / 'training' / f'{model_name}_seed{seed}'
    marker = verified(folder)
    if not compatible_source(marker['source_hashes']) or marker['features_marker_sha256'] != sha256(OUT / 'features/complete.json'):
        raise RuntimeError('training source/features changed')
    best = json.loads((folder / 'best.json').read_text())
    path = folder / best['checkpoint']
    if sha256(path) != best['checkpoint_sha256']:
        raise RuntimeError('best checkpoint replaced')
    model, assets = create(model_name, seed, device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)['state_dict'])
    model.eval()
    return model, assets, best['checkpoint_sha256']


def full_validation(args):
    cuda_preflight(args.device)
    model, assets, checkpoint_hash = load_best(args.model, args.seed, args.device)
    folder = OUT / 'validation' / f'{args.model}_seed{args.seed}'
    if folder.exists() and (folder / 'complete.json').exists():
        verified(folder)
        raise FileExistsError('completed validation is immutable')
    if folder.exists() and not args.resume:
        raise FileExistsError('formal validation already exists/incomplete; do not overwrite')
    guard = Guard(args.validation_seconds, OUT, args.max_rss_gib, disk_need_gib=budget_gib())
    _, valid = load_temporal_frames(ROOT)
    requests = grouped_requests(valid)
    if len(requests) != 13594:
        raise AssertionError('validation request count changed')
    folder.mkdir(parents=True, exist_ok=args.resume)
    started = time.monotonic()
    vector_path = folder / 'item_vectors.f32.npy'
    tmp = folder / 'item_vectors.building.npy'
    ready = folder / 'vectors_ready.json'
    if ready.exists():
        contract = json.loads(ready.read_text())
        if contract != {'checkpoint_sha256': checkpoint_hash,
                       'mapping_sha256': sha256(OUT / 'features/note_ids.npy'),
                       'vectors_sha256': sha256(vector_path)}:
            raise RuntimeError('cannot resume modified/mismatched vectors')
    else:
        vectors = encode_items(model, assets, np.arange(len(assets.catalog)), args.device, guard, tmp)
        del vectors
        os.replace(tmp, vector_path)
        atomic_json(ready, {'checkpoint_sha256': checkpoint_hash,
                           'mapping_sha256': sha256(OUT / 'features/note_ids.npy'),
                           'vectors_sha256': sha256(vector_path)})
    encoding = time.monotonic() - started
    vectors = np.load(vector_path, mmap_mode='r')
    rankings, timing = retrieve(model, assets, requests, assets.catalog, args.device, guard, vectors)
    timing['item_encoding_seconds'] = encoding
    result, frame = evaluate(requests, rankings, assets, args.model, guard)
    atomic_parquet(folder / 'rankings.parquet', frame)
    reference, ref_hashes = h2_rankings(requests, args.seed)
    comparison = paired_slices(requests, rankings, reference, assets)
    result.update(timing=timing, comparison_H2_same_seed=comparison, checkpoint_sha256=checkpoint_hash,
                  gpu_peak_memory_bytes=torch.cuda.max_memory_allocated(args.device),
                  process_tree_peak_rss_gib=guard.peak_rss / 2**30,
                  checkpoint_bytes=sum(p.stat().st_size for p in (OUT / 'training' / f'{args.model}_seed{args.seed}/checkpoints').glob('epoch_*.pt')),
                  item_vector_bytes=vector_path.stat().st_size,
                  reference_hashes=ref_hashes, vectors_sha256=sha256(vector_path),
                  vector_shape=list(vectors.shape), vector_dtype=str(vectors.dtype),
                  candidate_mapping_sha256=sha256(OUT / 'features/note_ids.npy'))
    atomic_json(folder / 'metrics.json', result)
    atomic_json(folder / 'vector_metadata.json', {k: result[k] for k in
                ('checkpoint_sha256', 'vectors_sha256', 'vector_shape', 'vector_dtype', 'candidate_mapping_sha256')})
    guard.check()
    complete(folder, [vector_path, ready, folder / 'rankings.parquet', folder / 'metrics.json', folder / 'vector_metadata.json'],
             {'checkpoint_sha256': checkpoint_hash, 'source_hashes': source_hashes(),
              'features_marker_sha256': sha256(OUT / 'features/complete.json')})


def h2_rankings(requests, seed):
    path = H2 / f'validation/rankings/h2_256/seed{seed}.parquet'
    metadata = H2 / f'validation/h2_256_seed{seed}.json'
    marker = H2 / f'configs/markers/full_validation_h2_256_seed{seed}.json'
    completion = json.loads(marker.read_text())
    result = json.loads(metadata.read_text())
    if not completion.get('complete') or completion.get('test_opened') is not False or result['checkpoint_sha256'] != completion['checkpoint_sha256']:
        raise RuntimeError('H2 formal validation completion invalid')
    expected = json.loads((OUT / 'features/feature_manifest.json').read_text())['h2_validation_dependencies']
    for p in (path, metadata, marker):
        if sha256(p) != expected[str(p.relative_to(ROOT))]:
            raise RuntimeError('H2 reference changed after audit')
    frame = pd.read_parquet(path).set_index('request_idx')
    rankings = []
    for r in requests:
        row = frame.loc[r.request_idx]
        if set(map(int, row.ground_truth)) != set(r.ground_truth) or int(row.user_idx) != r.user_idx:
            raise AssertionError('H2 validation protocol mismatch')
        rankings.append(list(map(int, row.retrieved_top500)))
    return rankings, {str(p.relative_to(ROOT)): sha256(p) for p in (path, metadata, marker)}


def paired_slices(requests, rankings, reference, assets):
    target = set(map(int, np.load(OUT / 'features/train_target_ids.npy')))
    history = set(map(int, np.load(OUT / 'features/train_history_ids.npy')))
    return {name: paired(requests, rankings, reference, subset) for name, subset in {
        'overall': None, 'train_target_seen': target, 'train_history_only': history - target,
        'completely_unseen': set(map(int, assets.catalog)) - target - history,
        'with_image': set(map(int, assets.catalog[assets.available])),
        'without_image': set(map(int, assets.catalog[~assets.available]))}.items()}


def diagnostics(args):
    cuda_preflight(args.device)
    model, assets, digest = load_best(args.model, args.seed, args.device)
    source = OUT / 'validation' / f'{args.model}_seed{args.seed}'
    verified(source)
    metadata = json.loads((source / 'vector_metadata.json').read_text())
    if metadata['checkpoint_sha256'] != digest:
        raise RuntimeError('vector/query checkpoint mismatch')
    destination = OUT / 'diagnostics' / f'{args.model}_seed{args.seed}'
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    guard = Guard(args.validation_seconds * 3, OUT, args.max_rss_gib, disk_need_gib=1.)
    _, valid = load_temporal_frames(ROOT)
    requests, _ = proxy(assets, valid, count=1000)
    normal = pd.read_parquet(source / 'rankings.parquet').set_index('request_idx')
    normal_ranks = [list(map(int, normal.loc[r.request_idx].retrieved_top500)) for r in requests]
    baseline, _ = evaluate(requests, normal_ranks, assets, 'ID-on subset', guard)
    results = {'ID-on': baseline}
    tmp = destination / 'id_off.temporary.npy'
    for name, off in {'Item-ID-off': ('item',), 'User-ID-off': ('user',), 'All-ID-off': ('item', 'user')}.items():
        with model.disable_id(*off):
            if 'item' in off:
                vectors = (np.load(tmp, mmap_mode='r') if tmp.exists() else
                           encode_items(model, assets, np.arange(len(assets.catalog)), args.device, guard, tmp))
            else:
                vectors = np.load(source / 'item_vectors.f32.npy', mmap_mode='r')
            rankings, timing = retrieve(model, assets, requests, assets.catalog, args.device, guard, vectors)
            result, frame = evaluate(requests, rankings, assets, name, guard)
            result['timing'] = timing
            results[name] = result
            atomic_parquet(destination / f'{name}.parquet', frame)
            if 'item' in off:
                del vectors
    tmp.unlink()  # Explicit disposable diagnostic cache, never an official vector.
    results['scope'] = 'fixed validation subset; OOD diagnostic, not model selection or official quality'
    retrieved = np.concatenate([np.asarray(x, np.int64) for x in normal_ranks])
    train_target = np.load(OUT / 'features/train_target_ids.npy')
    train_history = np.load(OUT / 'features/train_history_ids.npy')
    results['retrieved_item_share'] = {
        'train_target_seen': float(np.isin(retrieved, train_target).mean()),
        'history_only': float((np.isin(retrieved, train_history) & ~np.isin(retrieved, train_target)).mean()),
        'completely_unseen': float((~np.isin(retrieved, np.union1d(train_target, train_history))).mean())}
    results['source_validation_marker'] = sha256(source / 'complete.json')
    atomic_json(destination / 'diagnostics.json', results)
    guard.check()
    complete(destination, list(destination.glob('*.json')) + list(destination.glob('*.parquet')))


def select():
    from .selection import choose_models
    scores, initial_hashes, proxy_hashes = {}, {}, {}
    markers = {}
    for name in MODELS:
        seed_list = [42, 43, 44] if name[0] in 'AB' else [42] + [s for s in (43, 44)
                    if (OUT / 'validation' / f'{name}_seed{s}/complete.json').exists()]
        scores[name] = {}
        for seed in seed_list:
            folder = OUT / 'validation' / f'{name}_seed{seed}'
            validation_marker = verified(folder)
            if (not compatible_source(validation_marker['source_hashes'])
                    or validation_marker['features_marker_sha256'] != sha256(OUT / 'features/complete.json')):
                raise RuntimeError('selection source/features drift')
            diagnostic_folder = OUT / 'diagnostics' / f'{name}_seed{seed}'
            verified(diagnostic_folder)
            if json.loads((diagnostic_folder / 'diagnostics.json').read_text())['source_validation_marker'] != sha256(folder / 'complete.json'):
                raise RuntimeError('selection ID-off diagnostic validation mismatch')
            training = OUT / 'training' / f'{name}_seed{seed}'
            verified(training)
            config = json.loads((training / 'config.json').read_text())
            initial_hashes.setdefault(seed, set()).add(config['initial_state_sha256'])
            proxy_hashes.setdefault(seed, set()).add((sha256(training / 'proxy_request_ids.npy'), sha256(training / 'proxy_candidate_ids.npy')))
            result = json.loads((folder / 'metrics.json').read_text())
            scores[name][seed] = result['metrics']['overall']['Recall@500']
            markers[f'{name}_seed{seed}'] = sha256(folder / 'complete.json')
    if any(len(v) != 1 for v in initial_hashes.values()) or any(len(v) != 1 for v in proxy_hashes.values()):
        raise AssertionError('same-seed initialization/proxy did not match')
    decision = choose_models(scores)
    if not decision['selection_ready']:
        atomic_json(OUT / 'selection_pending.json', {**decision, 'scores': scores,
                    'validation_markers': markers, 'test_opened': False})
        raise RuntimeError(f'T screening winner requires replication: {decision["required_T_replications"]}; '
                           'no final selection written, no routes authorized')
    path = OUT / 'selection.json'
    if path.exists():
        raise FileExistsError('final selection is immutable; do not replace route source')
    compute_comparisons()
    atomic_json(path, {**decision, 'scores': scores, 'test_opened': False,
                'validation_markers': markers, 'source_hashes': source_hashes(),
                'features_marker_sha256': sha256(OUT / 'features/complete.json')})


def compute_comparisons():
    _, valid = load_temporal_frames(ROOT)
    requests = grouped_requests(valid)
    assets = Assets('S')
    from .routes import read_rankings
    targets = set(map(int, np.load(OUT / 'features/train_target_ids.npy')))
    history = set(map(int, np.load(OUT / 'features/train_history_ids.npy')))
    subsets = {'overall': None, 'train_target_seen': targets, 'history_only': history - targets,
               'completely_unseen': set(map(int, assets.catalog)) - targets - history,
               'with_image': set(map(int, assets.catalog[assets.available])),
               'without_image': set(map(int, assets.catalog[~assets.available]))}
    comparisons = {}
    dependencies = {}
    for first, second in [('A1', 'A0'), ('B1', 'B0'), ('B0', 'T0'), ('B1', 'T1'), ('A0', 'B0'), ('A1', 'B1')]:
        seeds = [s for s in (42, 43, 44) if all(
            (OUT / 'validation' / f'{name}_seed{s}/complete.json').exists()
            for name in (first, second))]
        per_segment = {k: [] for k in subsets}
        for seed in seeds:
            a = read_rankings(OUT / 'validation' / f'{first}_seed{seed}', requests)
            b = read_rankings(OUT / 'validation' / f'{second}_seed{seed}', requests)
            for name in (first, second):
                folder = OUT / 'validation' / f'{name}_seed{seed}'
                dependencies[str((folder / 'complete.json').relative_to(OUT))] = sha256(folder / 'complete.json')
            for name, subset in subsets.items():
                per_segment[name].append(recall_values(requests, a, subset) - recall_values(requests, b, subset))
        comparisons[f'{first}-{second}'] = {}
        for name, differences in per_segment.items():
            stack = np.stack(differences)
            eligible = np.isfinite(stack).all(0)
            averaged = stack[:, eligible].mean(0)
            comparisons[f'{first}-{second}'][name] = {
                'request_mean_bootstrap': bootstrap_delta(averaged),
                'per_seed_delta': [float(v[eligible].mean()) if eligible.any() else None for v in stack],
                'seed_delta_std': float(stack[:, eligible].mean(1).std()) if eligible.any() else None,
                'seeds': seeds,
                'interpretation': 'validation-selected evidence; request CI does not quantify initialization uncertainty'}
    atomic_json(OUT / 'ablation_bootstrap.json', comparisons)
    atomic_json(OUT / 'ablation_bootstrap_complete.json', {
        'complete': True, 'test_opened': False, 'source_hashes': source_hashes(),
        'bootstrap_sha256': sha256(OUT / 'ablation_bootstrap.json'),
        'validation_markers': dependencies})


def report():
    from .reporting import build_report
    build_report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', default='plan', choices=['plan', 'audit', 'self-check', 'smoke', 'train',
                        'validate', 'diagnostics', 'select', 'routes', 'report'])
    parser.add_argument('--model', choices=list(MODELS), default='B0')
    parser.add_argument('--seed', type=int, choices=[42, 43, 44], default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--confirm-run', action='store_true')
    parser.add_argument('--confirm-image-seeds', action='store_true', help='explicit extra T43/44 image attribution replication')
    parser.add_argument('--resume', action='store_true', help='resume incomplete validation/routes only, never overwrite completed assets')
    parser.add_argument('--train-seconds', type=int, default=3600)
    parser.add_argument('--validation-seconds', type=int, default=1200)
    parser.add_argument('--max-rss-gib', type=float, default=16)
    args = parser.parse_args()
    if args.stage == 'plan':
        print(json.dumps({'matrix': MODELS, 'protocol': PROTOCOL,
              'order': 'review → synthetic self-check → audit → six smoke → train/validate/diagnostics → A/B seeds43/44 → select → routes → report',
              'test': 'no code path', 'execution_authorized_this_turn': False}, ensure_ascii=False, indent=2))
        return
    if args.stage == 'self-check':
        from .self_check import run_checks
        print(json.dumps(run_checks(), ensure_ascii=False, indent=2))
        return
    if not args.confirm_run:
        parser.error('data/product stages require --confirm-run; review first')
    if args.model[0] == 'T' and args.seed != 42 and not args.confirm_image_seeds:
        parser.error('T seeds43/44 require --confirm-image-seeds (selection replication or image attribution)')
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / ('locks/full_catalog.lock' if args.stage in ('validate', 'diagnostics', 'routes')
                  else f'locks/{args.stage}_{args.model}_seed{args.seed}.lock')
    with stage_lock(lock):
        if args.stage == 'audit':
            build_features(Guard(1800, OUT, args.max_rss_gib, disk_need_gib=1.))
        elif args.stage == 'smoke':
            from .self_check import run_checks
            run_checks()
            execute_training(args, smoke=True)
        elif args.stage == 'train':
            execute_training(args)
        elif args.stage == 'validate':
            full_validation(args)
        elif args.stage == 'diagnostics':
            diagnostics(args)
        elif args.stage == 'select':
            select()
        elif args.stage == 'routes':
            from .routes import run_routes
            run_routes(args)
        else:
            report()


if __name__ == '__main__':
    main()
