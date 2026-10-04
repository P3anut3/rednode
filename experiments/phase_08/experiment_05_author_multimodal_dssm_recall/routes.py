"""Same-vector DSSM-I2I and independent frozen BGE-I2I, validation only."""
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from experiments.phase_06.experiment_01_id_two_tower_retrieval.data import load_temporal_frames, grouped_requests
from .artifacts import Guard, atomic_json, atomic_parquet, complete, verified, sha256, cuda_preflight
from .config import OUT, ROOT
from .data import Assets
from .evaluation import evaluate
from .retrieval import ResidentExact, validate_rankings


def quota_merge(main, secondary, secondary_quota):
    """Take prefixes at fixed 500 budget, globally dedup, main deterministic backfill."""
    merged = []
    seen = set()
    for stream in (main[:500 - secondary_quota], secondary[:secondary_quota], main, secondary):
        for note in stream:
            note = int(note)
            if note not in seen:
                seen.add(note)
                merged.append(note)
                if len(merged) == 500:
                    return merged
    return merged


def rrf(main, secondary):
    scores = {}
    for stream in (main, secondary):
        for rank, note in enumerate(stream, 1):
            note = int(note)
            scores[note] = scores.get(note, 0.) + 1. / (60 + rank)
    return sorted(scores, key=lambda n: (-scores[n], n))[:500]


def append_deeper_candidates(author180, deeper, topk=500):
    """Preserve author180 order, append deep-only candidates only on underflow."""
    if len(author180) >= topk:
        return list(author180[:topk])
    output = list(author180)
    seen = set(output)
    for note in deeper:
        note = int(note)
        if note not in seen:
            seen.add(note)
            output.append(note)
            if len(output) == topk:
                break
    return output


def read_rankings(folder, requests):
    verified(folder)
    frame = pd.read_parquet(Path(folder) / 'rankings.parquet').set_index('request_idx')
    result = []
    for r in requests:
        row = frame.loc[r.request_idx]
        if set(row.ground_truth) != set(r.ground_truth):
            raise AssertionError('ranking ground truth differs')
        result.append(list(map(int, row.retrieved_top500)))
    return result


def knn_cache(name, vectors, vector_hash, assets, history_rows, depth, args, guard):
    folder = OUT / 'knn' / name
    contract = {'vector_sha256': vector_hash, 'mapping_sha256': sha256(OUT / 'features/note_ids.npy'),
                'depth': depth, 'history_rows_sha256': __import__('hashlib').sha256(history_rows.tobytes()).hexdigest(),
                'query_rows': len(history_rows), 'backend': 'GPU resident FP32 exact', 'chunk_rows': 512}
    folder.mkdir(parents=True, exist_ok=True)
    meta = folder / 'contract.json'
    if meta.exists() and json.loads(meta.read_text()) != contract:
        raise RuntimeError('cannot reuse KNN cache with different vector/mapping/depth')
    if not meta.exists():
        atomic_json(meta, contract)
    chunks = []
    index = None
    try:
        for offset in range(0, len(history_rows), 512):
            guard.check()
            part = folder / f'chunk_{offset // 512:05d}'
            if (part / 'complete.json').exists():
                verified(part)
                chunks.append(part)
                continue
            if index is None:
                index = ResidentExact(vectors, args.device, guard)
            part.mkdir(parents=True, exist_ok=True)
            query_rows = history_rows[offset:offset + 512]
            score, row, seconds = index.search(vectors[query_rows], depth, query_batch=32)
            for filename, array in [('scores.npy', score), ('rows.npy', row), ('query_rows.npy', query_rows)]:
                tmp = part / (filename + '.tmp')
                with tmp.open('wb') as handle:
                    np.save(handle, array)
                os.replace(tmp, part / filename)
            complete(part, list(part.glob('*.npy')), {'source_contract_sha256': sha256(meta),
                     'search_seconds': seconds})
            chunks.append(part)
            print(f'{name} KNN cached: {offset + len(query_rows)}/{len(history_rows)}', flush=True)
    finally:
        if index is not None:
            index.close()
    # Read-only per-chunk mappings: no giant string/dict candidate object cache.
    score_arrays, row_arrays, lookup = [], [], {}
    for chunk_index, part in enumerate(chunks):
        marker = verified(part)
        if marker['source_contract_sha256'] != sha256(meta):
            raise RuntimeError('KNN chunk not bound to current vectors')
        scores = np.load(part / 'scores.npy', mmap_mode='r')
        rows = np.load(part / 'rows.npy', mmap_mode='r')
        for i, query_row in enumerate(np.load(part / 'query_rows.npy')):
            lookup[int(query_row)] = (chunk_index, i)
        score_arrays.append(scores)
        row_arrays.append(rows)
    complete(folder, [meta] + [p / 'complete.json' for p in chunks], {'contract': contract})
    return score_arrays, row_arrays, lookup


def aggregate(requests, assets, cache, depth, guard):
    score_arrays, row_arrays, lookup = cache
    output = []
    for request_index, r in enumerate(requests):
        if request_index % 100 == 0:
            guard.check()
        candidate_parts, score_parts = [], []
        # Author reverses sequence for 1/(1+idx). Our canonical order is last recent.
        for idx, note in enumerate(reversed([int(v) for v in r.history if int(v) >= 0][-20:])):
            row = int(assets.lookup([note])[0])
            if row not in lookup:
                continue
            chunk, query = lookup[row]
            candidate_parts.append(np.asarray(row_arrays[chunk][query, :depth], np.int64))
            score_parts.append(np.asarray(score_arrays[chunk][query, :depth], np.float64) / (1 + idx))
        if not candidate_parts:
            output.append([])
            continue
        rows, inverse = np.unique(np.concatenate(candidate_parts), return_inverse=True)
        score = np.bincount(inverse, weights=np.concatenate(score_parts), minlength=len(rows))
        ids = assets.catalog[rows]
        allowed = ~np.isin(ids, np.asarray(r.history, np.int64))
        ids, score = ids[allowed], score[allowed]
        order = np.lexsort((ids, -score))[:500]
        output.append(ids[order].astype(int).tolist())
    validate_rankings(requests, output, assets.catalog, strict=False)
    return output


def overlap(requests, first, second, merged=None, guard=None):
    jac, overlap_counts, first_hits, second_hits, both_hits = [], [], set(), set(), set()
    first_items, second_items = set(), set()
    first_requests, second_requests = set(), set()
    added, displaced = 0, 0
    union_ranking = []
    for i, (r, a, b) in enumerate(zip(requests, first, second)):
        if guard and i % 250 == 0:
            guard.check()
        aset, bset, truth = set(a), set(b), set(r.ground_truth)
        common = aset & bset
        overlap_counts.append(len(common))
        jac.append(len(common) / max(1, len(aset | bset)))
        ah, bh = aset & truth, bset & truth
        first_hits.update((r.request_idx, n) for n in ah)
        second_hits.update((r.request_idx, n) for n in bh)
        both_hits.update((r.request_idx, n) for n in ah & bh)
        first_items |= ah
        second_items |= bh
        if ah:
            first_requests.add(r.request_idx)
        if bh:
            second_requests.add(r.request_idx)
        union_ranking.append(len((aset | bset) & truth) / len(truth))
        if merged is not None:
            new = set(merged[i]) & truth
            added += len(new - ah)
            displaced += len(ah - new)
    return {'candidate_overlap_mean': float(np.mean(overlap_counts)), 'jaccard_mean': float(np.mean(jac)),
            'first_only_positive_interactions': len(first_hits - second_hits),
            'second_only_positive_interactions': len(second_hits - first_hits),
            'both_positive_interactions': len(both_hits),
            'first_only_unique_positive_items': len(first_items - second_items),
            'second_only_unique_positive_items': len(second_items - first_items),
            'first_only_hit_requests': len(first_requests - second_requests),
            'second_only_hit_requests': len(second_requests - first_requests),
            'union_oracle_request_macro_recall': float(np.mean(union_ranking)),
            'oracle_is_not_fixed_budget': True,
            'new_positive_hits': added, 'displaced_positive_hits': displaced,
            'net_positive_interaction_gain': added - displaced}


def run_routes(args):
    from .run import h2_rankings, paired_slices, source_hashes
    cuda_preflight(args.device)
    selection = json.loads((OUT / 'selection.json').read_text())
    if not selection.get('selection_ready') or selection.get('source_hashes') != source_hashes():
        raise RuntimeError('final three-seed selection missing or source changed')
    if selection.get('features_marker_sha256') != sha256(OUT / 'features/complete.json'):
        raise RuntimeError('selection feature contract changed')
    for name, digest in selection['validation_markers'].items():
        if sha256(OUT / 'validation' / name / 'complete.json') != digest:
            raise RuntimeError('selection validation changed')
    model_name = selection['safe_model']
    reference_model = selection['reference_model']
    assets = Assets('S')
    _, valid = load_temporal_frames(ROOT)
    requests = grouped_requests(valid)
    folder = OUT / 'routes' / f'seed{args.seed}'
    if folder.exists() and (folder / 'complete.json').exists():
        verified(folder)
        raise FileExistsError('completed routes immutable')
    if folder.exists() and not args.resume:
        raise FileExistsError('routes already exists/incomplete; do not mix formal runs')
    folder.mkdir(parents=True, exist_ok=args.resume)
    guard = Guard(7200, OUT, args.max_rss_gib, disk_need_gib=4.)
    d_folder = OUT / 'validation' / f'{model_name}_seed{args.seed}'
    vector_meta = json.loads((d_folder / 'vector_metadata.json').read_text())
    routes = {'H2': h2_rankings(requests, args.seed)[0],
              'D': read_rankings(d_folder, requests),
              'F-reference': read_rankings(OUT / 'validation' / f'{reference_model}_seed{args.seed}', requests)}
    all_history = np.unique(np.concatenate([assets.lookup([int(v) for v in r.history if int(v) >= 0][-20:])
                                           for r in requests]))
    all_history = all_history[all_history >= 0]
    # One shared KNN depth supports both author180 and coverage-based overfetch.
    depth = min(len(assets.catalog), 600 + max(len(set(r.history)) for r in requests))
    timings = {}
    for name, vectors, digest in [
        ('DI', np.load(d_folder / 'item_vectors.f32.npy', mmap_mode='r'), vector_meta['vectors_sha256']),
        ('C', assets.text, sha256(__import__('experiments.phase_07.experiment_01_feature_hybrid_two_tower.config',
                                          fromlist=['BGE_PATH']).BGE_PATH))]:
        started = time.monotonic()
        cache_name = f'{name}_{model_name}_seed{args.seed}' if name == 'DI' else 'C_frozen_BGE_shared'
        cache = knn_cache(cache_name, vectors, digest, assets, all_history, depth, args, guard)
        routes[name + '-author180'] = aggregate(requests, assets, cache, 180, guard)
        routes[name + '-deep-rerank'] = aggregate(requests, assets, cache, depth, guard)
        routes[name] = [append_deeper_candidates(base, deep) for base, deep in zip(
                       routes[name + '-author180'], routes[name + '-deep-rerank'])]
        timings[name] = {'elapsed_seconds': time.monotonic() - started, 'knn_depth': depth,
                         'overfetch_policy': 'preserve author180 prefix; append deep-only candidates on underflow',
                         'deep_rerank_is_separate_ablation': True,
                         'prefix_preserved_all_requests': all(main[:len(base)] == base for main, base in zip(
                             routes[name], routes[name + '-author180'])),
                         'author180_underfilled_request_rate': float(np.mean([len(x) < 500 for x in routes[name + '-author180']]))}
    results, analyses = {}, {}
    for name, ranking in routes.items():
        result, frame = evaluate(requests, ranking, assets, name, guard)
        results[name] = result
        atomic_parquet(folder / f'{name}.parquet', frame)
    for first, second in [('D', 'DI'), ('D', 'C'), ('H2', 'C')]:
        # Preregister both directions; do not add arbitrary new ratios.
        orientations = [(first, second)]
        if results[second]['metrics']['overall']['Recall@500'] > results[first]['metrics']['overall']['Recall@500']:
            orientations.append((second, first))
        for main, auxiliary in orientations:
            for quota in (50, 100, 150):
                name = f'{main}+{auxiliary}_{500-quota}_{quota}'
                merged = [quota_merge(a, b, quota) for a, b in zip(routes[main], routes[auxiliary])]
                validate_rankings(requests, merged, assets.catalog)
                results[name], frame = evaluate(requests, merged, assets, name, guard)
                analyses[name] = {'hits': overlap(requests, routes[main], routes[auxiliary], merged, guard),
                                  'paired_vs_primary': paired_slices(requests, merged, routes[main], assets),
                                  'paired_vs_H2': paired_slices(requests, merged, routes['H2'], assets)}
                atomic_parquet(folder / f'{name}.parquet', frame)
            name = f'{main}+{auxiliary}_RRF60'
            merged = [rrf(a, b) for a, b in zip(routes[main], routes[auxiliary])]
            validate_rankings(requests, merged, assets.catalog)
            results[name], frame = evaluate(requests, merged, assets, name, guard)
            analyses[name] = {'hits': overlap(requests, routes[main], routes[auxiliary], merged, guard),
                              'paired_vs_primary': paired_slices(requests, merged, routes[main], assets),
                              'paired_vs_H2': paired_slices(requests, merged, routes['H2'], assets)}
            atomic_parquet(folder / f'{name}.parquet', frame)
    pairwise = {f'{a}|{b}': overlap(requests, routes[a], routes[b], guard=guard)
                for i, a in enumerate(('H2', 'D', 'DI', 'C')) for b in ('H2', 'D', 'DI', 'C')[i + 1:]}
    # Deep reranking and underfilled author180 are diagnostics, never select
    # their results as the official route/fusion configuration.
    eligible = [name for name in results if '-author180' not in name and '-deep-rerank' not in name
                and name != 'F-reference']
    best = max(eligible, key=lambda name: results[name]['metrics']['overall']['Recall@500'])
    atomic_json(folder / 'summary.json', {'results': results, 'analyses': analyses,
                'pairwise': pairwise, 'timings': timings, 'validation_best': best,
                'process_tree_peak_rss_gib': guard.peak_rss / 2**30,
                'gpu_peak_memory_bytes': __import__('torch').cuda.max_memory_allocated(args.device),
                'knn_cache_bytes': {name: sum(p.stat().st_size for p in (OUT / 'knn' / cache_name).rglob('*') if p.is_file())
                    for name, cache_name in [('DI', f'DI_{model_name}_seed{args.seed}'), ('C', 'C_frozen_BGE_shared')]},
                'seed': args.seed, 'decision_scope': 'single-seed validation screening only; stable gain requires same configuration seeds42/43/44',
                'safe_model': model_name, 'reference_model': reference_model,
                'selection_sha256': sha256(OUT / 'selection.json'), 'test_opened': False})
    guard.check()
    complete(folder, list(folder.glob('*.parquet')) + [folder / 'summary.json'])
