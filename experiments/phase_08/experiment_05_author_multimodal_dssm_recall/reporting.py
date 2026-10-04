"""Completion-checked report: quality, ID diagnostics, resources, seed scope."""
import json

import numpy as np
import pandas as pd

from .artifacts import atomic_json, verified, sha256
from .config import OUT, ROOT, H2, MODELS
from .evaluation import recall_values, bootstrap_delta


def read_json(path):
    return json.loads(path.read_text())


def percent(value):
    return 'N/A' if value is None or not np.isfinite(value) else f'{value:.4%}'


def number(value):
    return 'N/A' if value is None else f'{value:.4f}'


def ci_label(row):
    ci = row.get('ci95')
    if not ci:
        return 'N/A（无 eligible request）'
    return f'Δ={row["delta"]*100:+.4f}pp，95%CI=[{ci[0]*100:+.4f},{ci[1]*100:+.4f}]pp'


def load_official(name, seed):
    from .run import compatible_source
    folder = OUT / 'validation' / f'{name}_seed{seed}'
    marker = verified(folder)
    if not compatible_source(marker['source_hashes']) or marker['features_marker_sha256'] != sha256(OUT / 'features/complete.json'):
        raise RuntimeError('validation not bound to current source/features')
    return read_json(folder / 'metrics.json'), folder


def route_stability(route_data):
    """Same named configuration across all three seeds; average request deltas."""
    if set(route_data) != {42, 43, 44}:
        return {'scope': 'single/partial-seed screening; no stable-gain claim',
                'completed_seeds': sorted(route_data), 'configurations': {}}
    from .data import Assets
    from experiments.phase_06.experiment_01_id_two_tower_retrieval.data import load_temporal_frames, grouped_requests
    assets = Assets('S')
    _, valid = load_temporal_frames(ROOT)
    requests = grouped_requests(valid)
    target = set(map(int, np.load(OUT / 'features/train_target_ids.npy')))
    history = set(map(int, np.load(OUT / 'features/train_history_ids.npy')))
    subsets = {'overall': None, 'train_target_seen': target, 'history_only': history - target,
               'completely_unseen': set(map(int, assets.catalog)) - target - history,
               'with_image': set(map(int, assets.catalog[assets.available])),
               'without_image': set(map(int, assets.catalog[~assets.available]))}
    common = set.intersection(*(set(data['analyses']) for data in route_data.values()))
    result = {}
    for name in sorted(common):
        per_segment = {s: [] for s in subsets}
        absolute = []
        for seed in (42, 43, 44):
            folder = OUT / 'routes' / f'seed{seed}'
            frames = [pd.read_parquet(folder / f'{route}.parquet').set_index('request_idx')
                      for route in (name, 'H2')]
            rankings = []
            for frame in frames:
                selected = []
                for r in requests:
                    row = frame.loc[r.request_idx]
                    if set(map(int, row.ground_truth)) != set(r.ground_truth):
                        raise AssertionError('three-seed route GT drift')
                    selected.append(list(map(int, row.retrieved_top500)))
                rankings.append(selected)
            absolute.append(route_data[seed]['results'][name]['metrics']['overall']['Recall@500'])
            for segment, subset in subsets.items():
                per_segment[segment].append(recall_values(requests, rankings[0], subset)
                                            - recall_values(requests, rankings[1], subset))
        result[name] = {'R500_mean': float(np.mean(absolute)), 'R500_std': float(np.std(absolute)), 'segments': {}}
        for segment, arrays in per_segment.items():
            stack = np.stack(arrays)
            good = np.isfinite(stack).all(0)
            deltas = [float(v[good].mean()) for v in stack] if good.any() else [None] * 3
            result[name]['segments'][segment] = {
                'per_seed_delta': deltas, 'seed_delta_std': float(np.std(deltas)) if good.any() else None,
                'request_mean_bootstrap': bootstrap_delta(stack[:, good].mean(0))}
        overall = result[name]['segments']['overall']
        unseen = result[name]['segments']['completely_unseen']['request_mean_bootstrap']['ci95']
        without = result[name]['segments']['without_image']['request_mean_bootstrap']['ci95']
        overall_ci = overall['request_mean_bootstrap']['ci95']
        result[name]['evidence'] = {
            'overall_CI_supports_gain': bool(overall_ci and overall_ci[0] > 0),
            'overall_seed_directions_positive': all(d is not None and d > 0 for d in overall['per_seed_delta']),
            'unseen_CI_supports_degradation': bool(unseen and unseen[1] < 0),
            'no_image_CI_supports_degradation': bool(without and without[1] < 0),
            'noninferiority_not_proven': True,
            'scope': 'validation evidence only; CI reflects request variation, not sufficient initialization uncertainty'}
    return {'scope': 'same configuration across seeds42/43/44, request-averaged paired bootstrap',
            'completed_seeds': [42, 43, 44], 'configurations': result}


def build_report():
    from .run import h2_rankings, source_hashes
    lines = ['# Phase 8-05：作者图文 DSSM 与 ID Dropout 消融', '',
             '只使用 validation，test_opened=false。F 是时间未验证的作者字段参考，不是无泄漏/上线证据。', '',
             '## 正式模型质量与同 seed H2 对照', '',
             '| 模型 | seed | 证据范围 | R@100 | R@500 | MRR@100 | seen | history-only | completely-unseen |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    recovery = OUT / 'debug/numerical_recovery/runtime_patch.json'
    if recovery.exists():
        lines[3:3] = ['数值运行修复：保留默认AMP初始scale；溢出更新由GradScaler拒绝，'
                      '降scale后重放同一batch及ID mask，不跳过样本。已完成旧产物通过'
                      '精确源码hash迁移校验保留，不重写marker。',
                      f'修复记录 SHA-256：`{sha256(recovery)}`。', '']
    rows, folders = {}, {}
    dependencies = {}
    for name in MODELS:
        for seed in (42, 43, 44):
            folder = OUT / 'validation' / f'{name}_seed{seed}'
            if not (folder / 'complete.json').exists():
                continue
            value, folder = load_official(name, seed)
            rows[(name, seed)], folders[(name, seed)] = value, folder
            dependencies[str((folder / 'complete.json').relative_to(OUT))] = sha256(folder / 'complete.json')
            metric, status = value['metrics']['overall'], value['phase6_item_status']
            scope = 'F快照参考' if name[0] == 'A' else '安全协议；seed独立结果'
            cells = [metric['Recall@100'], metric['Recall@500'], metric['MRR@100']] + [status[s]['Recall@500'] for s in
                       ('train_target_seen', 'train_history_only', 'completely_unseen')]
            lines.append(f'| {name} | {seed} | {scope} | ' + ' | '.join(map(percent, cells)) + ' |')
    if rows:
        # This validates H2 completion and audit-pinned reference hashes, without
        # reading test or rerunning retrieval.
        for seed in sorted({s for _, s in rows}):
            h2_rankings([], seed)
            value = read_json(H2 / f'validation/h2_256_seed{seed}.json')
            metric, status = value['metrics']['overall'], value['phase6_item_status']
            cells = [metric['Recall@100'], metric['Recall@500'], metric['MRR@100']] + [status[s]['Recall@500'] for s in
                       ('train_target_seen', 'train_history_only', 'completely_unseen')]
            lines.append(f'| H2-256 | {seed} | 已有validation；不重跑 | ' + ' | '.join(map(percent, cells)) + ' |')
        lines += ['', '### 跨 seed 摘要', '']
        for name in MODELS:
            seeds = [s for s in (42, 43, 44) if (name, s) in rows]
            if seeds:
                scores = [rows[(name, s)]['metrics']['overall']['Recall@500'] for s in seeds]
                lines.append(f'- {name}：seeds={seeds}，mean={percent(np.mean(scores))}，std={percent(np.std(scores))}；'
                             + ('三seed结果（F仍仅参考）。' if len(seeds) == 3 else '仅单/部分seed初筛，不能作稳定结论。'))
    lines += ['', '## 分层与 cold 保护', '',
              '旧cold_item若eligible=0记N/A；completely-unseen是本地temporal冷启动诊断，不用test补空切片。', '',
              '| 模型/seed | 有图 R500 | 无图 R500 | warm-item | cold-item | warm-user | cold-user | completely-unseen | Top500覆盖 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for (name, seed), value in rows.items():
        segments = value['segments']
        cells = [segments['with_image']['Recall@500'], segments['without_image']['Recall@500'],
                 value['metrics']['warm_item']['Recall@500'], value['metrics']['cold_item']['Recall@500'],
                 value['metrics']['warm_user']['Recall@500'], value['metrics']['cold_user']['Recall@500'],
                 value['phase6_item_status']['completely_unseen']['Recall@500'], value['top500_coverage']]
        lines.append(f'| {name}/{seed} | ' + ' | '.join(map(percent, cells)) + ' |')
        lines.append(f'\n{name}/{seed} vs 同seed H2：' + '；'.join(
            f'{s} {ci_label(v)}' for s, v in value['comparison_H2_same_seed'].items()) + '\n')
    lines += ['', '## ID-off 与 ID 捷径诊断（OOD，不参与选型）', '',
              '| 模型/seed | 模式 | R500 | 相对ID-on Δpp | seen | history-only | unseen | warm-user | cold-user |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    diagnostics = {}
    for key in rows:
        name, seed = key
        folder = OUT / 'diagnostics' / f'{name}_seed{seed}'
        if not (folder / 'complete.json').exists():
            lines.append(f'\n{name}/{seed}：诊断未完成，不判断ID捷径。\n')
            continue
        verified(folder)
        diagnostic = read_json(folder / 'diagnostics.json')
        if diagnostic['source_validation_marker'] != sha256(folders[key] / 'complete.json'):
            raise RuntimeError('ID-off diagnostic belongs to different validation')
        dependencies[str((folder / 'complete.json').relative_to(OUT))] = sha256(folder / 'complete.json')
        diagnostics[key] = diagnostic
        on = diagnostic['ID-on']['metrics']['overall']['Recall@500']
        for mode in ('ID-on', 'Item-ID-off', 'User-ID-off', 'All-ID-off'):
            value = diagnostic[mode]
            m, status = value['metrics'], value['phase6_item_status']
            score = m['overall']['Recall@500']
            cells = [status[s]['Recall@500'] for s in ('train_target_seen', 'train_history_only', 'completely_unseen')]
            cells += [m[s]['Recall@500'] for s in ('warm_user', 'cold_user')]
            lines.append(f'| {name}/{seed} | {mode} | {percent(score)} | {(score-on)*100:+.4f} | '
                         + ' | '.join(map(percent, cells)) + ' |')
        lines.append(f'\n{name}/{seed}：检索候选组占比={diagnostic["retrieved_item_share"]}。'
                     'ID-off损失只说明推理依赖；不能据此证明或排除捷径。\n')
    lines += ['', '## 训练尺度、梯度与 false-negative 审计', '']
    resource_rows = []
    for key in rows:
        name, seed = key
        folder = OUT / 'training' / f'{name}_seed{seed}'
        training_marker = verified(folder)
        dependencies[str((folder / 'complete.json').relative_to(OUT))] = sha256(folder / 'complete.json')
        curves = read_json(folder / 'training_curves.json')
        config = read_json(folder / 'config.json')
        best = read_json(folder / 'best.json')
        value = rows[key]
        lines.append(f'### {name}/seed{seed}' + '\n\n'
                     f'best epoch={best["epoch"]}；parameters={config["parameters"]}，ID parameters={config["id_parameters"]}。')
        for epoch in curves:
            lines.append(f'- epoch{epoch["epoch"]}：loss={number(epoch["loss"])}；'
                         f'cosine/输入尺度={epoch["scales"]}；false-negative rates={epoch["false_negative_rates"]}；'
                         f'整路dropout={epoch["dropout"]}。')
            gradients = epoch['gradient_norms_mean']
            groups = {prefix: float(sum(v for k, v in gradients.items() if k.startswith(prefix)))
                      for prefix in ('seq_input_proj', 'seq_gru', 'seq_attn', 'u_emb_', 'i_emb_', 'user_mlp', 'item_mlp')}
            lines.append(f'  梯度范数均值之和（按分支、非全局gradient norm）={groups}。')
        timing = value['timing']
        resource_rows.append((name, seed, training_marker.get('stage_wall_seconds'), sum(e['seconds'] for e in curves),
                              np.mean([e['data_wait_fraction'] for e in curves]),
                              value.get('checkpoint_bytes'), value['item_vector_bytes'],
                              timing['item_encoding_seconds'], timing.get('query_encoding_seconds'), timing['index_build_seconds'],
                              timing['query_search_seconds'], timing['mean_search_latency_seconds'],
                              value.get('gpu_peak_memory_bytes'), value.get('process_tree_peak_rss_gib')))
    lines += ['', '## 资源成本（训练与 full validation 分开）', '',
              '训练秒数是train_epoch时间之和，不含proxy；每epoch proxy额外时间保存在training_curves.json，不能当总wall time。', '',
              '| 模型/seed | 训练阶段wall秒 | train秒 | data-wait | checkpoint bytes（所有epoch） | vectors bytes | 全库编码秒 | query编码秒 | index秒 | search秒 | latency秒/query | GPU peak bytes | RSS GiB |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, seed, *values in resource_rows:
        lines.append(f'| {name}/{seed} | ' + ' | '.join(map(number, values)) + ' |')
    lines += ['', '## 受控消融与三 seed 配对结果', '']
    bootstrap_path = OUT / 'ablation_bootstrap.json'
    if bootstrap_path.exists():
        marker = read_json(OUT / 'ablation_bootstrap_complete.json')
        if (not marker.get('complete') or marker.get('test_opened') is not False
                or marker['bootstrap_sha256'] != sha256(bootstrap_path)
                or marker['source_hashes'] != source_hashes()):
            raise RuntimeError('ablation bootstrap incomplete or changed')
        for relative, digest in marker['validation_markers'].items():
            if sha256(OUT / relative) != digest:
                raise RuntimeError('ablation source validation changed')
        dependencies['ablation_bootstrap_complete.json'] = sha256(OUT / 'ablation_bootstrap_complete.json')
        bootstrap = read_json(bootstrap_path)
        for contrast, segments in bootstrap.items():
            for segment, value in segments.items():
                lines.append(f'- {contrast}/{segment}：seeds={value["seeds"]}；'
                             f'{ci_label(value["request_mean_bootstrap"])}；per-seed Δ={value["per_seed_delta"]}；'
                             f'seed Δstd={number(value["seed_delta_std"])}。')
        lines.append('F−S反映未验证快照数值的影响，不能解释为无泄漏增益。T仅一seed时图片归因只作初筛。')
    else:
        lines.append('尚未完成受控bootstrap，不下正式显著性结论。')
    lines += ['', '## 双路与固定预算（原180排序补位，不是深层重排）', '']
    route_data = {}
    for seed in (42, 43, 44):
        folder = OUT / 'routes' / f'seed{seed}'
        if not (folder / 'complete.json').exists():
            continue
        verified(folder)
        data = read_json(folder / 'summary.json')
        if data['selection_sha256'] != sha256(OUT / 'selection.json'):
            raise RuntimeError('route source selection mismatch')
        route_data[seed] = data
        dependencies[str((folder / 'complete.json').relative_to(OUT))] = sha256(folder / 'complete.json')
        lines += ['', f'### Routes seed{seed}：单 seed 筛选（F-reference仅参考）', '',
                  '| 路线 | R100 | R500 | MRR100 | unseen R500 | Top500覆盖率 |',
                  '|---|---:|---:|---:|---:|---:|']
        for name, value in data['results'].items():
            metric = value['metrics']['overall']
            values = [metric['Recall@100'], metric['Recall@500'], metric['MRR@100'],
                      value['phase6_item_status']['completely_unseen']['Recall@500'], value['top500_coverage']]
            lines.append(f'| {name} | ' + ' | '.join(map(percent, values)) + ' |')
        lines.append(f'\nKNN成本、原180覆盖与补位规则：{data["timings"]}\n')
        lines.append(f'\n多路RSS峰值={number(data.get("process_tree_peak_rss_gib"))}GiB，'
                     f'GPU峰值bytes={data.get("gpu_peak_memory_bytes")}，cache bytes={data.get("knn_cache_bytes")}。\n')
        for name, value in data['analyses'].items():
            lines.append(f'- {name}：新增/挤掉/净增={value["hits"]["new_positive_hits"]}/'
                         f'{value["hits"]["displaced_positive_hits"]}/{value["hits"]["net_positive_interaction_gain"]}；'
                         + '；'.join(f'{segment}: {ci_label(row)}' for segment, row in value['paired_vs_H2'].items()))
        for pair, value in data['pairwise'].items():
            lines.append(f'- {pair}：overlap/Jaccard={value["candidate_overlap_mean"]:.2f}/{value["jaccard_mean"]:.4f}；'
                         f'独有positive interactions={value["first_only_positive_interactions"]}/{value["second_only_positive_interactions"]}；'
                         f'union oracle={percent(value["union_oracle_request_macro_recall"])}（非固定预算）。')
    stability = route_stability(route_data)
    lines += ['', '### 多路稳定性与 cold 风险', '', f'范围：{stability["scope"]}。']
    for name, value in stability['configurations'].items():
        lines.append(f'- {name}：R500 mean/std={percent(value["R500_mean"])}/{percent(value["R500_std"])}；'
                     f'evidence={value["evidence"]}。')
        for segment, data in value['segments'].items():
            lines.append(f'  {segment}: {ci_label(data["request_mean_bootstrap"])}；'
                         f'seed deltas={data["per_seed_delta"]}。')
    lines += ['', '## 决策解释与停止边界', '',
              '只有单seed多路时，不宣称稳定净增益；同一融合配置需42/43/44对应H2对照。',
              'overall CI为正不自动GO：必须同时检查completely-unseen、无图和各seed方向；CI支持退化的切片明确列为风险。'
              '未预注册non-inferiority margin时，不宣称cold非劣效已证明。',
              'ID-off、cosine、尺度和梯度是机制线索，不能仅凭loss下降或ID-off不降证明已学会图文语义。',
              'F仅作者字段快照参考，S是保守数值协议，类别/内容快照时间假设仍需披露。',
              'DSSM128与H2-256横向比较同时改变架构/训练/维度，不解释为单一结构因果差异。',
              '不读取test，不自动替换H2，不进入粗排/精排。']
    if not rows:
        lines += ['', '当前只有代码实现与合成自检，无真实数据实验结果；等待用户review。']
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / 'report_evidence.json', {'source_hashes': source_hashes(),
                'dependencies': dependencies, 'route_stability': stability,
                'test_opened': False, 'formal_model_count': len(rows)})
    tmp = OUT / 'summary.building.md'
    tmp.write_text('\n'.join(lines) + '\n')
    tmp.replace(OUT / 'summary.md')
