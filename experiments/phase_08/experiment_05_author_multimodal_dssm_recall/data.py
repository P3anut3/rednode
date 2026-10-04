"""Two feature protocols, same slots; canonical/mmap and target-free history."""
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from experiments.phase_07.experiment_01_feature_hybrid_two_tower.config import (
    BGE_PATH, NOTE_IDS_PATH, TRAIN_PATH, VALID_PATH)
from experiments.phase_06.experiment_01_id_two_tower_retrieval.data import (
    load_temporal_frames, grouped_requests)
from experiments.phase_08.experiment_02_image_hybrid_recall.data import (
    image_paths, verify_image_assets)
from .artifacts import atomic_json, complete, verified, sha256
from .config import (ROOT, OUT, H2, ITEM_BASIC, ITEM_COUNTS, ITEM_NUMERIC, USER_NUMERIC,
                     USER_CATEGORICAL, ITEM_CATEGORICAL, RATE_INPUTS, PROTOCOL)


def category(value):
    text = str(value).strip()
    return '<MISSING>' if text.lower() in {'', 'none', 'nan', 'null', '<na>'} else text


def rows_for(ids, catalog):
    """Works with unsorted canonical mappings; -1 means truly outside catalog."""
    order = np.argsort(catalog)
    sorted_ids = np.asarray(catalog)[order]
    flat = np.asarray(ids, dtype=np.int64)
    pos = np.searchsorted(sorted_ids, flat)
    safe = np.minimum(pos, len(sorted_ids) - 1)
    return np.where((pos < len(sorted_ids)) & (sorted_ids[safe] == flat), order[safe], -1)


def fit_column(raw, fitting_rows, logarithm):
    raw = np.nan_to_num(raw, nan=0, posinf=0, neginf=0).astype(np.float64)
    if logarithm:
        raw = np.log1p(np.maximum(raw, 0))
    selected = raw[fitting_rows]
    mu = float(selected.mean())
    # Author pandas std defaults to sample standard deviation ddof=1.
    std = float(selected.std(ddof=1)) if len(selected) > 1 else 1.
    if not np.isfinite(std) or std < 1e-6:
        std = 1.
    return ((raw - mu) / std).astype(np.float32), {'mean': mu, 'std': std,
                                               'log1p': logarithm, 'ddof': 1}


def build_features(guard):
    """Only explicit audit stage reads raw feature tables; never test logs."""
    destination = OUT / 'features'
    if destination.exists():
        raise FileExistsError('feature directory exists; do not overwrite audited cache')
    building = OUT / 'features_building'
    if building.exists():
        raise FileExistsError('incomplete feature build; inspect/recover manually')
    building.mkdir(parents=True)
    train, valid = load_temporal_frames(ROOT)
    request_consistency = {}
    for name, frame in [('train', train), ('valid', valid)]:
        inconsistent = 0
        for _, group in frame.groupby('request_id', sort=False):
            inconsistent += int(group.user_id.nunique() != 1 or len({tuple(h) for h in group.history_item_ids}) != 1)
        if inconsistent:
            raise AssertionError(f'{name} request history/user inconsistent')
        overlap = sum(int(row.positive_item_id in row.history_item_ids) for row in frame.itertuples())
        request_consistency[name] = {'inconsistent_requests': inconsistent,
                                     'history_target_overlap_samples': overlap,
                                     'pre_request_history_time': 'inherited Phase4/5 protocol; per-history timestamps unavailable'}
        if name == 'train' and overlap:
            raise AssertionError('train history-target overlap: stop and review inherited protocol')
    catalog = np.load(NOTE_IDS_PATH)
    if len(catalog) != PROTOCOL['items'] or len(np.unique(catalog)) != len(catalog):
        raise AssertionError('canonical catalog changed')
    verify_image_assets('i3_all', full_hash=True)
    users = pd.concat([pd.read_parquet(p) for p in sorted((ROOT / 'data/user_feat').glob('*.parquet'))],
                      ignore_index=True)
    if users.user_idx.duplicated().any():
        raise AssertionError('duplicate user feature row')
    users = users.sort_values('user_idx').reset_index(drop=True)
    all_users = users.user_idx.to_numpy(np.int64)
    train_users = np.sort(train.user_id.unique().astype(np.int64))
    user_fit = rows_for(train.user_id.to_numpy(np.int64), all_users)
    if (user_fit < 0).any():
        raise AssertionError('missing temporal-train user features')
    targets = train.positive_item_id.to_numpy(np.int64)
    fitting = rows_for(targets, catalog)
    if (fitting < 0).any():
        raise AssertionError('train target outside corpus')
    history_ids = np.unique(np.concatenate([np.asarray(v, dtype=np.int64)
                                          for v in train.history_item_ids if len(v)]))
    fitting_categories = rows_for(np.union1d(targets, history_ids), catalog)
    fitting_categories = fitting_categories[fitting_categories >= 0]
    raw_numeric = np.zeros((len(catalog), len(ITEM_BASIC + ITEM_COUNTS)), np.float32)
    raw_cats = np.empty((len(catalog), 3), dtype=object)
    raw_cats[:] = '<MISSING>'
    note_type = np.zeros(len(catalog), np.int64)
    covered = np.zeros(len(catalog), bool)
    manifest, present_fields, missing, totals = {}, set(), {}, {}
    invalid_note_type_count = 0
    required = ['note_idx', 'note_type'] + ITEM_CATEGORICAL + ITEM_BASIC + ITEM_COUNTS
    input_files = {}
    for path in sorted((ROOT / 'data/notes').glob('*.parquet')):
        guard.check()
        input_files[str(path.relative_to(ROOT))] = sha256(path)
        fields = set(pq.ParquetFile(path).schema_arrow.names)
        frame = pd.read_parquet(path, columns=[c for c in required if c in fields])
        rows = rows_for(frame.note_idx.to_numpy(np.int64), catalog)
        if (rows < 0).any() or covered[rows].any() or frame.note_idx.duplicated().any():
            raise AssertionError('notes/canonical mapping duplicate or outside corpus')
        covered[rows] = True
        for c in required[1:]:
            totals[c] = totals.get(c, 0) + len(frame)
            if c not in fields:
                missing[c] = missing.get(c, 0) + len(frame)
                continue
            present_fields.add(c)
            missing[c] = missing.get(c, 0) + int(frame[c].isna().sum())
            if c in ITEM_CATEGORICAL:
                raw_cats[rows, ITEM_CATEGORICAL.index(c)] = frame[c].map(category)
                missing[c] -= int(frame[c].isna().sum())
                missing[c] += int(frame[c].map(category).eq('<MISSING>').sum())
            else:
                numeric = pd.to_numeric(frame[c], errors='coerce').fillna(0).to_numpy()
                if c == 'note_type':
                    invalid_note_type_count += int((~np.isin(numeric, [1, 2])).sum())
                    note_type[rows] = np.where(np.isin(numeric, [1, 2]), numeric, 0).astype(np.int64)
                else:
                    raw_numeric[rows, (ITEM_BASIC + ITEM_COUNTS).index(c)] = numeric
        print(f'feature audit: {path.name}, mapped={int(covered.sum())}', flush=True)
    if not covered.all():
        raise AssertionError('notes/canonical sets do not match')
    raw_by_name = {c: raw_numeric[:, i] for i, c in enumerate(ITEM_BASIC + ITEM_COUNTS)}
    for name, (numerator, denominator) in RATE_INPUTS.items():
        raw_by_name[name] = np.divide(raw_by_name[numerator], raw_by_name[denominator],
                                     out=np.zeros(len(catalog), np.float32),
                                     where=raw_by_name[denominator] != 0)
    normalization = {'fit_scope': 'temporal-train positive samples only', 'F': {}, 'S': {}}
    item_f = np.zeros((len(catalog), len(ITEM_NUMERIC)), np.float32)
    item_s = np.zeros_like(item_f)
    for i, c in enumerate(ITEM_NUMERIC):
        derived = c in RATE_INPUTS
        available = all(x in present_fields for x in RATE_INPUTS[c]) if derived else c in present_fields
        if available:
            item_f[:, i], normalization['F'][c] = fit_column(raw_by_name[c], fitting, not derived)
        # Safe static asset attributes only. No snapshot behavioral count enters S.
        safe = c in ITEM_BASIC and available
        if safe:
            item_s[:, i] = item_f[:, i]
            normalization['S'][c] = normalization['F'][c]
        manifest[c] = {'side': 'item', 'slot': i, 'source': RATE_INPUTS.get(c, c),
                       'available': available, 'F': 'snapshot_reference' if available else 'zero_missing',
                       'S': 'static_asset' if safe else 'zero_unverified_time_or_missing',
                       'time_verified': safe, 'missing_rate': missing.get(c, len(catalog)) / len(catalog)
                       if not derived else None}
    user_f = np.zeros((len(users), 42), np.float32)
    user_s = np.zeros_like(user_f)
    for i, c in enumerate(USER_NUMERIC):
        available = c in users.columns
        if available:
            raw = pd.to_numeric(users[c], errors='coerce').fillna(0).to_numpy()
            user_f[:, i], normalization['F'][c] = fit_column(raw, user_fit, i < 2)
        manifest[c] = {'side': 'user', 'slot': i, 'available': available,
                       'F': 'snapshot_reference' if available else 'zero_missing',
                       'S': 'zero_unverified_time', 'time_verified': False,
                       'missing_rate': float(users[c].isna().mean()) if available else 1.}
    vocabs, item_c, user_c = {}, np.zeros((len(catalog), 3), np.int64), np.zeros((len(users), 4), np.int64)
    for i, c in enumerate(ITEM_CATEGORICAL):
        vocab = {v: j + 1 for j, v in enumerate(sorted(set(raw_cats[fitting_categories, i]) - {'<MISSING>'}))}
        vocabs[c] = vocab
        item_c[:, i] = [vocab.get(v, 0) for v in raw_cats[:, i]]
    for i, c in enumerate(USER_CATEGORICAL):
        raw = users[c].map(category) if c in users else pd.Series(['<MISSING>'] * len(users))
        vocab = {v: j + 1 for j, v in enumerate(sorted(set(raw.iloc[user_fit]) - {'<MISSING>'}))}
        vocabs[c] = vocab
        user_c[:, i] = raw.map(lambda v: vocab.get(v, 0)).to_numpy()
    for c in ITEM_CATEGORICAL + USER_CATEGORICAL + ['note_type']:
        manifest[c] = {'side': 'user' if c in USER_CATEGORICAL else 'item',
                       'available': c in users if c in USER_CATEGORICAL else c in present_fields,
                       'vocab_scope': 'temporal_train_users/items_history_union_target',
                       'time_status': 'static_category_assumption; snapshot effective date unavailable',
                       'missing_rate': float(users[c].map(category).eq('<MISSING>').mean())
                       if c in users else missing.get(c, len(catalog)) / len(catalog)}
    sizes = dict(user_idx=len(train_users) + 1, note_idx=len(catalog) + 1,
                 **{name: len(vocabs[c]) + 1 for name, c in zip(
                     ['gender', 'platform', 'age', 'location', 'tax1', 'tax2', 'tax3'],
                     USER_CATEGORICAL + ITEM_CATEGORICAL)})
    for name, array in dict(note_ids=catalog, user_ids=all_users, train_users=train_users,
                            train_target_ids=np.unique(targets), train_history_ids=history_ids,
                            item_categorical=item_c, item_type=note_type, user_categorical=user_c,
                            item_numeric_F=item_f, item_numeric_S=item_s,
                            user_numeric_F=user_f, user_numeric_S=user_s).items():
        np.save(building / f'{name}.npy', array)
    dependencies = {str(p.relative_to(ROOT)): sha256(p) for p in (NOTE_IDS_PATH, TRAIN_PATH, VALID_PATH, BGE_PATH)}
    dependencies.update(input_files)
    dependencies.update({str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / 'data/user_feat').glob('*.parquet'))})
    dependencies.update({str(p.relative_to(ROOT)): sha256(p) for p in image_paths('i3_all')})
    h2_dependencies = {}
    for seed in (42, 43, 44):
        for path in (H2 / f'validation/h2_256_seed{seed}.json',
                     H2 / f'validation/rankings/h2_256/seed{seed}.parquet',
                     H2 / f'configs/markers/full_validation_h2_256_seed{seed}.json'):
            guard.check()
            h2_dependencies[str(path.relative_to(ROOT))] = sha256(path)
    atomic_json(building / 'vocab.json', {'categories': vocabs, 'sizes': sizes})
    atomic_json(building / 'normalization.json', normalization)
    atomic_json(building / 'feature_manifest.json', {'fields': manifest,
                'S_behavior_counts': 'all zero: no per-request historical counter available; no end-of-train future counts',
                'catalog_setting': 'known full corpus; all in-catalog ID rows are available',
                'missing_fields_are_zero_not_substituted': True, 'dependencies': dependencies,
                'h2_validation_dependencies': h2_dependencies,
                'history_audit': request_consistency,
                'invalid_or_missing_note_type_count': invalid_note_type_count,
                'train': len(train), 'valid': len(valid), 'requests': len(grouped_requests(valid))})
    guard.check()
    complete(building, list(building.glob('*.npy')) + list(building.glob('*.json')))
    os.rename(building, destination)


class Assets:
    def __init__(self, protocol, images=True, verify=True):
        if verify:
            verified(OUT / 'features')
            dependencies = json.loads((OUT / 'features/feature_manifest.json').read_text())['dependencies']
            # Runtime encodings reference these external arrays; bind them to audit.
            for path in (BGE_PATH, NOTE_IDS_PATH, *image_paths('i3_all')):
                if sha256(path) != dependencies[str(path.relative_to(ROOT))]:
                    raise RuntimeError(f'input asset changed since audit: {path}')
        self.protocol, self.images = protocol, images
        self.catalog = np.load(OUT / 'features/note_ids.npy', mmap_mode='r')
        self.order = np.argsort(self.catalog)
        self.sorted_ids = self.catalog[self.order]
        self.vocabs = json.loads((OUT / 'features/vocab.json').read_text())['sizes']
        self.item_cat = np.load(OUT / 'features/item_categorical.npy', mmap_mode='r')
        self.item_type = np.load(OUT / 'features/item_type.npy', mmap_mode='r')
        self.item_num = np.load(OUT / f'features/item_numeric_{protocol}.npy', mmap_mode='r')
        self.user_ids = np.load(OUT / 'features/user_ids.npy', mmap_mode='r')
        self.train_users = np.load(OUT / 'features/train_users.npy', mmap_mode='r')
        self.user_cat = np.load(OUT / 'features/user_categorical.npy', mmap_mode='r')
        self.user_num = np.load(OUT / f'features/user_numeric_{protocol}.npy', mmap_mode='r')
        self.text = np.memmap(BGE_PATH, mode='r', dtype=np.float16, shape=(len(self.catalog), 768))
        self.image = np.load(image_paths('i3_all')[0], mmap_mode='r') if images else None
        self.available = np.load(image_paths('i3_all')[1], mmap_mode='r')

    def lookup(self, ids):
        values = np.asarray(ids, dtype=np.int64)
        pos = np.searchsorted(self.sorted_ids, values)
        safe = np.minimum(pos, len(self.catalog) - 1)
        return np.where((pos < len(self.catalog)) & (self.sorted_ids[safe] == values), self.order[safe], -1)

    def content(self, rows):
        rows = np.asarray(rows, np.int64)
        good = rows >= 0
        text = np.zeros(rows.shape + (768,), np.float32)
        image = np.zeros_like(text)
        text[good] = self.text[rows[good]]
        if self.images:
            image_good = good.copy()
            image_good[good] &= self.available[rows[good]]
            image[image_good] = self.image[rows[image_good]]
        return torch.from_numpy(text), torch.from_numpy(image)

    def items(self, rows):
        rows = np.asarray(rows, np.int64)
        safe = np.maximum(rows, 0)
        batch = {'note_idx': torch.from_numpy(np.where(rows >= 0, rows + 1, 0)),
                 'note_type': torch.from_numpy(np.where(rows >= 0, self.item_type[safe], 0)),
                 'taxonomy': torch.from_numpy(np.where((rows >= 0)[:, None], self.item_cat[safe], 0).copy()),
                 'dense_stats': torch.from_numpy(np.where((rows >= 0)[:, None], self.item_num[safe], 0).copy())}
        text, image = self.content(rows)
        return batch, text, image

    def users(self, ids, histories):
        ids = np.asarray(ids, np.int64)
        feature_rows = rows_for(ids, self.user_ids)
        id_rows = rows_for(ids, self.train_users)
        cat = np.zeros((len(ids), 4), np.int64)
        numeric = np.zeros((len(ids), 42), np.float32)
        good = feature_rows >= 0
        cat[good] = self.user_cat[feature_rows[good]]
        numeric[good] = self.user_num[feature_rows[good]]
        user = {'user_idx': torch.from_numpy(np.where(id_rows >= 0, id_rows + 1, 0)),
                **{name + '_enc': torch.from_numpy(cat[:, i]) for i, name in enumerate(USER_CATEGORICAL)},
                'fans_num': torch.from_numpy(numeric[:, 0]), 'follows_num': torch.from_numpy(numeric[:, 1]),
                'dense_feats': torch.from_numpy(numeric[:, 2:])}
        # No target input whatsoever. Preserve frozen chronological last-20 order.
        rows = np.full((len(ids), 20), -1, np.int64)
        mask = torch.zeros((len(ids), 20))
        for i, history in enumerate(histories):
            h = [int(v) for v in history if int(v) >= 0][-20:]
            mapped = self.lookup(h)
            # OOV content remains zero, and is not a valid sequence element.
            mapped = mapped[mapped >= 0]
            rows[i, :len(mapped)] = mapped
            mask[i, :len(mapped)] = 1
        text, image = self.content(rows)
        fused = .72 * text + .28 * image
        summary = summarize_batch(fused, mask)
        return user, summary, fused, mask


def summarize_batch(sequence, mask):
    """Author formula vectorized on padded data, only actual valid rows included."""
    lengths = mask.sum(1).long()
    valid = mask.bool()
    denominator = lengths.clamp_min(1).to(sequence.dtype)
    mean = (sequence * mask[..., None]).sum(1) / denominator[:, None]
    position = torch.arange(sequence.shape[1], device=sequence.device)[None, :]
    recent_mask = valid & (position >= (lengths - 5).clamp_min(0)[:, None])
    recent = (sequence * recent_mask[..., None]).sum(1) / recent_mask.sum(1).clamp_min(1)[:, None]
    last = sequence[torch.arange(len(sequence), device=sequence.device), (lengths - 1).clamp_min(0)]
    maximum = sequence.masked_fill(~valid[..., None], -torch.inf).max(1).values
    maximum = torch.where(lengths[:, None] > 0, maximum, torch.zeros_like(maximum))
    weights = (.35 + .65 * position / (lengths - 1).clamp_min(1)[:, None]) * mask
    weights = weights / weights.sum(1).clamp_min(1e-12)[:, None]
    recency = (sequence * weights[..., None]).sum(1)
    result = F.normalize(.34 * recency + .26 * recent + .18 * last + .14 * mean + .08 * maximum, dim=1)
    return result * lengths.gt(0)[:, None]


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return tuple(to_device(v, device) for v in value)
