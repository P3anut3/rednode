"""Synthetic only. No parquet, model download, corpus encoding or test access."""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'experiments.phase_08.experiment_05_author_multimodal_dssm_recall'

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .config import ITEM_NUMERIC, MODELS
from .data import summarize_batch, rows_for, Assets, fit_column, category
from .artifacts import atomic_json, complete, verified
from .evaluation import bootstrap_delta
from .models import MultimodalDSSM
from .source_contract import author_namespace
from .trainer import false_negative_mask, amp_optimizer_step
from .retrieval import filtered
from .routes import quota_merge, rrf, append_deeper_candidates
from .selection import choose_models


def synthetic():
    torch.manual_seed(91)
    user = {'user_idx': torch.tensor([1, 2, 2, 4]),
            'gender_enc': torch.tensor([1, 2, 1, 2]),
            'platform_enc': torch.tensor([1, 2, 3, 2]),
            'age_enc': torch.tensor([1, 2, 3, 2]),
            'location_enc': torch.tensor([1, 2, 3, 2]),
            'fans_num': torch.randn(4), 'follows_num': torch.randn(4), 'dense_feats': torch.randn(4, 40)}
    item = {'note_idx': torch.tensor([1, 2, 3, 4]), 'note_type': torch.tensor([1, 2, 1, 2]),
            'taxonomy': torch.tensor([[1, 2, 3], [2, 1, 2], [1, 2, 1], [1, 1, 1]]),
            'dense_stats': torch.randn(4, len(ITEM_NUMERIC))}
    text, image, sequence = torch.randn(4, 768), torch.randn(4, 768), torch.randn(4, 20, 768)
    mask = torch.zeros(4, 20)
    for i, length in enumerate((20, 10, 5, 1)):
        mask[i, :length] = 1
    sequence *= mask[..., None]
    return user, item, text, image, sequence, mask


def model(dropout=0):
    sizes = dict(user_idx=17, note_idx=19, gender=5, platform=5, age=5, location=5, tax1=5, tax2=5, tax3=5)
    return MultimodalDSSM(sizes, len(ITEM_NUMERIC), dropout), sizes


def run_checks():
    torch.set_num_threads(2)
    # Actual CPU GradScaler: nonfinite update must not modify parameter or
    # optimizer state; a finite replay must update once at the reduced scale.
    small = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(small.parameters(), lr=.001, weight_decay=.01)
    scaler = torch.amp.GradScaler('cpu', init_scale=65536.)
    initial_weight = small.weight.detach().clone()
    scaler.scale(small(torch.ones(1, 1)).sum()).backward()
    small.weight.grad.fill_(float('inf'))
    scaler.unscale_(optimizer)
    bad, before, after = amp_optimizer_step(small, optimizer, scaler)
    assert bad == ['weight'] and before == 65536. and after == 32768.
    assert torch.equal(small.weight, initial_weight) and not optimizer.state
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(small(torch.ones(1, 1)).sum()).backward()
    scaler.unscale_(optimizer)
    assert not amp_optimizer_step(small, optimizer, scaler)[0]
    assert not torch.equal(small.weight, initial_weight)
    namespace = author_namespace()
    u, i, text, image, sequence, mask = synthetic()
    local, sizes = model()
    author = namespace['DSSMModel']('rec', sizes, len(ITEM_NUMERIC), 938)
    # Same state, original pure class executes independently from local hooks/overrides.
    author.load_state_dict(local.state_dict())
    author.eval()
    local.eval()
    summary = summarize_batch(sequence, mask)
    for row, length in enumerate(mask.sum(1).long()):
        expected = namespace['summarize_history_sequence'](sequence[row, :int(length)])
        torch.testing.assert_close(summary[row], expected, rtol=1e-6, atol=1e-6)
    with torch.no_grad():
        reference_item = author.forward_item(i, text, image)
        reference_user = author.forward_user(u, summary, sequence, mask)
        local_item = local.forward_item(i, text, image)
        local_user = local.forward_user(u, summary, sequence, mask)
    torch.testing.assert_close(local_item, reference_item, rtol=0, atol=0)
    torch.testing.assert_close(local_user, reference_user, rtol=0, atol=0)
    assert local.item_mlp[0].in_features == 1620 + len(ITEM_NUMERIC)
    assert local.user_mlp[0].in_features == 1002
    assert local_item.shape == local_user.shape == (4, 128)
    assert not any(isinstance(m, (torch.nn.LayerNorm, torch.nn.GELU)) for m in local.modules())
    # Every same-seed matrix member starts with bit-identical parameters.
    states = []
    for configuration in MODELS.values():
        torch.manual_seed(42)
        instance, _ = model(configuration['id_dropout'])
        states.append(instance.state_dict())
    for state in states[1:]:
        assert all(torch.equal(state[k], states[0][k]) for k in state)
    # Gradients: full F multimodal input (including GRU/category/IDs and both MLPs).
    local.train()
    local.zero_grad(set_to_none=True)
    query = local.forward_user(u, summary, sequence, mask)
    candidate = local.forward_item(i, text, image)
    loss = F.cross_entropy(query @ candidate.T / .07, torch.arange(4))
    loss.backward()
    for name, parameter in local.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert not text.requires_grad and not image.requires_grad and not sequence.requires_grad
    assert text.grad is None and image.grad is None
    assert not local.i_emb_item.weight.grad[0].any() and not local.u_emb_user.weight.grad[0].any()
    # Structured whole-ID mask, shared duplicate IDs, no inverted scaling.
    dropout, _ = model(.5)
    dropout.train()
    ids = torch.tensor([1, 1, 2, 3, 4, 5, 6, 7, 8])
    before = dropout.i_emb_item.weight[ids].detach().clone()
    after = dropout.i_emb_item(ids)
    kept = after.abs().sum(-1) > 0
    assert torch.equal(after[0], after[1])
    assert kept.any() and (~kept).any()
    assert torch.equal(after[kept], before[kept])
    assert not after[~kept].any()
    dropout.eval()
    torch.testing.assert_close(dropout.i_emb_item(ids), before, rtol=0, atol=0)
    dropout.id_dropout = 0
    dropout.train()
    torch.testing.assert_close(dropout.i_emb_item(ids), before, rtol=0, atol=0)
    # All-drop ID branch gradient must be zero (other branches remain trainable).
    dropout.id_dropout = 1.
    dropout.zero_grad(set_to_none=True)
    (dropout.forward_user(u, summary, sequence, mask).sum()
     + dropout.forward_item(i, text, image).sum()).backward()
    assert not dropout.i_emb_item.weight.grad.any() and not dropout.u_emb_user.weight.grad.any()
    # Global torch RNG is untouched by ID masks.
    rng = torch.random.get_rng_state().clone()
    dropout.i_emb_item(ids)
    assert torch.equal(rng, torch.random.get_rng_state())
    # Explicit OOV mapping and all-empty history finite.
    local.eval()
    out_u = dict(u, user_idx=torch.tensor([-1, 100, 0, 2]))
    out_i = dict(i, note_idx=torch.tensor([-1, 100, 0, 2]))
    empty = torch.zeros_like(mask)
    for value in (local.forward_user(out_u, torch.zeros_like(summary), torch.zeros_like(sequence), empty),
                  local.forward_item(out_i, text, torch.zeros_like(image))):
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value.norm(dim=1), torch.ones(4), atol=1e-6, rtol=1e-6)
    # Cold invariance: perturb only the matching ID table, not both towers at once.
    cold_item = dict(i, note_idx=torch.zeros(4, dtype=torch.long))
    cold_user = dict(u, user_idx=torch.zeros(4, dtype=torch.long))
    with torch.no_grad():
        expected = local.forward_item(cold_item, text, image)
        item_backup = local.i_emb_item.weight.clone()
        local.i_emb_item.weight[1:].normal_()
        torch.testing.assert_close(local.forward_item(cold_item, text, image), expected, atol=0, rtol=0)
        local.i_emb_item.weight.copy_(item_backup)
        expected = local.forward_user(cold_user, summary, sequence, mask)
        local.u_emb_user.weight[1:].normal_()
        torch.testing.assert_close(local.forward_user(cold_user, summary, sequence, mask), expected, atol=0, rtol=0)
        # Train-batch and catalog encoding are the same callable, test batch invariance.
        whole = local.forward_item(i, text, image)
        parts = torch.cat([local.forward_item({k: v[j:j+1] for k, v in i.items()},
                                             text[j:j+1], image[j:j+1]) for j in range(4)])
        torch.testing.assert_close(whole, parts, atol=1e-6, rtol=1e-6)
    masks = false_negative_mask(pd.DataFrame({'positive_item_id': [4, 4, 8], 'user_id': [1, 1, 2],
                 'request_id': [1, 1, 2], 'history_item_ids': [[8], [], [4]]}))
    assert masks['duplicate_target'][0, 1] and masks['history_positive'][0, 2]
    assert masks['same_request_positive'][1, 0] and masks['same_user'][0, 1]
    # Original easy loss matches local CE; audited masks do not silently change it.
    original_loss = namespace['compute_inbatch_loss'](query.detach(), candidate.detach(),
                    torch.zeros(4, 1, 128), torch.zeros(4, dtype=torch.long))[0]
    torch.testing.assert_close(original_loss, loss.detach())
    np.testing.assert_array_equal(rows_for([7, 9, 1, -1], np.array([9, 1, 7])), [2, 0, 1, -1])
    result = filtered(np.array([[0, 0, 1, 2, 3]]), np.array([9, 2, 7, 1]), [[9]], topk=3)
    assert result == [[2, 7, 1]]
    assert len(set(quota_merge(list(range(500)), list(range(250, 750)), 100))) == 500
    assert len(rrf(list(range(500)), list(range(250, 750)))) == 500
    # Deep candidates cannot change the author180 prefix or full rankings.
    full = list(range(500))
    assert append_deeper_candidates(full, list(reversed(full))) == full
    short = [7, 3, 9]
    assert append_deeper_candidates(short, [9, 8, 7, 5, 4], topk=5) == [7, 3, 9, 8, 5]
    assert append_deeper_candidates([], [], topk=500) == []
    # Selection uses means; a one-seed text-only winner is blocked, not final D.
    scores = {'A0': {42: .03, 43: .03, 44: .03}, 'A1': {42: .04, 43: .04, 44: .04},
              'B0': {42: .10, 43: .01, 44: .01}, 'B1': {42: .09, 43: .09, 44: .09},
              'T0': {42: .02}, 'T1': {42: .03}}
    decision = choose_models(scores)
    assert decision['selection_ready'] and decision['safe_model'] == 'B1'
    scores['T0'][42] = .11
    decision = choose_models(scores)
    assert not decision['selection_ready'] and decision['safe_model'] is None
    assert decision['required_T_replications'] == {'T0': [43, 44]}
    scores['T0'].update({43: .01, 44: .01})
    assert choose_models(scores)['safe_model'] == 'B1'
    scores['T0'].update({43: .11, 44: .11})
    assert choose_models(scores)['safe_model'] == 'T0'
    # Test actual mmap gather interface against small synthetic arrays, no real assets.
    assets = Assets.__new__(Assets)
    assets.catalog = np.array([40, 10, 30, 20, 50])
    assets.order = np.argsort(assets.catalog)
    assets.sorted_ids = assets.catalog[assets.order]
    assets.user_ids, assets.train_users = np.array([100, 200]), np.array([100])
    assets.user_cat = np.ones((2, 4), np.int64)
    assets.user_num = np.zeros((2, 42), np.float32)
    assets.text = np.ones((5, 768), np.float16)
    assets.image = np.full((5, 768), 2, np.float16)
    assets.available, assets.images = np.array([True, False, False, True, False]), True
    actual = assets.users([100, 200], [[10, 40], [99999]])
    assert actual[0]['user_idx'].tolist() == [1, 0]
    assert actual[3].sum(1).tolist() == [2., 0.]
    torch.testing.assert_close(actual[2][0, 0], torch.full((768,), .72))
    torch.testing.assert_close(actual[2][0, 1], torch.full((768,), 1.28))
    assert not actual[1][1].any()
    assets.images = False
    text_only = assets.users([100], [[40]])
    torch.testing.assert_close(text_only[2][0, 0], torch.full((768,), .72))
    transformed, statistics = fit_column(np.array([5., 10., 100000.]), [0, 1], False)
    assert statistics['mean'] == 7.5 and transformed[2] > 1000
    assert category('nan') == category('NULL') == '<MISSING>'
    assert bootstrap_delta(np.array([np.nan]))['status'] == 'N/A'
    with tempfile.TemporaryDirectory(prefix='phase805_selfcheck_') as temporary:
        folder = Path(temporary)
        atomic_json(folder / 'product.json', {'finite': 1.})
        complete(folder, [folder / 'product.json'])
        verified(folder)
        atomic_json(folder / 'product.json', {'finite': 2.})
        try:
            verified(folder)
        except RuntimeError:
            pass
        else:
            raise AssertionError('changed artifact not rejected')
        # Empty report must be renderable without opening real temporal labels.
        from . import reporting
        from .reporting import route_stability
        assert not route_stability({})['configurations']
        empty_output = folder / 'report'
        with patch.object(reporting, 'OUT', empty_output):
            reporting.build_report()
        text_report = (empty_output / 'summary.md').read_text()
        assert '当前只有代码实现与合成自检' in text_report
        assert 'ID-off' in text_report and '资源成本' in text_report
        assert json.loads((empty_output / 'report_evidence.json').read_text())['formal_model_count'] == 0
        # Render a populated report with entirely synthetic official products.
        # This covers null cold slices, ID-off, resources and H2 reference rows.
        synthetic_report = folder / 'populated_report'
        vfolder = synthetic_report / 'validation/A0_seed42'
        t = {'Recall@100': .01, 'Recall@500': .02, 'MRR@100': .001}
        value = {'metrics': {'overall': t, 'warm_item': t,
                            'cold_item': {'Recall@500': None}, 'warm_user': t, 'cold_user': t},
                 'phase6_item_status': {name: t for name in
                    ('train_target_seen', 'train_history_only', 'completely_unseen')},
                 'segments': {'with_image': t, 'without_image': t},
                 'top500_coverage': 1., 'comparison_H2_same_seed': {'overall': {
                    'delta': -.01, 'ci95': [-.02, 0.]}},
                 'timing': {'item_encoding_seconds': 1., 'query_encoding_seconds': .5,
                           'index_build_seconds': 1., 'query_search_seconds': 1.,
                           'mean_search_latency_seconds': .01},
                 'item_vector_bytes': 512, 'checkpoint_bytes': 128,
                 'gpu_peak_memory_bytes': 1000, 'process_tree_peak_rss_gib': .1}
        atomic_json(vfolder / 'metrics.json', value)
        complete(vfolder, [vfolder / 'metrics.json'])
        training_folder = synthetic_report / 'training/A0_seed42'
        atomic_json(training_folder / 'config.json', {'parameters': 123, 'id_parameters': 100})
        atomic_json(training_folder / 'best.json', {'epoch': 1})
        atomic_json(training_folder / 'training_curves.json', [{
            'epoch': 1, 'seconds': 1., 'data_wait_fraction': .1, 'loss': .5,
            'scales': {'text_norm': 1.}, 'false_negative_rates': {},
            'dropout': {}, 'gradient_norms_mean': {'seq_gru.weight': .1}}])
        complete(training_folder, list(training_folder.glob('*.json')), {'stage_wall_seconds': 3.})
        diagnostic_folder = synthetic_report / 'diagnostics/A0_seed42'
        atomic_json(diagnostic_folder / 'diagnostics.json', {
            **{mode: value for mode in ('ID-on', 'Item-ID-off', 'User-ID-off', 'All-ID-off')},
            'source_validation_marker': __import__(__package__ + '.artifacts', fromlist=['sha256']).sha256(vfolder / 'complete.json'),
            'retrieved_item_share': {'completely_unseen': .5}})
        complete(diagnostic_folder, [diagnostic_folder / 'diagnostics.json'])
        h2_folder = folder / 'h2'
        atomic_json(h2_folder / 'validation/h2_256_seed42.json', value)
        from . import run as runner
        with patch.object(reporting, 'OUT', synthetic_report), patch.object(reporting, 'H2', h2_folder), \
                patch.object(reporting, 'load_official', return_value=(value, vfolder)), \
                patch.object(runner, 'h2_rankings', return_value=([], {})):
            reporting.build_report()
        populated = (synthetic_report / 'summary.md').read_text()
        assert '| H2-256 | 42 |' in populated and 'All-ID-off' in populated
        assert 'N/A' in populated and 'false-negative' in populated
    return {'passed': True, 'author_item_max_abs': float((reference_item-local_item).abs().max()),
            'author_user_max_abs': float((reference_user-local_user).abs().max()),
            'summary_max_abs_tolerance': 1e-6, 'initial_weights_shared_six_models': True,
            'structured_dropout_and_gradients': True, 'OOV_empty_history': True,
            'three_seed_selection_and_T_replication_gate': True,
            'author180_prefix_preserved_on_backfill': True,
            'empty_report_has_diagnostics_costs_and_seed_scope': True,
            'reads_real_data': False, 'test_opened': False}


if __name__ == '__main__':
    print(json.dumps(run_checks(), ensure_ascii=False, indent=2))
