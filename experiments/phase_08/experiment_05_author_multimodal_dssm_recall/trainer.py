"""Author easy loss; unique target encoding and an audit of unmasked false negatives."""
import time

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

from .artifacts import atomic_torch
from .config import PROTOCOL
from .data import to_device


def amp_optimizer_step(model, optimizer, scaler):
    """Let GradScaler skip an overflowing update and lower its scale.

    Return the offending names so the caller can replay the SAME batch/mask.
    Non-AMP nonfinite gradients remain fatal; no NaN replacement or clipping.
    """
    bad = [name for name, p in model.named_parameters()
           if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
    before = scaler.get_scale()
    if bad and not scaler.is_enabled():
        raise FloatingPointError(f'non-AMP gradient nonfinite: {bad}')
    scaler.step(optimizer)
    scaler.update()
    if bad and scaler.get_scale() >= before:
        raise FloatingPointError('GradScaler did not reject/back off overflowing update')
    return bad, before, scaler.get_scale()


def false_negative_mask(frame):
    targets = frame.positive_item_id.to_numpy(np.int64)
    users = frame.user_id.to_numpy(np.int64)
    requests = frame.request_id.to_numpy(np.int64)
    diagonal = np.eye(len(frame), dtype=bool)
    return {
        'duplicate_target': (targets[:, None] == targets[None, :]) & ~diagonal,
        'same_user': (users[:, None] == users[None, :]) & ~diagonal,
        'same_request_positive': (requests[:, None] == requests[None, :]) & ~diagonal,
        'history_positive': np.stack([np.isin(targets, h) for h in frame.history_item_ids]) & ~diagonal,
    }


def train_epoch(model, assets, frame, order, optimizer, scaler, device, guard, epoch, checkpoint_dir):
    model.train()
    model.reset_dropout_counts()
    started = time.monotonic()
    records, audit = [], {k: 0 for k in ('duplicate_target', 'same_user', 'same_request_positive', 'history_positive')}
    potential = 0
    wait = 0.
    last_log = started
    auc_score, auc_truth = [], []
    gradient_sums = {}
    scales = []
    overflow_events = []
    try:
        for batch_start in range(0, len(order), PROTOCOL['batch_size']):
            guard.check()
            load_start = time.monotonic()
            selected = frame.iloc[order[batch_start:batch_start + PROTOCOL['batch_size']]]
            if len(selected) < 2:
                continue
            user = to_device(assets.users(selected.user_id, selected.history_item_ids), device)
            target_rows = assets.lookup(selected.positive_item_id)
            unique, inverse = np.unique(target_rows, return_inverse=True)
            if (unique < 0).any():
                raise AssertionError('positive item out of catalog')
            item = to_device(assets.items(unique), device)
            wait += time.monotonic() - load_start
            masks = false_negative_mask(selected)
            for name, mask in masks.items():
                audit[name] += int(mask.sum())
            potential += len(selected) * (len(selected) - 1)
            dropout_rng = model.dropout_generator.get_state()
            dropout_counts = {k: list(v) for k, v in model.dropout_counts.items()}
            for attempt in range(9):
                guard.check()
                # The author tower has no other stochastic layers. Restoring
                # this independent RNG replays identical candidate/user masks.
                model.dropout_generator.set_state(dropout_rng)
                model.dropout_counts = {k: list(v) for k, v in dropout_counts.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    query = model.forward_user(*user)
                    candidates = model.forward_item(*item)
                    positives = candidates[torch.as_tensor(inverse, device=device)]
                    raw = query.float() @ positives.float().T
                    # Exact author's EASY objective: no silent mask correction.
                    loss = F.cross_entropy(raw / .07, torch.arange(len(query), device=device))
                if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(raw).all()):
                    raise FloatingPointError('nonfinite forward/loss; not an AMP-scale retry')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                bad, before, after = amp_optimizer_step(model, optimizer, scaler)
                if not bad:
                    break
                event = {'batch_start': batch_start, 'attempt': attempt + 1,
                         'parameters': bad, 'scale_before': before, 'scale_after': after}
                overflow_events.append(event)
                print(f'AMP overflow: {event}; replaying same batch and ID masks', flush=True)
            else:
                raise FloatingPointError('AMP gradients still nonfinite after 8 same-batch retries')
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise FloatingPointError(f'gradient nonfinite: {name}')
                    gradient_sums[name] = gradient_sums.get(name, 0.) + float(parameter.grad.float().norm())
            records.append(float(loss))
            # Bounded deterministic AUC diagnostic, not full B^2 epoch storage.
            if len(auc_score) < 20:
                diag = raw.detach().diagonal().cpu().numpy()
                off = raw.detach()[torch.arange(len(query), device=device),
                                   (torch.arange(len(query), device=device) + 1) % len(query)].cpu().numpy()
                auc_score.append(np.concatenate((diag, off)))
                auc_truth.append(np.concatenate((np.ones(len(diag)), np.zeros(len(off)))))
            scales.append({'item_id_norm': float(model.i_emb_item.weight[item[0]['note_idx']].detach().norm(dim=1).mean()),
                           'user_id_norm': float(model.u_emb_user.weight[user[0]['user_idx']].detach().norm(dim=1).mean()),
                           'user_numeric_norm': float(torch.cat([user[0]['fans_num'][:, None], user[0]['follows_num'][:, None], user[0]['dense_feats']], dim=1).norm(dim=1).mean()),
                           'item_numeric_norm': float(item[0]['dense_stats'].norm(dim=1).mean()),
                           'text_norm': float(item[1].norm(dim=1).mean()),
                           'image_norm': float(item[2].norm(dim=1).mean()),
                           'positive_cosine': float(raw.detach().diagonal().mean()),
                           'negative_cosine': float(raw.detach()[~torch.eye(len(query), device=device, dtype=torch.bool)].mean())})
            if time.monotonic() - last_log >= 30:
                print(f'epoch={epoch} samples={batch_start + len(selected)}/{len(order)} '
                      f'loss={np.mean(records):.5f} RSS={guard.peak_rss / 2**30:.2f}GiB', flush=True)
                last_log = time.monotonic()
    except BaseException:
        atomic_torch(checkpoint_dir / 'interrupted.pt', {'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'epoch': epoch,
                     'batch_start': batch_start if 'batch_start' in locals() else 0,
                     'dropout_rng': model.dropout_generator.get_state(), 'complete': False})
        raise
    guard.check()
    elapsed = time.monotonic() - started
    return {'epoch': epoch, 'loss': float(np.mean(records)), 'seconds': elapsed,
            'samples_per_second': len(order) / elapsed, 'data_wait_fraction': wait / elapsed,
            'false_negative_pairs': audit, 'off_diagonal_pairs': potential,
            'false_negative_rates': {k: v / max(1, potential) for k, v in audit.items()},
            'false_negative_policy': 'audited, NOT masked; same original easy loss in all six models',
            'auc_auxiliary_bounded': float(roc_auc_score(np.concatenate(auc_truth), np.concatenate(auc_score))),
            'dropout': {k: {'seen': v[0], 'dropped_seen': v[1], 'oov': v[2],
                       'drop_ratio_seen': v[1] / max(v[0], 1)} for k, v in model.dropout_counts.items()},
            'gradient_norms_mean': {k: v / len(records) for k, v in gradient_sums.items()},
            'scales': {k: float(np.mean([s[k] for s in scales])) for k in scales[0]},
            'peak_rss_gib': guard.peak_rss / 2**30,
            'amp_overflow_retries': len(overflow_events), 'amp_overflow_events': overflow_events,
            'amp_final_scale': scaler.get_scale(),
            'gpu_peak_gib': torch.cuda.max_memory_allocated(device) / 2**30}


def validation_loss(model, assets, frame, device, guard):
    """Fixed proxy positive interactions, author easy loss; scheduler-only diagnostic."""
    model.eval()
    losses = []
    with torch.inference_mode():
        for start in range(0, len(frame), PROTOCOL['batch_size']):
            guard.check()
            batch = frame.iloc[start:start + PROTOCOL['batch_size']]
            if len(batch) < 2:
                continue
            user = to_device(assets.users(batch.user_id, batch.history_item_ids), device)
            unique, inverse = np.unique(assets.lookup(batch.positive_item_id), return_inverse=True)
            item = to_device(assets.items(unique), device)
            query = model.forward_user(*user)
            positives = model.forward_item(*item)[torch.as_tensor(inverse, device=device)]
            losses.append(float(F.cross_entropy(query @ positives.T / .07,
                                               torch.arange(len(query), device=device))))
    return float(np.mean(losses))


def encode_items(model, assets, rows, device, guard, destination=None):
    vectors = (np.lib.format.open_memmap(destination, mode='w+', dtype=np.float32,
                                        shape=(len(rows), 128)) if destination
               else np.empty((len(rows), 128), np.float32))
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), 4096):
            guard.check()
            batch = to_device(assets.items(rows[start:start + 4096]), device)
            output = model.forward_item(*batch).float()
            if not torch.isfinite(output).all() or not torch.allclose(output.norm(dim=1), torch.ones(len(output), device=device), atol=1e-4, rtol=1e-4):
                raise FloatingPointError('catalog vector is nonfinite or not normalized')
            vectors[start:start + len(batch[1])] = output.cpu().numpy()
            if start % 131072 == 0:
                print(f'item encoding {start}/{len(rows)}', flush=True)
    if hasattr(vectors, 'flush'):
        vectors.flush()
    guard.check()
    return vectors


def encode_queries(model, assets, requests, device, guard):
    result = np.empty((len(requests), 128), np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(requests), 256):
            guard.check()
            batch = requests[start:start + 256]
            user = to_device(assets.users([r.user_idx for r in batch], [r.history for r in batch]), device)
            output = model.forward_user(*user).float()
            if not torch.isfinite(output).all() or not torch.allclose(output.norm(dim=1), torch.ones(len(output), device=device), atol=1e-4, rtol=1e-4):
                raise FloatingPointError('query vector is nonfinite or not normalized')
            result[start:start + len(batch)] = output.cpu().numpy()
    return result
