"""Actual author class plus explicitly registered OOV/empty/dropout adaptations."""
import contextlib

import torch

from .source_contract import author_namespace

AuthorDSSM = author_namespace()['DSSMModel']


class MultimodalDSSM(AuthorDSSM):
    def __init__(self, cat_vocabs, item_dense_dim, id_dropout=0, dropout_seed=900042):
        # Exact initialization order and defaults from locked original class.
        super().__init__('rec', cat_vocabs, item_dense_dim, 2 + 40 + 768 + 128)
        self.id_dropout = float(id_dropout)
        self.dropout_generator = torch.Generator(device='cpu').manual_seed(dropout_seed)
        self.id_off = set()
        self.dropout_counts = {'user': [0, 0, 0], 'item': [0, 0, 0]}
        # Only row0 is reserved. Known rows retain author's default initialization.
        for name, embedding in (('user', self.u_emb_user), ('item', self.i_emb_item)):
            embedding.padding_idx = 0
            with torch.no_grad():
                embedding.weight[0].zero_()
            embedding.register_forward_hook(self._id_hook(name))

    @staticmethod
    def _safe_index(x, vocab_size):
        # NOT author's clamping-to-known-ID behavior.
        return torch.where((x >= 0) & (x < vocab_size), x, torch.zeros_like(x))

    def _id_hook(self, name):
        def hook(module, inputs, output):
            ids = inputs[0]
            seen = ids.ne(0)
            keep = seen.clone()
            if name in self.id_off:
                keep.zero_()
            elif self.training and self.id_dropout:
                # Unique-ID mask ensures shared candidates get the same mask.
                unique, inverse = torch.unique(ids, return_inverse=True)
                draws = torch.rand(len(unique), generator=self.dropout_generator)
                sampled = (draws >= self.id_dropout).to(ids.device)
                keep &= sampled[inverse].reshape_as(ids)
            if self.training:
                self.dropout_counts[name][0] += int(seen.sum())
                self.dropout_counts[name][1] += int((seen & ~keep).sum())
                self.dropout_counts[name][2] += int((~seen).sum())
            return output * keep.unsqueeze(-1).to(output.dtype)
        return hook

    def forward_user(self, batch_data, history_vecs, seq_tokens=None, seq_mask=None):
        if seq_mask is None or seq_tokens is None or bool(seq_mask.bool().any(1).all()):
            return super().forward_user(batch_data, history_vecs, seq_tokens, seq_mask)
        # Original all-masked softmax returns a spurious GRU bias vector.
        # Explicit zero sequence representation for empty histories, nonempty
        # rows execute precisely the original computation.
        valid = seq_mask.bool().any(1)
        outputs = history_vecs.new_zeros((len(history_vecs), 128))
        for selection, tokens, mask in ((valid, seq_tokens[valid], seq_mask[valid]),
                                        (~valid, None, None)):
            if bool(selection.any()):
                outputs[selection] = super().forward_user(
                    {k: v[selection] for k, v in batch_data.items()},
                    history_vecs[selection], tokens, mask)
        return outputs

    @contextlib.contextmanager
    def disable_id(self, *names):
        old = self.id_off
        self.id_off = set(names)
        try:
            yield
        finally:
            self.id_off = old

    def reset_dropout_counts(self):
        self.dropout_counts = {'user': [0, 0, 0], 'item': [0, 0, 0]}
