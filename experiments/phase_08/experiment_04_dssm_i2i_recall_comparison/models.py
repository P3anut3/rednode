"""Author-style concat tower under Qilin's frozen, audited feature contract."""

import torch

from experiments.phase_07.experiment_03_author_concat_regularization_ablation.models import (
    AuthorConcatTower,
    Variant,
)

# This is a controlled *author-style* model, not a reproduction of an external
# training run.  No image, anonymous dense or future aggregate behaviour enters.
D_VARIANT = Variant("d_author_style", 768, 32, 0.01, 0.0)


class TimeAuditedAuthorConcatTower(AuthorConcatTower):
    """Suppress dynamic user counts whose snapshot cutoff cannot be verified."""

    def query_components(self, batch, *, item_id_enabled=True, user_id_enabled=True):
        # gender/platform/age remain; fans/follows are not fed to the tower.
        safe = dict(batch)
        safe["user_numeric"] = torch.zeros_like(batch["user_numeric"])
        return super().query_components(
            safe, item_id_enabled=item_id_enabled, user_id_enabled=user_id_enabled
        )


def make_model(store, device):
    schema = store.schema()
    return TimeAuditedAuthorConcatTower(
        D_VARIANT,
        user_vocab=schema["user_count"],
        item_vocab=schema["item_id_count"],
        user_category_sizes=schema["user_category_sizes"],
        item_category_sizes=schema["item_category_sizes"],
        bge_dim=768,
        output_dim=128,
    ).to(device)
