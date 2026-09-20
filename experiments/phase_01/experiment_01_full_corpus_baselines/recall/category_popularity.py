"""Taxonomy-aware train popularity with global-popularity backfill."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence

import numpy as np
from tqdm.auto import tqdm

from experiments.common.data import TestRequest
from .popularity import filter_ranking


class CategoryPopularity:
    def __init__(self, click_counts: dict[int, int], exposure_counts: dict[int, int],
                 category_by_note: np.ndarray, global_ranking: Sequence[int], top_categories: int = 1):
        self.category_by_note = category_by_note
        self.global_ranking = list(global_ranking)
        self.top_categories = int(top_categories)
        grouped: dict[int, list[int]] = defaultdict(list)
        for note in click_counts:
            if 0 <= note < len(category_by_note):
                code = int(category_by_note[note])
                if code >= 0:
                    grouped[code].append(note)
        self.category_rankings = {
            code: sorted(notes, key=lambda n: (-click_counts[n], -exposure_counts.get(n, 0), n))
            for code, notes in grouped.items()
        }

    def recommend(self, requests: Sequence[TestRequest], k: int = 500, exclude_history: bool = True) -> list[list[int]]:
        outputs = []
        for request in tqdm(requests, desc=f"Category popularity top{self.top_categories}", leave=False):
            counts = Counter(
                int(self.category_by_note[note]) for note in request.history
                if 0 <= note < len(self.category_by_note) and self.category_by_note[note] >= 0
            )
            categories = [code for code, _ in counts.most_common(self.top_categories)]
            selected: list[int] = []
            seen: set[int] = set()
            # Round-robin prevents a large first category from starving Top-3 categories.
            category_lists = [self.category_rankings.get(code, ()) for code in categories]
            offsets = [0] * len(category_lists)
            while len(selected) < k + 32 and category_lists:
                advanced = False
                for i, ranking in enumerate(category_lists):
                    if offsets[i] < len(ranking):
                        note = int(ranking[offsets[i]])
                        offsets[i] += 1
                        advanced = True
                        if note not in seen:
                            seen.add(note)
                            selected.append(note)
                if not advanced:
                    break
            for note in self.global_ranking:
                if len(selected) >= k + 32:
                    break
                if note not in seen:
                    seen.add(note)
                    selected.append(int(note))
            outputs.append(filter_ranking(selected, request.history, k, exclude_history))
        return outputs
