"""GPU resident FP32 exact IP; candidate filtering never reads labels."""
import time

import numpy as np
import torch
from torch.nn import functional as F


class ResidentExact:
    def __init__(self, vectors, device, guard):
        started = time.monotonic()
        self.device, self.guard = device, guard
        self.corpus = torch.empty(vectors.shape, dtype=torch.float32, device=device)
        with torch.inference_mode():
            for start in range(0, len(vectors), 16384):
                guard.check()
                batch = torch.tensor(np.asarray(vectors[start:start + 16384]),
                                     dtype=torch.float32, device=device)
                self.corpus[start:start + len(batch)] = F.normalize(batch, dim=-1)
        torch.cuda.synchronize(device)
        self.build_seconds = time.monotonic() - started
        self.index_bytes = self.corpus.numel() * 4

    def search(self, queries, k, query_batch=64):
        scores, rows = [], []
        started = time.monotonic()
        with torch.inference_mode():
            for start in range(0, len(queries), query_batch):
                self.guard.check()
                q = torch.tensor(np.asarray(queries[start:start + query_batch]),
                                 device=self.device, dtype=torch.float32)
                matrix = F.normalize(q, dim=-1) @ self.corpus.T
                value, row = matrix.topk(min(k, len(self.corpus)), dim=-1)
                scores.append(value.cpu().numpy())
                rows.append(row.cpu().numpy())
                del matrix, q, value, row
        self.guard.check()
        return np.concatenate(scores), np.concatenate(rows), time.monotonic() - started

    def close(self):
        del self.corpus
        torch.cuda.empty_cache()


def filtered(rows, candidate_ids, histories, topk=500, strict=True):
    result = []
    for candidates, history in zip(rows, histories):
        blocked, seen, output = set(map(int, history)), set(), []
        for row in candidates:
            note = int(candidate_ids[int(row)])
            if note not in blocked and note not in seen:
                seen.add(note)
                output.append(note)
                if len(output) == topk:
                    break
        if strict and len(output) != topk:
            raise RuntimeError('not enough legal candidates; increase overfetch before publishing')
        result.append(output)
    return result


def validate_rankings(requests, rankings, catalog, strict=True):
    lookup = set(map(int, catalog))
    for r, ranking in zip(requests, rankings):
        if len(ranking) != len(set(ranking)) or not set(ranking) <= lookup:
            raise AssertionError('duplicate or outside-corpus candidate')
        if set(ranking) & set(r.history):
            raise AssertionError('unfiltered history')
        if strict and len(ranking) != 500:
            raise AssertionError('Top500 underflow')
