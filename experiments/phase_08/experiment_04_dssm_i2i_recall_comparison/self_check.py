"""Small synthetic checks; no Qilin data, GPU or test labels."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_08.experiment_04_dssm_i2i_recall_comparison.i2i import merge_request
from experiments.phase_08.experiment_04_dssm_i2i_recall_comparison.run import (
    combine,
)


def main():
    catalog = np.asarray([10, 11, 12, 13, 14], dtype=np.int64)
    rows = np.asarray([[0, 2, 3, 4, 1], [1, 3, 2, 4, 0]], dtype=np.int32)
    scores = np.asarray([[1.0, 0.8, 0.4, 0.2, 0.1],
                         [1.0, 0.9, 0.3, 0.2, 0.1]], dtype=np.float32)
    one = merge_request((10, 11), {10: 0, 11: 1}, catalog,
                        {0: 0, 1: 1}, rows, scores, 5, topk=3)
    two = merge_request((10, 11), {10: 0, 11: 1}, catalog,
                        {0: 0, 1: 1}, rows, scores, 5, topk=3)
    assert one == two == [13, 12, 14]
    assert not set(one) & {10, 11}
    primary = list(range(1, 501))
    secondary = list(range(451, 951))
    fused = combine(primary, secondary, 50, ())
    assert len(fused) == len(set(fused)) == 500
    assert fused[:450] == primary[:450]
    assert all(note in primary or note in secondary for note in fused)
    print("phase8-04 synthetic self-check passed")


if __name__ == "__main__":
    main()
