import math

import torch

from neural_network_optimization.data import sample_alpha, build_dataset, ALPHA_RANGE, EPS


def test_sample_alpha_range_and_shape():
    a = sample_alpha(1024, N_peaks=4)
    assert a.shape == (1024, 4)
    lo, hi = ALPHA_RANGE
    assert a.min().item() >= lo - 1e-12
    assert a.max().item() <= hi + 1e-12
    assert a.min().item() >= -EPS - 1e-12
    assert a.max().item() <= 4 * math.pi + EPS + 1e-12


def test_build_dataset_reproducible():
    t1, e1 = build_dataset(n_train=256, n_eval=64, seed=42)
    t2, e2 = build_dataset(n_train=256, n_eval=64, seed=42)
    assert torch.equal(t1, t2)
    assert torch.equal(e1, e2)

    # Different seed => different tensors
    t3, _ = build_dataset(n_train=256, n_eval=64, seed=1)
    assert not torch.equal(t1, t3)
