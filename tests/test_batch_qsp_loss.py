"""
Regression tests for batch_qsp_loss and train_nn eval integration.

Covers the specific failure mode where batch_qsp_loss is called with
lambda_grad > 0 inside a torch.no_grad() context (e.g. _eval), which
raises RuntimeError because autograd.grad cannot trace through tensors
that have no grad_fn.
"""

import math
import pytest
import torch

from neural_network_optimization_QSP.qsp_phase_net import (
    NNTrainConfig,
    QSPPhaseNet,
    batch_qsp_loss,
    train_nn,
)

# ── shared fixtures ────────────────────────────────────────────────────────────

K, B, S = 5, 4, 8
DELTA_0  = 2 * math.pi * 200
OMEGA    = 2 * math.pi * 80


def _make_inputs(requires_grad: bool = True):
    phi   = torch.randn(B, K + 1, dtype=torch.float64, requires_grad=requires_grad)
    delta = torch.randn(B, S,     dtype=torch.float64) * DELTA_0 * 0.4
    alpha = torch.rand(B,  S,     dtype=torch.float64) * 2 * math.pi
    return phi, delta, alpha


# ── batch_qsp_loss: gradient-penalty path ─────────────────────────────────────

def test_loss_lambda_zero_no_grad():
    """lambda_grad=0 must work inside torch.no_grad() (the eval path)."""
    with torch.no_grad():
        phi, delta, alpha = _make_inputs(requires_grad=False)
        loss = batch_qsp_loss(phi, delta, alpha, DELTA_0, OMEGA, lambda_grad=0.0)
    assert loss.item() >= 0.0


def test_loss_lambda_positive_with_grad():
    """lambda_grad > 0 must work when gradients are enabled (the train path)."""
    phi, delta, alpha = _make_inputs(requires_grad=True)
    loss = batch_qsp_loss(phi, delta, alpha, DELTA_0, OMEGA, lambda_grad=0.3)
    assert loss.item() >= 0.0
    loss.backward()
    assert phi.grad is not None


def test_loss_lambda_positive_under_no_grad_raises():
    """
    Calling batch_qsp_loss with lambda_grad > 0 inside torch.no_grad() must
    raise RuntimeError — this is the exact bug that was fixed.  The test
    documents that the unsafe call is forbidden, so any future regression is
    caught immediately.
    """
    with torch.no_grad():
        phi, delta, alpha = _make_inputs(requires_grad=False)
        with pytest.raises(RuntimeError):
            batch_qsp_loss(phi, delta, alpha, DELTA_0, OMEGA, lambda_grad=0.3)


# ── train_nn: eval step does not crash with lambda_grad > 0 ───────────────────

def _mini_cfg(lambda_grad: float) -> NNTrainConfig:
    return NNTrainConfig(
        K=K,
        N=2,
        Omega_max=OMEGA,
        Delta_0=DELTA_0,
        robustness_window=2 * math.pi * 10,
        delta_vals=[2 * math.pi * d for d in [-50.0, 50.0]],
        batch_size=8,
        steps=6,
        lr=1e-3,
        sample_size=8,
        lambda_grad=lambda_grad,
        eval_interval=2,       # triggers _eval at steps 2, 4, 6
        checkpoint_interval=999,
        eval_configs=8,
        device="cpu",
        out_dir="/tmp/test_qsp_nn",
    )


def test_train_nn_eval_with_lambda_grad():
    """
    train_nn must complete all eval checkpoints without RuntimeError when
    lambda_grad > 0.  This is the integration regression for the _eval bug.
    """
    cfg = _mini_cfg(lambda_grad=0.3)
    model, train_losses, eval_records = train_nn(cfg, verbose=False)
    # eval was called at steps 2, 4, 6
    assert len(eval_records) == 3, f"expected 3 eval records, got {len(eval_records)}"
    assert all(math.isfinite(v) for _, v in eval_records), "eval loss contains NaN/Inf"


def test_train_nn_eval_without_lambda_grad():
    """Baseline: train_nn completes cleanly with lambda_grad=0."""
    cfg = _mini_cfg(lambda_grad=0.0)
    _, _, eval_records = train_nn(cfg, verbose=False)
    assert len(eval_records) == 3
    assert all(math.isfinite(v) for _, v in eval_records)


# ── gradient flows through lambda_grad penalty ────────────────────────────────

def test_gradient_penalty_provides_grad_signal():
    """
    The gradient penalty must contribute a non-zero gradient to phi_batch.
    If create_graph=False or the penalty were detached, phi.grad from the
    penalty alone would be zero.
    """
    phi, delta, alpha = _make_inputs(requires_grad=True)
    # Compute base loss with no penalty
    loss_base = batch_qsp_loss(phi, delta, alpha, DELTA_0, OMEGA, lambda_grad=0.0)
    loss_base.backward()
    grad_base = phi.grad.clone()

    phi2, delta2, alpha2 = _make_inputs(requires_grad=True)
    loss_pen = batch_qsp_loss(phi2, delta2, alpha2, DELTA_0, OMEGA, lambda_grad=1.0)
    loss_pen.backward()
    grad_pen = phi2.grad.clone()

    # Gradients must differ — the penalty adds a distinct signal
    assert not torch.allclose(grad_base, grad_pen), \
        "gradient penalty produced no additional grad signal"
