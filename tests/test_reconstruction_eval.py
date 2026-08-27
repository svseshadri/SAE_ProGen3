import numpy as np
import torch

from scripts.evaluate_reconstruction import (
    additive_sae_intervention,
    compute_explained_variance,
    compute_mean_kl,
    compute_nmse,
    relative_nll_degradation_pct,
    top1_agreement,
)


def test_relative_degradation_and_kl_helpers():
    base = 10.0
    patched = 11.5
    assert np.isclose(relative_nll_degradation_pct(base, patched), 15.0)

    base_logits = torch.tensor([[[3.0, 1.0], [1.0, 3.0]]], dtype=torch.float32)
    patched_logits = torch.tensor([[[2.8, 1.2], [0.8, 3.2]]], dtype=torch.float32)
    kl = compute_mean_kl(base_logits, patched_logits)
    assert np.isfinite(kl)
    assert kl >= 0.0

    agreement = top1_agreement(base_logits, patched_logits)
    assert 0.0 <= agreement <= 1.0


def test_nmse_and_explained_variance_are_reasonable():
    x = torch.tensor([[1.0, 2.0], [2.0, 4.0]], dtype=torch.float32)
    x_hat = torch.tensor([[1.0, 1.0], [2.0, 4.0]], dtype=torch.float32)

    nmse = compute_nmse(x, x_hat)
    assert nmse >= 0.0
    assert np.isfinite(nmse)

    ev = compute_explained_variance(x, x_hat)
    assert ev >= 0.0
    assert ev <= 1.0
    assert np.isfinite(ev)


def test_additive_identity_intervention_is_exact_noop():
    torch.manual_seed(0)
    hidden = torch.randn(2, 5, dtype=torch.float32)
    dec_weight = torch.randn(5, 4, dtype=torch.float32)

    class DummySAE:
        def __init__(self):
            self.decoder = type("D", (), {"weight": dec_weight})()

        def __call__(self, x):
            z = torch.relu(x @ torch.eye(5, dtype=torch.float32))
            return {"z": z}

        def decode(self, z):
            return z @ dec_weight.T

    sae = DummySAE()
    target = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]], dtype=torch.float32)
    out = additive_sae_intervention(hidden, sae, feature_id=0, target_activation=target[:, 0], input_mean=None, input_std=None)
    assert torch.allclose(out, hidden, atol=1e-6, rtol=0.0)
