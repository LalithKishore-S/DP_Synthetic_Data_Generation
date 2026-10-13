"""Tests for the paper-inspired Hilprecht et al. (2019) attacks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.generate import generate_samples, generate_samples_batched
from src.evaluation.hilprecht_mia import (
    HOGConfig,
    PCAProjector,
    compute_mc_membership_score,
    compute_reconstruction_scores,
    extract_hog_features,
    hog_feature_dim,
    iter_generated_batches,
    paper_mi_accuracy,
    parse_epsilon,
)
from src.models.vae import VAE


@pytest.mark.parametrize("cfg", [HOGConfig(), HOGConfig(orientations=8, pixels_per_cell=(4, 4), cells_per_block=(3, 3), block_norm="L2"),
                                 HOGConfig(block_norm="L1"), HOGConfig(block_norm="L1-sqrt")])
def test_hog_matches_skimage(cfg: HOGConfig) -> None:
    skf = pytest.importorskip("skimage.feature")
    rng = np.random.default_rng(0)
    imgs = np.concatenate([(rng.random((4, 28, 28)) > 0.6).astype(np.float64), rng.random((4, 28, 28))])
    ours = extract_hog_features(torch.from_numpy(imgs).unsqueeze(1), cfg, dtype=torch.float64).numpy()
    ref = np.stack([skf.hog(im, orientations=cfg.orientations, pixels_per_cell=cfg.pixels_per_cell,
                            cells_per_block=cfg.cells_per_block, block_norm=cfg.block_norm, feature_vector=True)
                    for im in imgs])
    assert ours.shape == ref.shape == (8, hog_feature_dim((28, 28), cfg))
    np.testing.assert_allclose(ours, ref, atol=1e-6)


def test_parse_epsilon() -> None:
    assert parse_epsilon("median") == ("median", None)
    assert parse_epsilon("p0.1") == ("percentile", 0.1)
    assert parse_epsilon("percentile:1") == ("percentile", 1.0)
    assert parse_epsilon("2.5") == ("fixed", 2.5)
    with pytest.raises(ValueError):
        parse_epsilon("p150")


def _toy():
    g = torch.Generator().manual_seed(0)
    members = torch.randn(50, 5, generator=g)
    nonmembers = torch.randn(50, 5, generator=g)
    # generated samples cluster around the members -> members must score higher
    samples = members.repeat(20, 1) + 0.05 * torch.randn(1000, 5, generator=g)
    return torch.cat([members, nonmembers]), samples


@pytest.mark.parametrize("eps", ["median", "p1", "0.5"])
@pytest.mark.parametrize("cache_mb", [512.0, 0.0])
def test_mc_score_separates_and_is_batch_invariant(eps: str, cache_mb: float) -> None:
    targets, samples = _toy()
    y = np.r_[np.ones(50), np.zeros(50)]

    def batches(bs):
        return lambda: (samples[i : i + bs] for i in range(0, len(samples), bs))

    a = compute_mc_membership_score(targets, batches(64), eps, feature_cache_mb=cache_mb)
    b = compute_mc_membership_score(targets, batches(333), eps, feature_cache_mb=cache_mb)
    np.testing.assert_allclose(a["mc_eps"], b["mc_eps"])
    assert a["n_samples"] == 1000 and np.all(np.isfinite(a["mc_d"]))
    assert a["mc_eps"][y == 1].mean() > a["mc_eps"][y == 0].mean()
    # brute-force check of the definition
    d = torch.cdist(targets, samples)
    np.testing.assert_allclose(a["count"], (d <= a["epsilon"]).sum(1).numpy())
    if eps == "median":
        assert a["epsilon"] == pytest.approx(float(d.min(1).values.median()))


def test_paper_accuracy_extremes() -> None:
    hi, lo = np.arange(100, 200.0), np.arange(0, 100.0)
    perfect = paper_mi_accuracy(hi, lo, 50, 30, 0)
    assert perfect["single_mi_accuracy_mean"] == 1.0 and perfect["set_mi_accuracy_mean"] == 1.0
    inverse = paper_mi_accuracy(lo, hi, 50, 30, 0)
    assert inverse["single_mi_accuracy_mean"] == 0.0 and inverse["set_mi_accuracy_mean"] == 0.0
    tied = paper_mi_accuracy(np.zeros(100), np.zeros(100), 50, 400, 0)
    assert 0.4 < tied["set_mi_accuracy_mean"] < 0.6 and 0.45 < tied["single_mi_accuracy_mean"] < 0.55


def test_generation_batched_and_deterministic() -> None:
    torch.manual_seed(0)
    model = VAE(latent_dim=8).eval()
    dev = torch.device("cpu")
    torch.manual_seed(123)
    ref = generate_samples(model, n=70, device=dev)
    torch.manual_seed(123)
    bat = generate_samples_batched(model, n=70, device=dev, batch_size=16)
    torch.testing.assert_close(ref, bat, atol=1e-6, rtol=1e-5)

    state = torch.get_rng_state()
    a = torch.cat(list(iter_generated_batches(model, 70, 16, seed=5, device=dev)))
    b = torch.cat(list(iter_generated_batches(model, 70, 16, seed=5, device=dev)))
    assert torch.equal(torch.get_rng_state(), state), "MC generation must not touch the global RNG"
    assert a.shape == (70, 1, 28, 28) and torch.equal(a, b)


def test_reconstruction_scores_finite_and_chunk_invariant() -> None:
    torch.manual_seed(0)
    model = VAE(latent_dim=8).eval()
    x = (torch.rand(10, 1, 28, 28) > 0.5).float()
    r = compute_reconstruction_scores(model, x, n_rec=6, batch_size=64, seed=1, device=torch.device("cpu"))
    assert r["score_l2"].shape == (10,) and np.all(np.isfinite(r["score_l2"])) and np.all(r["score_l2"] <= 0)


def test_pca_projector_matches_sklearn() -> None:
    from sklearn.decomposition import PCA

    g = torch.Generator().manual_seed(0)
    scales = torch.linspace(5, 0.5, 784)  # well-separated spectrum
    x = (torch.randn(300, 784, generator=g) * scales).reshape(300, 1, 28, 28)
    ours = PCAProjector(10).fit(x).transform(x[:20]).numpy()
    sk = PCA(10, svd_solver="full").fit(x.flatten(1).numpy()).transform(x[:20].flatten(1).numpy())
    np.testing.assert_allclose(np.abs(ours), np.abs(sk), atol=1e-3)  # component signs are arbitrary
