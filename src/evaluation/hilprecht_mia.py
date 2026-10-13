"""Paper-inspired/adapted implementation of the reconstruction and Monte Carlo
membership inference attacks of

    Hilprecht, Haerterich & Bernau (2019), "Monte Carlo and Reconstruction
    Membership Inference Attacks against Generative Models",
    arXiv:1906.03006 (PoPETs 2019, issue 4).

This is NOT an exact reproduction of that paper: our models are the
unconditional MNIST VAEs of papers/DP_VAE.pdf (vanilla / DP-SGD / L-VAE),
trained on 9,999 binarised images + one white outlier, not the paper's
conditional DCGAN-style VAE trained on a random 10% subset.

What follows the paper (VERIFIED against arXiv:1906.03006, Sections 3-4):
  * MC-eps score  f(x) = 1/n * sum_i 1{ g_i in U_eps(x) },
    U_eps(x) = {x' : d(x, x') <= eps}, g_i ~ generator.
  * MC-d score    f(x) = -1/n * sum_i 1{ g_i in U_eps(x) } * log d(g_i, x).
  * eps heuristics: median heuristic  eps = median_i min_j d(x_i, g_j)
    (the paper's default after Table 2) and percentile heuristic
    (eps = p-th percentile of all d(x_i, g_j), p in {0.01, 0.1, 1}%).
  * d = Euclidean distance after PCA (40 components for MNIST, PCA fitted on
    held-out test data) or after HOG feature extraction.
  * Reconstruction attack (VAE only):
    f_rec(x) = -1/n * sum_i || D(z_i) - x ||,  z_i ~ N(E_mu(x), E_Sigma(x)).
  * Single MI accuracy: score 2M records (M members + M non-members), label
    the M highest as members, accuracy = fraction of them that are members.
  * Set MI accuracy: of the two M-sets, pick the one contributing more of
    the top-M records (ties broken uniformly at random).

ASSUMED / ADAPTED (the paper does not specify, or our setup differs):
  * ||.|| in f_rec is the L2 norm on decoder Bernoulli means (sigmoid of
    logits); the training-loss variant (BCE) is reported as a secondary score.
  * HOG parameters (orientations=9, 7x7 cells, 2x2 blocks, L2-Hys) - the paper
    names scikit-image but gives no parameters.
  * PCA is fitted on the MNIST test set with the non-member targets REMOVED,
    so the attack's feature map never sees the records it is scored on.
  * log(0) in MC-d is avoided with log(max(d, 1e-8)) (the paper's continuous
    form adds a small delta; the discrete Eq. (2) omits it).
  * eps is chosen once over all 2*N targets (N = 1000 by default), not
    re-chosen inside every M=100 trial.
  * For the percentile heuristic with very large n, the percentile is
    estimated from the first generated batches (capped number of distances).
  * Default MC budget n = 1e5 (paper: 1e6) and n_rec = 300 (the paper notes
    300 already gives good accuracy; its tables use 1e6). Both configurable.

Memory: generated samples are produced, featurised and scored batch by batch.
At no point is the full Monte Carlo sample set held as one image tensor.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F

PAPER_REF = "Hilprecht et al. 2019, arXiv:1906.03006"
PAPER_LABEL = f"Adapted from {PAPER_REF} - OUR RESULT (not the paper's numbers)"


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class HOGConfig:
    orientations: int = 9
    pixels_per_cell: tuple[int, int] = (7, 7)
    cells_per_block: tuple[int, int] = (2, 2)
    block_norm: str = "L2-Hys"


@dataclass
class PaperAttackConfig:
    mc_samples: int = 100_000
    mc_batch_size: int = 256
    pca_components: int = 40
    epsilon: str = "median"
    rec_samples: int = 300
    postprocess: str = "sigmoid"
    seed: int = 42
    paper_m: int = 100
    paper_trials: int = 100
    feature_cache_mb: float = 512.0
    percentile_max_distances: int = 20_000_000
    hog: HOGConfig = field(default_factory=HOGConfig)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["epsilon_parsed"] = parse_epsilon(self.epsilon)
        return out


def parse_epsilon(spec: str | float) -> tuple[str, float | None]:
    """'median' | 'p0.1' / 'percentile:0.1' (percent) | '<float>' fixed radius."""
    s = str(spec).strip().lower()
    if s in ("median", "median_heuristic"):
        return "median", None
    for prefix in ("percentile:", "pct:", "p"):
        if s.startswith(prefix):
            p = float(s[len(prefix):])
            if not 0.0 < p < 100.0:
                raise ValueError(f"percentile must be in (0, 100), got {p}")
            return "percentile", p
    r = float(s)
    if r <= 0:
        raise ValueError("fixed epsilon radius must be > 0")
    return "fixed", r


# --------------------------------------------------------------------------- #
# batched Monte Carlo generation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def iter_generated_batches(
    model: torch.nn.Module,
    n: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    postprocess: str = "sigmoid",
) -> Iterator[torch.Tensor]:
    """Yield n generated images in batches of <= batch_size (on `device`).

    z is drawn from a private CPU generator seeded with `seed`, so every model
    with the same latent dimension receives the same latent draws (common
    random numbers) and a second pass reproduces the first exactly.
    The global torch RNG is NOT consumed.
    """
    model.eval()
    gen_z = torch.Generator().manual_seed(int(seed))
    gen_b = torch.Generator().manual_seed(int(seed) + 1)
    latent = int(model.latent_dim)
    done = 0
    while done < n:
        b = min(batch_size, n - done)
        z = torch.randn(b, latent, generator=gen_z)
        probs = torch.sigmoid(model.decoder(z.to(device)))
        if postprocess == "threshold":
            x = (probs >= 0.5).float()
        elif postprocess == "bernoulli":
            x = torch.bernoulli(probs.cpu(), generator=gen_b).to(device)
        else:
            x = probs
        yield x
        done += b


# --------------------------------------------------------------------------- #
# feature maps
# --------------------------------------------------------------------------- #
class PCAProjector:
    """Linear PCA (SVD) fitted on a held-out image pool."""

    def __init__(self, n_components: int) -> None:
        self.n_components = int(n_components)
        self.mean: torch.Tensor | None = None
        self.components: torch.Tensor | None = None
        self.explained_variance_ratio: np.ndarray | None = None
        self.n_fit: int = 0

    def fit(self, images: torch.Tensor) -> "PCAProjector":
        x = images.flatten(1).double().cpu().numpy()
        mean = x.mean(0)
        _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
        k = min(self.n_components, vt.shape[0])
        var = s**2
        self.mean = torch.from_numpy(mean).float()
        self.components = torch.from_numpy(vt[:k]).float()
        self.explained_variance_ratio = (var[:k] / var.sum()).astype(np.float64)
        self.n_fit = int(x.shape[0])
        return self

    def transform(self, images: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None and self.components is not None, "fit first"
        dev = images.device
        x = images.flatten(1).float()
        return (x - self.mean.to(dev)) @ self.components.to(dev).T


def extract_pca_features(images: torch.Tensor, projector: PCAProjector) -> torch.Tensor:
    return projector.transform(images)


def hog_feature_dim(image_hw: tuple[int, int], cfg: HOGConfig) -> int:
    ncr, ncc = image_hw[0] // cfg.pixels_per_cell[0], image_hw[1] // cfg.pixels_per_cell[1]
    nbr, nbc = ncr - cfg.cells_per_block[0] + 1, ncc - cfg.cells_per_block[1] + 1
    return nbr * nbc * cfg.cells_per_block[0] * cfg.cells_per_block[1] * cfg.orientations


def extract_hog_features(images: torch.Tensor, cfg: HOGConfig, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Batched torch port of skimage.feature.hog (single channel, feature_vector=True).

    Reproduces skimage's central-difference gradients (zero at borders),
    hard orientation binning over [0, 180), per-cell magnitude averaging and
    block normalisation; tests/test_hilprecht_mia.py checks it against
    skimage numerically.
    """
    x = images.to(dtype)
    if x.dim() == 4:
        x = x[:, 0]
    bsz, h, w = x.shape
    cr, cc = cfg.pixels_per_cell
    br, bc = cfg.cells_per_block
    n_or = cfg.orientations
    g_row = torch.zeros_like(x)
    g_col = torch.zeros_like(x)
    g_row[:, 1:-1, :] = x[:, 2:, :] - x[:, :-2, :]
    g_col[:, :, 1:-1] = x[:, :, 2:] - x[:, :, :-2]
    mag = torch.hypot(g_col, g_row)
    orient = torch.remainder(torch.rad2deg(torch.atan2(g_row, g_col)), 180.0)
    edges = torch.tensor([180.0 / n_or * i for i in range(1, n_or)], dtype=dtype, device=x.device)
    bins = torch.bucketize(orient.contiguous(), edges, right=True)  # bin i <=> [i*w, (i+1)*w)
    ncr, ncc = h // cr, w // cc
    if ncr < br or ncc < bc:
        raise ValueError("image too small for the chosen HOG cell/block size")
    contrib = F.one_hot(bins, n_or).to(dtype) * mag.unsqueeze(-1)
    contrib = contrib[:, : ncr * cr, : ncc * cc]
    hist = contrib.reshape(bsz, ncr, cr, ncc, cc, n_or).sum(dim=(2, 4)) / (cr * cc)
    nbr, nbc = ncr - br + 1, ncc - bc + 1
    blocks = torch.stack(
        [torch.stack([hist[:, i : i + nbr, j : j + nbc] for j in range(bc)], dim=3) for i in range(br)], dim=3
    )  # (B, nbr, nbc, br, bc, n_or)
    flat = blocks.reshape(bsz, nbr, nbc, -1)
    eps = 1e-5
    norm = cfg.block_norm
    if norm == "L1":
        out = flat / (flat.abs().sum(-1, keepdim=True) + eps)
    elif norm == "L1-sqrt":
        out = torch.sqrt(flat / (flat.abs().sum(-1, keepdim=True) + eps))
    elif norm == "L2":
        out = flat / torch.sqrt((flat**2).sum(-1, keepdim=True) + eps**2)
    elif norm == "L2-Hys":
        out = flat / torch.sqrt((flat**2).sum(-1, keepdim=True) + eps**2)
        out = torch.clamp(out, max=0.2)
        out = out / torch.sqrt((out**2).sum(-1, keepdim=True) + eps**2)
    else:
        raise ValueError(f"unknown block_norm {norm}")
    return out.reshape(bsz, -1)


# --------------------------------------------------------------------------- #
# Monte Carlo score
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_mc_membership_score(
    target_feats: torch.Tensor,
    sample_feature_batches: Callable[[], Iterator[torch.Tensor]],
    epsilon: str,
    feature_cache_mb: float = 512.0,
    percentile_max_distances: int = 20_000_000,
    log_floor: float = 1e-8,
) -> dict[str, Any]:
    """MC-eps and MC-d scores (Hilprecht et al. Eq. 1-2) with streamed samples.

    `sample_feature_batches()` must return a fresh iterator over feature
    batches of generated samples and must be deterministic (same samples on
    every call), because eps heuristics need a first pass over all samples.
    Generated features are cached between passes only while they fit in
    `feature_cache_mb`; otherwise the second pass regenerates them.
    """
    mode, value = parse_epsilon(epsilon)
    t = target_feats.float()
    n_t = t.shape[0]
    cache: list[torch.Tensor] | None = []
    cache_bytes = 0.0
    limit = feature_cache_mb * 1024**2
    min_d = torch.full((n_t,), float("inf"), device=t.device)
    probe: list[torch.Tensor] = []
    probe_n = 0
    n_samples = 0

    def _score(eps: float, batches: Iterator[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, int]:
        # float64 accumulators live on CPU (MPS has no float64)
        cnt = torch.zeros(n_t, dtype=torch.float64)
        logsum = torch.zeros(n_t, dtype=torch.float64)
        n = 0
        for f in batches:
            d = torch.cdist(t, f.float())
            inside = d <= eps
            cnt += inside.sum(1).cpu().double()
            logsum += (inside * torch.log(d.clamp_min(log_floor))).sum(1).cpu().double()
            n += f.shape[0]
        return cnt, logsum, n

    if mode == "fixed":
        assert value is not None
        eps = float(value)
        cnt, logsum, n_samples = _score(eps, sample_feature_batches())
        # nearest distance as a diagnostic (second cheap pass, regenerated)
        for f in sample_feature_batches():
            min_d = torch.minimum(min_d, torch.cdist(t, f.float()).min(1).values)
    else:
        for f in sample_feature_batches():
            d = torch.cdist(t, f.float())
            min_d = torch.minimum(min_d, d.min(1).values)
            n_samples += f.shape[0]
            if mode == "percentile" and probe_n * n_t < percentile_max_distances:
                probe.append(d.flatten().cpu())
                probe_n += f.shape[0]
            if cache is not None:
                cache_bytes += f.numel() * f.element_size()
                if cache_bytes <= limit:
                    cache.append(f)
                else:
                    cache = None
        if mode == "median":
            eps = float(min_d.median())  # median_i min_j d(x_i, g_j)  (paper Eq. 4)
        else:
            assert value is not None
            eps = float(np.percentile(torch.cat(probe).numpy(), value))
        it = iter(cache) if cache is not None else sample_feature_batches()
        cnt, logsum, n2 = _score(eps, it)
        assert n2 == n_samples, "generator is not deterministic between passes"

    mc_eps = (cnt / n_samples).numpy()
    mc_d = (-logsum / n_samples).numpy()
    return {
        "mc_eps": mc_eps,
        "mc_d": mc_d,
        "count": cnt.numpy(),
        "nearest_distance": min_d.cpu().numpy(),
        "epsilon": eps,
        "epsilon_mode": mode,
        "epsilon_param": value,
        "n_samples": n_samples,
        "percentile_estimated_from_n_distances": int(probe_n * n_t) if mode == "percentile" else None,
        "features_cached_between_passes": cache is not None if mode != "fixed" else False,
    }


# --------------------------------------------------------------------------- #
# reconstruction score
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_reconstruction_scores(
    model: torch.nn.Module,
    targets: torch.Tensor,
    n_rec: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """f_rec(x) = -1/n sum_i ||D(z_i) - x||_2, z_i ~ N(mu(x), diag(exp(logvar(x)))).

    Decoding is chunked so that at most ~batch_size images are decoded at once.
    Noise comes from a private CPU generator (common random numbers across
    models); the global RNG is not consumed.
    """
    model.eval()
    gen = torch.Generator().manual_seed(int(seed))
    tb = max(1, min(64, batch_size))
    r_chunk = max(1, batch_size // tb)
    latent = int(model.latent_dim)
    l2_all, bce_all, l2_mu_all = [], [], []
    for i in range(0, targets.shape[0], tb):
        x = targets[i : i + tb].to(device).float()
        b = x.shape[0]
        mu, logvar = model.encoder(x)
        std = torch.exp(0.5 * logvar)
        # float64 accumulators live on CPU (MPS has no float64)
        l2_sum = torch.zeros(b, dtype=torch.float64)
        bce_sum = torch.zeros(b, dtype=torch.float64)
        done = 0
        while done < n_rec:
            r = min(r_chunk, n_rec - done)
            eps = torch.randn(r, b, latent, generator=gen).to(device)
            z = mu.unsqueeze(0) + std.unsqueeze(0) * eps
            logits = model.decoder(z.reshape(r * b, latent)).reshape(r, b, *x.shape[1:])
            xr = x.unsqueeze(0).expand_as(logits)
            l2_sum += (torch.sigmoid(logits) - xr).flatten(2).norm(dim=2).sum(0).cpu().double()
            bce_sum += F.binary_cross_entropy_with_logits(logits, xr, reduction="none").flatten(2).sum(2).sum(0).cpu().double()
            done += r
        l2_mu = (torch.sigmoid(model.decoder(mu)) - x).flatten(1).norm(dim=1)
        l2_all.append(l2_sum / n_rec)
        bce_all.append(bce_sum / n_rec)
        l2_mu_all.append(l2_mu.cpu())
    l2 = torch.cat(l2_all).numpy()
    bce = torch.cat(bce_all).numpy()
    return {
        "score_l2": -l2,  # higher = more likely member
        "score_bce": -bce,
        "mean_l2_error": l2,
        "mean_bce_error": bce,
        "l2_error_at_mu": torch.cat(l2_mu_all).numpy(),
    }


# --------------------------------------------------------------------------- #
# paper evaluation protocol (single / set MI accuracy)
# --------------------------------------------------------------------------- #
def paper_mi_accuracy(
    member_scores: np.ndarray, nonmember_scores: np.ndarray, m: int, trials: int, seed: int
) -> dict[str, Any]:
    """Single and set MI accuracy over `trials` random draws of M members + M non-members."""
    rng = np.random.default_rng(seed)
    m = int(min(m, len(member_scores), len(nonmember_scores)))
    single, set_acc = [], []
    for _ in range(trials):
        im = rng.choice(len(member_scores), m, replace=False)
        jn = rng.choice(len(nonmember_scores), m, replace=False)
        s = np.concatenate([member_scores[im], nonmember_scores[jn]])
        order = np.lexsort((rng.random(2 * m), -s))  # by score desc, random tie-break
        n_mem = int((order[:m] < m).sum())
        single.append(n_mem / m)
        if 2 * n_mem > m:
            set_acc.append(1.0)
        elif 2 * n_mem == m:
            set_acc.append(float(rng.random() < 0.5))
        else:
            set_acc.append(0.0)
    single_a, set_a = np.asarray(single), np.asarray(set_acc)
    return {
        "M": m,
        "trials": int(trials),
        "single_mi_accuracy_mean": float(single_a.mean()),
        "single_mi_accuracy_std": float(single_a.std()),
        "set_mi_accuracy_mean": float(set_a.mean()),
        "set_mi_accuracy_std": float(set_a.std()),
    }


def top_n_rule_rates(y: np.ndarray, s: np.ndarray, seed: int) -> dict[str, float]:
    """Paper decision rule on the full pool: the n_members highest scores are 'members'."""
    rng = np.random.default_rng(seed)
    k = int(y.sum())
    order = np.lexsort((rng.random(len(s)), -s))
    pred = np.zeros(len(s), dtype=bool)
    pred[order[:k]] = True
    pos = y == 1
    tpr = float((pred & pos).sum() / max(pos.sum(), 1))
    fpr = float((pred & ~pos).sum() / max((~pos).sum(), 1))
    return {"tpr_top_n_rule": tpr, "fpr_top_n_rule": fpr, "accuracy_top_n_rule": float((pred == pos).mean())}
