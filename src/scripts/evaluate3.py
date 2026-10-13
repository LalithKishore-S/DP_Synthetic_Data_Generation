#!/usr/bin/env python3
"""Evaluate one or more trained VAE checkpoints (vanilla / DP-SGD / L-VAE).

Replaces the original single-checkpoint evaluate.py. What is new:

PRIVACY
  * Exact ROC (sklearn.roc_curve) instead of 200 linspace thresholds.
  * The paper's attack (distance to nearest synthetic record) plus two
    Hilprecht-style variants (mean of k-NN distances, MC-eps neighbour count).
  * AUC with bootstrap CI, privacy advantage, TPR at low FPR.
  * Heuristic empirical-epsilon lower bound (Clopper-Pearson), directly
    comparable to the epsilon reported by DP-SGD.
  * Theoretical (eps, delta)-DP trade-off curve (Figure 2 green curve), read
    automatically from training_history.csv of DP-SGD runs.
  * Outlier audit for the all-white image (isolation percentile, near-white
    synthetic fraction, reconstruction BCE).
  * Sensitivity of attack AUC to the number of synthetic samples and to the
    sample representation (sigmoid / threshold / bernoulli).
  * Train-vs-test negative ELBO gap (white-box diagnostic, not the paper's
    threat model).

UTILITY (the paper only shows qualitative grids)
  * Reference CNN trained on real binarised MNIST: confidence, class
    histogram/entropy, Frechet distance in feature space ("FD_clf").
  * Real-train-vs-real-test FD as a noise floor.

LIPSCHITZ CHECK
  * Power-iteration estimate of the local spectral norm of the decoder Jacobian
    (a lower bound on the true Lipschitz constant) so L-VAE can be verified.

HILPRECHT ET AL. (2019) ATTACKS  [optional, added; arXiv:1906.03006]
  Paper-inspired/adapted implementation of the reconstruction and Monte Carlo
  membership inference attacks (see src/evaluation/hilprecht_mia.py for what
  is VERIFIED vs ASSUMED). Enabled with --run-paper-attacks or individually
  with --run-reconstruction-attack / --run-mc-pca / --run-mc-hog. Results go
  to results/<experiment_id>/paper_mia/. Without these flags the script's
  behaviour and outputs are unchanged.

Everything written is labelled OUR RESULT and goes to results/<experiment_id>/.

Example:
  python src/scripts/evaluate.py \
      --checkpoint results/exp_vanilla_mnist/models/vae_final.pt \
                   results/exp_dpsgd_mnist/models/vae_final.pt \
                   results/exp_lvae_gp_mnist/models/vae_final.pt \
      --labels vanilla dpsgd lvae_gp --experiment-id eval_compare \
      --n-synthetic 10000 1000 100000

  # same, plus the three Hilprecht et al. attacks
  python src/scripts/evaluate2.py --checkpoint ... --labels vanilla dpsgd lvae_gp \
      --experiment-id eval_compare --run-paper-attacks \
      --mc-samples 100000 --mc-batch-size 256 --pca-components 40 --epsilon median
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from scipy.stats import beta as beta_dist
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.mnist import _load_mnist_tensors, binarize, build_mnist_paper_data, sample_mia_sets  # noqa: E402
from src.evaluation.generate import generate_samples, generate_samples_batched  # noqa: E402
from src.evaluation.hilprecht_mia import (  # noqa: E402
    PAPER_LABEL,
    PAPER_REF,
    HOGConfig,
    PaperAttackConfig,
    PCAProjector,
    compute_mc_membership_score,
    compute_reconstruction_scores,
    extract_hog_features,
    extract_pca_features,
    iter_generated_batches,
    paper_mi_accuracy,
    parse_epsilon,
    top_n_rule_rates,
)
from src.evaluation.hilprecht_plots import (  # noqa: E402
    plot_attack_results,
    plot_paper_comparison,
    plot_summary_rows,
    replot_from_dir,
)
from src.evaluation.plots import save_image_grid  # noqa: E402
from src.models.losses import elbo_loss  # noqa: E402
from src.training.trainer import load_checkpoint  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.environment import collect_environment  # noqa: E402
from src.utils.experiment import create_experiment_dir  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _abs(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _to_py(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# --------------------------------------------------------------------------- #
# membership inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def nearest_distance(targets: torch.Tensor, synth: torch.Tensor, chunk: int = 256) -> np.ndarray:
    t = targets.flatten(1).float().cpu()
    s = synth.flatten(1).float().cpu()
    out = [torch.cdist(t[i : i + chunk], s).min(dim=1).values for i in range(0, t.shape[0], chunk)]
    return torch.cat(out).numpy()


@torch.no_grad()
def distance_features(
    targets: torch.Tensor,
    synth: torch.Tensor,
    k: int = 5,
    chunk: int = 256,
    eps_percentile: float = 1.0,
    n_probe: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    """nn1, mean k-NN distance and MC-eps neighbour count (Hilprecht et al. [17] style)."""
    t = targets.flatten(1).float().cpu()
    s = synth.flatten(1).float().cpu()
    k = min(k, s.shape[0])
    gen = torch.Generator().manual_seed(seed)
    probe = t[torch.randperm(t.shape[0], generator=gen)[:n_probe]]
    eps = float(np.percentile(torch.cdist(probe, s).flatten().numpy(), eps_percentile))
    nn1, knn, cnt = [], [], []
    for i in range(0, t.shape[0], chunk):
        d = torch.cdist(t[i : i + chunk], s)
        nn1.append(d.min(dim=1).values)
        knn.append(d.topk(k, dim=1, largest=False).values.mean(dim=1))
        cnt.append((d <= eps).sum(dim=1))
    return {
        "nn1": torch.cat(nn1).numpy(),
        "knn": torch.cat(knn).numpy(),
        "cnt": torch.cat(cnt).numpy(),
        "eps": eps,
    }


def _cp_upper(k: int, n: int, alpha: float) -> float:
    """One-sided Clopper-Pearson upper bound on an error rate."""
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(1.0 - alpha, k + 1, n - k))


def empirical_epsilon_lb(y: np.ndarray, s: np.ndarray, delta: float, conf: float) -> float:
    """Heuristic empirical eps lower bound from the attack's error rates.

    (eps, delta)-DP implies  1 - FNR <= e^eps * FPR + delta  and
                             1 - FPR <= e^eps * FNR + delta.
    Using upper confidence bounds on FPR and FNR gives a lower bound on eps.
    NOT corrected for scanning many thresholds -> treat as indicative only.
    """
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    a = (1.0 - conf) / 2.0
    best = 0.0
    for thr in np.unique(np.quantile(s, np.linspace(0, 1, 201))):
        pred = s >= thr
        fp = int((pred & ~pos).sum())
        fn = int((~pred & pos).sum())
        fpr_ub = _cp_upper(fp, n_neg, a)
        fnr_ub = _cp_upper(fn, n_pos, a)
        for num, den in ((1 - fnr_ub - delta, fpr_ub), (1 - fpr_ub - delta, fnr_ub)):
            if num > 0 and den > 0:
                best = max(best, math.log(num / den))
    return best


def roc_metrics(
    y: np.ndarray, s: np.ndarray, delta: float, conf: float, n_boot: int, rng: np.random.Generator
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    fpr, tpr, _ = roc_curve(y, s)
    auc = float(roc_auc_score(y, s))
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        boots.append(roc_auc_score(y[idx], s[idx]))
    if boots:
        lo, hi = np.percentile(boots, [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100])
    else:
        lo = hi = float("nan")
    metrics = {
        "auc": auc,
        "auc_ci_low": float(lo),
        "auc_ci_high": float(hi),
        "advantage_max_tpr_minus_fpr": float(np.max(tpr - fpr)),
        "tpr_at_fpr_1pct": float(np.interp(0.01, fpr, tpr)),
        "tpr_at_fpr_10pct": float(np.interp(0.10, fpr, tpr)),
        "empirical_eps_lower_bound_heuristic": empirical_epsilon_lb(y, s, delta, conf),
    }
    return metrics, fpr, tpr


def dp_tradeoff_curve(alpha: np.ndarray, eps: float, delta: float) -> np.ndarray:
    """Optimal type-II error of an (eps, delta)-DP mechanism at type-I error alpha."""
    return np.maximum.reduce(
        [np.zeros_like(alpha), 1 - delta - np.exp(eps) * alpha, np.exp(-eps) * (1 - delta - alpha)]
    )


def read_dp_epsilon(ckpt: Path, config: dict[str, Any]) -> tuple[float | None, float | None]:
    if config["train"].get("variant") != "dpsgd":
        return None, None
    hist = ckpt.parent.parent / "metrics" / "training_history.csv"
    if not hist.exists():
        return None, None
    with open(hist, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not rows[-1].get("epsilon"):
        return None, None
    return float(rows[-1]["epsilon"]), float(config["dpsgd"]["delta"])


# --------------------------------------------------------------------------- #
# reference classifier + utility metrics
# --------------------------------------------------------------------------- #
class RefCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
        )
        self.head = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def get_reference_classifier(results_root: Path, data_dir: Path, thr: float, device: torch.device,
                             smoke: bool, seed: int) -> RefCNN:
    """Train (or load cached) CNN on REAL binarised MNIST. Used only for utility metrics."""
    path = results_root / "models" / ("reference_classifier_smoke.pt" if smoke else "reference_classifier_mnist.pt")
    clf = RefCNN()
    if path.exists():
        clf.load_state_dict(torch.load(path, map_location="cpu")["state"])
        return clf.to(device).eval()
    set_seed(seed)
    x_tr, y_tr, x_te, y_te = _load_mnist_tensors(data_dir, download=True)
    x_tr, x_te = binarize(x_tr, thr), binarize(x_te, thr)
    if smoke:
        x_tr, y_tr = x_tr[:5000], y_tr[:5000]
    clf.to(device).train()
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    epochs = 1 if smoke else 3
    for _ in range(epochs):
        perm = torch.randperm(x_tr.shape[0])
        for i in range(0, x_tr.shape[0], 256):
            idx = perm[i : i + 256]
            loss = F.cross_entropy(clf(x_tr[idx].to(device)), y_tr[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    clf.eval()
    with torch.no_grad():
        acc = float((clf(x_te.to(device)).argmax(1).cpu() == y_te).float().mean())
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": clf.state_dict(), "test_accuracy": acc}, path)
    print(f"[reference classifier] test accuracy on real binarised MNIST: {acc:.4f}")
    return clf


@torch.no_grad()
def clf_forward(clf: RefCNN, x: torch.Tensor, device: torch.device, bs: int = 1000) -> tuple[np.ndarray, torch.Tensor]:
    feats, logits = [], []
    for i in range(0, x.shape[0], bs):
        f = clf.features(x[i : i + bs].to(device))
        feats.append(f.cpu())
        logits.append(clf.head(f).cpu())
    return torch.cat(feats).numpy().astype(np.float64), torch.cat(logits)


def frechet_distance(f1: np.ndarray, f2: np.ndarray) -> float:
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def utility_metrics(clf: RefCNN, synth_bin: torch.Tensor, train_imgs: torch.Tensor,
                    test_imgs: torch.Tensor, device: torch.device) -> dict[str, Any]:
    f_s, logits = clf_forward(clf, synth_bin, device)
    probs = logits.softmax(1)
    conf, pred = probs.max(1).values, probs.argmax(1)
    p = torch.bincount(pred, minlength=10).float()
    p = p / p.sum()
    nz = p[p > 0]
    entropy = float(-(nz * nz.log()).sum() / math.log(10))
    f_te, _ = clf_forward(clf, test_imgs, device)
    f_tr, _ = clf_forward(clf, train_imgs, device)
    return {
        "mean_confidence": float(conf.mean()),
        "frac_confidence_below_0.9": float((conf < 0.9).float().mean()),
        "class_histogram": [float(v) for v in p],
        "class_entropy_normalised": entropy,
        "min_class_fraction": float(p.min()),
        "FD_clf_synthetic_vs_test": frechet_distance(f_s, f_te),
        "FD_clf_real_train_vs_test_floor": frechet_distance(f_tr, f_te),
    }


# --------------------------------------------------------------------------- #
# model diagnostics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def mean_elbo(model: nn.Module, x: torch.Tensor, device: torch.device, bs: int = 500) -> dict[str, float]:
    tot = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    for i in range(0, x.shape[0], bs):
        xb = x[i : i + bs].to(device)
        out = model(xb)
        terms = elbo_loss(out["logits"], xb, out["mu"], out["logvar"])
        for key in tot:
            tot[key] += float(terms[key]) * xb.shape[0]
    return {f"neg_elbo_{k}" if k == "loss" else k: v / x.shape[0] for k, v in tot.items()}


def estimate_local_lipschitz(decoder: nn.Module, z: torch.Tensor, iters: int = 30) -> np.ndarray:
    """Per-sample top singular value of the decoder Jacobian (power iteration).

    This is a LOWER bound on the global Lipschitz constant over the probed region.
    """
    from torch.func import jvp, vjp

    f = lambda t: decoder(t)  # noqa: E731
    _, vjp_fn = vjp(f, z)
    v = torch.randn_like(z)
    v = v / v.norm(dim=1, keepdim=True)
    sigma = torch.zeros(z.shape[0])
    for _ in range(iters):
        _, u = jvp(f, (z,), (v,))
        sigma = u.flatten(1).norm(dim=1)
        u = u / (sigma.view(-1, *([1] * (u.dim() - 1))) + 1e-12)
        (w,) = vjp_fn(u)
        v = w / (w.norm(dim=1, keepdim=True) + 1e-12)
    return sigma.detach().numpy()


# --------------------------------------------------------------------------- #
# Hilprecht et al. (2019) attacks  [ADDED - paper-inspired/adapted]
# --------------------------------------------------------------------------- #
PAPER_ATTACK_DIRS = {"reconstruction": "reconstruction_attack", "mc_pca": "mc_pca", "mc_hog": "mc_hog"}


def enabled_paper_attacks(args: argparse.Namespace) -> list[str]:
    on = []
    if args.run_paper_attacks or args.run_reconstruction_attack:
        on.append("reconstruction")
    if args.run_paper_attacks or args.run_mc_pca:
        on.append("mc_pca")
    if args.run_paper_attacks or args.run_mc_hog:
        on.append("mc_hog")
    return on


def build_paper_attack_config(args: argparse.Namespace, seed: int) -> PaperAttackConfig:
    """One config object shared by every model (fairness)."""
    cfg = PaperAttackConfig(
        mc_samples=args.mc_samples,
        mc_batch_size=args.mc_batch_size,
        pca_components=args.pca_components,
        epsilon=args.epsilon,
        rec_samples=args.rec_samples,
        postprocess=args.postprocess,
        seed=seed,
        paper_m=args.paper_m,
        paper_trials=args.paper_trials,
        feature_cache_mb=args.mc_feature_cache_mb,
        hog=HOGConfig(
            orientations=args.hog_orientations,
            pixels_per_cell=tuple(args.hog_pixels_per_cell),
            cells_per_block=tuple(args.hog_cells_per_block),
            block_norm=args.hog_block_norm,
        ),
    )
    if args.smoke:
        cfg.mc_samples = min(cfg.mc_samples, 512)
        cfg.rec_samples = min(cfg.rec_samples, 4)
        cfg.paper_trials = min(cfg.paper_trials, 20)
    parse_epsilon(cfg.epsilon)  # fail fast on a bad spec
    return cfg


def _fingerprint(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def _image_keys(images: torch.Tensor) -> set[bytes]:
    return {row.tobytes() for row in (images.flatten(1).cpu().numpy() >= 0.5)}


def get_pca_projector(split: Any, nonmember_idx: np.ndarray, n_components: int, cache: dict) -> tuple[PCAProjector, dict]:
    """PCA fitted on the MNIST test set MINUS the non-member targets (no target leakage)."""
    key = ("pca", _fingerprint(np.sort(nonmember_idx)), int(split.test_images.shape[0]), n_components)
    if key not in cache:
        pool = np.setdiff1d(np.arange(split.test_images.shape[0]), nonmember_idx)
        assert np.intersect1d(pool, nonmember_idx).size == 0
        proj = PCAProjector(n_components).fit(split.test_images[pool])
        info = {
            "fit_on": "MNIST test set minus non-member targets",
            "n_fit": int(pool.size),
            "n_components": int(proj.components.shape[0]),
            "explained_variance_ratio_total": float(proj.explained_variance_ratio.sum()),
            "overlap_with_nonmember_targets": 0,
        }
        cache[key] = (proj, info)
    return cache[key]


def check_separation(split: Any, sets: dict[str, Any]) -> dict[str, Any]:
    """Members from the model's training set, non-members from MNIST test; report overlap."""
    mi, ni = np.asarray(sets["member_indices"]), np.asarray(sets["nonmember_indices"])
    n_train, n_test = int(split.train_images.shape[0]), int(split.test_images.shape[0])
    assert mi.min() >= 0 and mi.max() < n_train, "member index outside the training set"
    assert ni.min() >= 0 and ni.max() < n_test, "non-member index outside MNIST test"
    assert len(np.unique(mi)) == len(mi) and len(np.unique(ni)) == len(ni), "duplicate target indices"
    train_keys = _image_keys(split.train_images)
    non_keys = [row.tobytes() for row in (sets["nonmember_images"].flatten(1).numpy() >= 0.5)]
    return {
        "members_source": "training set of this checkpoint (incl. white outlier if configured)",
        "nonmembers_source": "MNIST test split (never used for training)",
        "generated_source": "fresh samples z ~ N(0, I) from a private seeded generator",
        "n_members": int(len(mi)),
        "n_nonmembers": int(len(ni)),
        "nonmember_images_pixel_identical_to_a_training_image": int(sum(k in train_keys for k in non_keys)),
        "member_indices_fingerprint": _fingerprint(np.sort(mi)),
        "nonmember_indices_fingerprint": _fingerprint(np.sort(ni)),
    }


def _attack_dir(exp_dir: Path, attack: str) -> Path:
    d = exp_dir / "paper_mia" / PAPER_ATTACK_DIRS[attack]
    d.mkdir(parents=True, exist_ok=True)
    return d


def score_paper_attack(
    attack: str,
    label: str,
    ckpt: Path,
    y: np.ndarray,
    score: np.ndarray,
    target_ids: dict[str, np.ndarray],
    extra_cols: dict[str, np.ndarray],
    record: dict[str, Any],
    cfg: PaperAttackConfig,
    args: argparse.Namespace,
    exp_dir: Path,
    n_samples: int | None,
) -> dict[str, Any]:
    """Shared metrics + outputs for one model x attack. Higher score = member."""
    assert np.all(np.isfinite(score)), f"{attack}/{label}: non-finite membership scores"
    rng = np.random.default_rng(cfg.seed)  # same bootstrap draws for every model
    m, fpr, tpr = roc_metrics(y, score, args.delta, args.confidence, args.n_bootstrap, rng)
    m["tpr_at_fpr_0.1pct"] = float(np.interp(0.001, fpr, tpr))
    m.update(top_n_rule_rates(y, score, cfg.seed))
    m["paper_protocol"] = paper_mi_accuracy(score[y == 1], score[y == 0], cfg.paper_m, cfg.paper_trials, cfg.seed)
    m["member_score_mean"] = float(score[y == 1].mean())
    m["nonmember_score_mean"] = float(score[y == 0].mean())

    out_dir = _attack_dir(exp_dir, attack)
    cols = {"role": np.where(y == 1, "member", "nonmember"), "dataset_index": target_ids["dataset_index"],
            "is_member": y, "membership_score": score, **extra_cols}
    with open(out_dir / f"scores_{label}.csv", "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(list(cols))
        for row in zip(*cols.values()):
            w.writerow([v.item() if hasattr(v, "item") else v for v in row])
    np.savetxt(out_dir / f"roc_{label}.csv", np.column_stack([fpr, tpr]), delimiter=",", header="fpr,tpr", comments="")
    full = {
        "source": "OUR RESULT",
        "method": f"Paper-inspired/adapted implementation of {PAPER_REF}",
        "model": label,
        "checkpoint": str(ckpt),
        "attack": attack,
        "score_direction": "higher membership_score = more likely a training member",
        "n_generated_samples": n_samples,
        "batch_size": cfg.mc_batch_size,
        "pca_components": cfg.pca_components if attack == "mc_pca" else None,
        "hog_params": vars(cfg.hog) if attack == "mc_hog" else None,
        "epsilon_spec": cfg.epsilon if attack.startswith("mc_") else None,
        "random_seed": cfg.seed,
        **record,
        "metrics": m,
    }
    with open(out_dir / f"metrics_{label}.json", "w", encoding="utf-8") as handle:
        json.dump(full, handle, indent=2, default=_to_py)
    plot_attack_results(attack, label, score[y == 1], score[y == 0], fpr, tpr, m, n_samples,
                        out_dir / f"plot_{label}.png")
    full["_raw"] = {"member": score[y == 1], "nonmember": score[y == 0], "fpr": fpr, "tpr": tpr, "n_samples": n_samples}
    return full


def run_reconstruction_attack(model: nn.Module, label: str, ckpt: Path, targets: torch.Tensor, y: np.ndarray,
                              target_ids: dict[str, np.ndarray], cfg: PaperAttackConfig, args: argparse.Namespace,
                              device: torch.device, exp_dir: Path) -> dict[str, Any]:
    rec = compute_reconstruction_scores(model, targets, cfg.rec_samples, cfg.mc_batch_size, cfg.seed, device)
    record = {
        "n_reconstruction_samples_per_target": cfg.rec_samples,
        "distance": "L2 between decoder Bernoulli means sigmoid(D(z)) and x (ASSUMED norm)",
        "secondary_score_auc_bce": float(roc_auc_score(y, rec["score_bce"])),
        "diagnostic_auc_single_recon_at_mu": float(roc_auc_score(y, -rec["l2_error_at_mu"])),
    }
    extra = {"mean_l2_error": rec["mean_l2_error"], "mean_bce_error": rec["mean_bce_error"],
             "l2_error_at_mu": rec["l2_error_at_mu"]}
    return score_paper_attack("reconstruction", label, ckpt, y, rec["score_l2"], target_ids, extra, record,
                              cfg, args, exp_dir, None)


def _run_mc_attack(attack: str, featurise: Any, model: nn.Module, label: str, ckpt: Path, targets: torch.Tensor,
                   y: np.ndarray, target_ids: dict[str, np.ndarray], cfg: PaperAttackConfig,
                   args: argparse.Namespace, device: torch.device, exp_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    t_feat = torch.cat([featurise(targets[i : i + cfg.mc_batch_size].to(device))
                        for i in range(0, targets.shape[0], cfg.mc_batch_size)])

    def batches():  # deterministic: identical samples on every call
        for x in iter_generated_batches(model, cfg.mc_samples, cfg.mc_batch_size, cfg.seed, device, cfg.postprocess):
            yield featurise(x)

    mc = compute_mc_membership_score(t_feat, batches, cfg.epsilon, feature_cache_mb=cfg.feature_cache_mb,
                                     percentile_max_distances=cfg.percentile_max_distances)
    record = {
        **record,
        "feature_dim": int(t_feat.shape[1]),
        "sample_representation": cfg.postprocess,
        "epsilon_mode": mc["epsilon_mode"],
        "epsilon_param": mc["epsilon_param"],
        "epsilon_radius": mc["epsilon"],
        "percentile_estimated_from_n_distances": mc["percentile_estimated_from_n_distances"],
        "frac_targets_with_nonzero_count": float((mc["count"] > 0).mean()),
        "secondary_score_auc_mc_d": float(roc_auc_score(y, mc["mc_d"])),
        "diagnostic_auc_nearest_distance": float(roc_auc_score(y, -mc["nearest_distance"])),
        "primary_score": "MC-eps: fraction of generated samples within eps of the target",
    }
    extra = {"count_in_eps_ball": mc["count"], "mc_d_score": mc["mc_d"], "nearest_distance": mc["nearest_distance"]}
    return score_paper_attack(attack, label, ckpt, y, mc["mc_eps"], target_ids, extra, record, cfg, args, exp_dir,
                              mc["n_samples"])


def run_mc_pca_attack(model: nn.Module, label: str, ckpt: Path, targets: torch.Tensor, y: np.ndarray,
                      target_ids: dict[str, np.ndarray], projector: PCAProjector, pca_info: dict,
                      cfg: PaperAttackConfig, args: argparse.Namespace, device: torch.device,
                      exp_dir: Path) -> dict[str, Any]:
    return _run_mc_attack("mc_pca", lambda x: extract_pca_features(x, projector), model, label, ckpt, targets, y,
                          target_ids, cfg, args, device, exp_dir, {"distance": "Euclidean in PCA space", "pca": pca_info})


def run_mc_hog_attack(model: nn.Module, label: str, ckpt: Path, targets: torch.Tensor, y: np.ndarray,
                      target_ids: dict[str, np.ndarray], cfg: PaperAttackConfig, args: argparse.Namespace,
                      device: torch.device, exp_dir: Path) -> dict[str, Any]:
    return _run_mc_attack("mc_hog", lambda x: extract_hog_features(x, cfg.hog), model, label, ckpt, targets, y,
                          target_ids, cfg, args, device, exp_dir,
                          {"distance": "Euclidean in HOG space (torch port of skimage.feature.hog)"})


def run_paper_attacks(model: nn.Module, label: str, ckpt: Path, split: Any, sets: dict[str, Any],
                      targets: torch.Tensor, y: np.ndarray, args: argparse.Namespace, device: torch.device,
                      exp_dir: Path, cache: dict) -> dict[str, Any]:
    cfg: PaperAttackConfig = args._paper_cfg
    target_ids = {"dataset_index": np.concatenate([sets["member_indices"], sets["nonmember_indices"]])}
    separation = check_separation(split, sets)
    out: dict[str, Any] = {"separation_check": separation, "attacks": {}}
    for attack in args._paper_attacks:
        print(f"    [paper MIA] {attack} ...")
        if attack == "reconstruction":
            res = run_reconstruction_attack(model, label, ckpt, targets, y, target_ids, cfg, args, device, exp_dir)
        elif attack == "mc_pca":
            proj, info = get_pca_projector(split, np.asarray(sets["nonmember_indices"]), cfg.pca_components, cache)
            res = run_mc_pca_attack(model, label, ckpt, targets, y, target_ids, proj, info, cfg, args, device, exp_dir)
        else:
            res = run_mc_hog_attack(model, label, ckpt, targets, y, target_ids, cfg, args, device, exp_dir)
        out["attacks"][attack] = res
        m = res["metrics"]
        print(f"      AUC={m['auc']:.4f} adv={m['advantage_max_tpr_minus_fpr']:.4f} "
              f"single={m['paper_protocol']['single_mi_accuracy_mean']:.3f} set={m['paper_protocol']['set_mi_accuracy_mean']:.3f}")
    return out


def write_paper_attack_summary(results: list[dict[str, Any]], exp_dir: Path, cfg: PaperAttackConfig,
                               attacks: list[str]) -> None:
    root = exp_dir / "paper_mia"
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    runs, rows = [], []
    for r in results:
        pm = r.get("_paper_mia")
        if pm is None:
            continue
        run = {"label": r["label"], "attacks": {}}
        for a, res in pm["attacks"].items():
            raw = res["_raw"]
            run["attacks"][a] = {**raw, "metrics": res["metrics"]}
            m = res["metrics"]
            rows.append({
                "model": r["label"], "checkpoint": r["checkpoint"], "attack": a,
                "n_generated_samples": res["n_generated_samples"],
                "n_reconstruction_samples": res.get("n_reconstruction_samples_per_target"),
                "batch_size": cfg.mc_batch_size,
                "pca_components": res["pca_components"],
                "hog_params": json.dumps(res["hog_params"]) if res["hog_params"] else None,
                "epsilon_spec": res["epsilon_spec"], "epsilon_radius": res.get("epsilon_radius"),
                "random_seed": cfg.seed,
                "auc": m["auc"], "auc_ci_low": m["auc_ci_low"], "auc_ci_high": m["auc_ci_high"],
                "advantage": m["advantage_max_tpr_minus_fpr"],
                "tpr_at_fpr_0.1pct": m["tpr_at_fpr_0.1pct"], "tpr_at_fpr_1pct": m["tpr_at_fpr_1pct"],
                "tpr_at_fpr_10pct": m["tpr_at_fpr_10pct"],
                "tpr_top_n_rule": m["tpr_top_n_rule"], "fpr_top_n_rule": m["fpr_top_n_rule"],
                "single_mi_acc": m["paper_protocol"]["single_mi_accuracy_mean"],
                "single_mi_acc_std": m["paper_protocol"]["single_mi_accuracy_std"],
                "set_mi_acc": m["paper_protocol"]["set_mi_accuracy_mean"],
                "set_mi_acc_std": m["paper_protocol"]["set_mi_accuracy_std"],
                "paper_m": m["paper_protocol"]["M"], "paper_trials": m["paper_protocol"]["trials"],
                "eps_lb_heuristic": m["empirical_eps_lower_bound_heuristic"],
            })
        runs.append(run)
    if not rows:
        return

    seps = {r["label"]: r["_paper_mia"]["separation_check"] for r in results if r.get("_paper_mia")}
    fp_m = {s["member_indices_fingerprint"] for s in seps.values()}
    fp_n = {s["nonmember_indices_fingerprint"] for s in seps.values()}
    fairness = {
        "same_attack_config_for_all_models": True,
        "same_member_indices_across_models": len(fp_m) == 1,
        "same_nonmember_indices_across_models": len(fp_n) == 1,
        "note": "member indices refer to each checkpoint's own training subset; they coincide when the "
                "checkpoints were trained with the same data seed / n_train",
        "per_model": seps,
    }
    if len(fp_m) != 1 or len(fp_n) != 1:
        print("WARNING: target sets differ between models (different training seeds / splits?) - see fairness_check.json")

    with open(root / "attack_config.json", "w", encoding="utf-8") as handle:
        json.dump({"paper": PAPER_REF, "label": PAPER_LABEL, "attacks": attacks, **cfg.to_dict()}, handle, indent=2,
                  default=_to_py)
    with open(root / "fairness_check.json", "w", encoding="utf-8") as handle:
        json.dump(fairness, handle, indent=2, default=_to_py)
    with open(root / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump({"source": "OUR RESULT", "method": f"Paper-inspired/adapted implementation of {PAPER_REF}",
                   "config": cfg.to_dict(), "fairness": fairness, "results": rows}, handle, indent=2, default=_to_py)

    labels = [run["label"] for run in runs]
    lines = [f"# {PAPER_LABEL}", "", "ROC-AUC (single-MI acc / set-MI acc)", "",
             "| attack | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for a in attacks:
        cells = []
        for run in runs:
            m = run["attacks"].get(a, {}).get("metrics")
            cells.append(f"{m['auc']:.3f} ({m['paper_protocol']['single_mi_accuracy_mean']:.2f} / "
                         f"{m['paper_protocol']['set_mi_accuracy_mean']:.2f})" if m else "-")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    (root / "comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_paper_comparison(runs, [a for a in attacks if any(a in run["attacks"] for run in runs)], plots)
    plot_summary_rows(rows, plots)  # the printed AUC / single / set table as figures

    print(f"\n=== PAPER-STYLE ATTACKS ({PAPER_LABEL}) ===")
    print("\n".join(lines[2:]))


# --------------------------------------------------------------------------- #
# per-checkpoint evaluation
# --------------------------------------------------------------------------- #
def evaluate_one(label: str, ckpt: Path, args: argparse.Namespace, device: torch.device,
                 exp_dir: Path, clf: RefCNN | None, split_cache: dict) -> dict[str, Any]:
    model, config = load_checkpoint(ckpt, device=device)
    seed = args.seed if args.seed is not None else int(config["train"]["seed"])
    set_seed(seed)
    rng = np.random.default_rng(seed)

    ds, ev = config["dataset"], config["eval"]
    key = (int(ds["n_train"]), float(ds["binarize_threshold"]), bool(ds["add_white_outlier"]), int(config["train"]["seed"]))
    if key not in split_cache:
        split_cache[key] = build_mnist_paper_data(
            data_dir=_abs(ds["data_dir"]), n_train=key[0], binarize_threshold=key[1],
            add_white_outlier=key[2], seed=key[3],
        )
    split = split_cache[key]

    n_members, n_non = int(ev["n_members"]), int(ev["n_nonmembers"])
    n_list = args.n_synthetic or [int(ev["n_synthetic"])]
    if args.smoke:
        n_list, n_members, n_non = [64], 16, 16
    n_primary, n_max = n_list[0], max(n_list)
    delta_attack = args.delta

    if args.gen_batch_size > 0:  # same z draw as generate_samples, decoder applied in chunks (memory-safe)
        probs = generate_samples_batched(model, n=n_max, device=device, batch_size=args.gen_batch_size)
    else:  # legacy: decode all n_max samples in one tensor
        probs = generate_samples(model, n=n_max, device=device, postprocess="sigmoid")
    reps = {"sigmoid": probs, "threshold": (probs >= 0.5).float(), "bernoulli": torch.bernoulli(probs)}
    rep = args.postprocess
    synth = reps[rep][:n_primary]

    sets = sample_mia_sets(split, n_members, n_non, seed=seed)
    members, nonmembers = sets["member_images"], sets["nonmember_images"]
    targets = torch.cat([members, nonmembers])
    y = np.concatenate([np.ones(n_members), np.zeros(n_non)]).astype(int)

    n_train = int(split.train_images.shape[0])
    out_pos = None
    if bool(ds["add_white_outlier"]):
        hit = np.where(sets["member_indices"] == n_train - 1)[0]
        out_pos = int(hit[0]) if len(hit) else None

    # ---- attacks (higher score = more likely member) ----
    feats = distance_features(targets, synth, seed=seed)
    scores = {
        "nn1_distance (paper)": -feats["nn1"],
        "knn5_mean_distance": -feats["knn"],
        "mc_eps_count": feats["cnt"].astype(float),
    }
    attacks, curves = {}, {}
    for name, s in scores.items():
        m, fpr, tpr = roc_metrics(y, s, delta_attack, args.confidence, args.n_bootstrap, rng)
        attacks[name] = m
        curves[name] = (fpr, tpr)
    fpr, tpr = curves["nn1_distance (paper)"]
    np.savetxt(exp_dir / "metrics" / f"mia_roc_{label}.csv",
               np.column_stack([fpr, 1 - tpr, tpr]), delimiter=",",
               header="type_i_fpr,type_ii_fnr,tpr", comments="")

    # ---- sensitivity of AUC (nn1) to n_synthetic and representation ----
    sweep: dict[str, dict[str, float]] = {}
    for r, imgs in reps.items():
        sweep[r] = {}
        for n in n_list:
            d = nearest_distance(targets, imgs[:n])
            sweep[r][str(n)] = float(roc_auc_score(y, -d))

    # ---- outlier audit ----
    outlier: dict[str, Any] = {}
    if out_pos is not None:
        white = split.outlier.unsqueeze(0)
        d_out = float(feats["nn1"][out_pos])
        nm_d = feats["nn1"][n_members:]
        near_white = float((probs[:n_primary].flatten(1).mean(1) > 0.9).float().mean())
        with torch.no_grad():
            mu, _ = model.encoder(white.to(device))
            logits = model.decoder(mu)
            rec_bce = float(F.binary_cross_entropy_with_logits(logits, white.to(device), reduction="sum"))
            rec_probs = torch.sigmoid(logits).cpu()
        outlier = {
            "nn1_distance_to_synthetic": d_out,
            "isolation_percentile_vs_nonmembers": float((nm_d < d_out).mean()),
            "synthetic_near_white_fraction": near_white,
            "reconstruction_bce_sum_z_eq_mu": rec_bce,
            "reconstruction_mean_pixel": float(rec_probs.mean()),
            "note": "isolation percentile near 1.0 => outlier is farther from synthetic data than almost all non-members",
        }
        save_image_grid(torch.cat([white, rec_probs]), exp_dir / "plots" / f"outlier_reconstruction_{label}.png",
                        nrow=2, title=f"{label}: white outlier (real | recon)")

    # ---- generalisation gap in neg. ELBO (white-box diagnostic) ----
    keep = torch.ones(n_members, dtype=torch.bool)
    if out_pos is not None:
        keep[out_pos] = False
    elbo = {
        "members_wo_outlier": mean_elbo(model, members[keep], device),
        "nonmembers": mean_elbo(model, nonmembers, device),
    }
    elbo["gap_nonmember_minus_member_neg_elbo"] = (
        elbo["nonmembers"]["neg_elbo_loss"] - elbo["members_wo_outlier"]["neg_elbo_loss"]
    )

    # ---- utility ----
    utility = None
    if clf is not None:
        utility = utility_metrics(clf, (probs[:n_primary] >= 0.5).float(), split.train_images, split.test_images, device)

    # ---- plots / samples ----
    save_image_grid(synth[: int(ev["n_sample_grid"])], exp_dir / "plots" / f"samples_{label}.png",
                    title=f"{label} ({rep})")
    np.save(exp_dir / "samples" / f"generated_{label}.npy", synth.numpy())

    # ---- Lipschitz estimate (last: moves a copy of the decoder to CPU) ----
    lipschitz: dict[str, Any] | None
    try:
        dec = copy.deepcopy(model.decoder).cpu().eval()
        d_lat = int(model.latent_dim)
        with torch.no_grad():
            mu_post, _ = model.encoder(members[:512].to(device))
        s_prior = estimate_local_lipschitz(dec, torch.randn(512, d_lat))
        s_post = estimate_local_lipschitz(dec, mu_post.cpu())
        lipschitz = {
            "configured_L": float(config.get("lvae", {}).get("lipschitz_L", float("nan"))),
            "lvae_method": config.get("lvae", {}).get("method"),
            "prior_z": {"median": float(np.median(s_prior)), "p95": float(np.percentile(s_prior, 95)), "max": float(s_prior.max())},
            "posterior_mu": {"median": float(np.median(s_post)), "p95": float(np.percentile(s_post, 95)), "max": float(s_post.max())},
            "note": "local spectral norm of d decoder / d z (logit space): a LOWER bound on the global Lipschitz constant",
        }
    except Exception as exc:  # pragma: no cover
        lipschitz = {"error": repr(exc)}

    # ---- Hilprecht et al. (2019) attacks [optional; own seeded RNGs, global RNG untouched] ----
    paper_mia = None
    if args._paper_attacks:
        paper_mia = run_paper_attacks(model, label, ckpt, split, sets, targets, y, args, device, exp_dir, split_cache)

    eps_dp, delta_dp = read_dp_epsilon(ckpt, config)
    result = {
        "source": "OUR RESULT",
        "label": label,
        "checkpoint": str(ckpt),
        "variant": config["train"].get("variant"),
        "seed": seed,
        "n_synthetic_primary": n_primary,
        "sample_representation": rep,
        "n_members": n_members,
        "n_nonmembers": n_non,
        "mc_eps_radius": feats["eps"],
        "attacks": attacks,
        "auc_nn1_by_representation_and_n_synthetic": sweep,
        "outlier_audit": outlier,
        "elbo_generalisation": elbo,
        "utility": utility,
        "lipschitz_estimate": lipschitz,
        "dpsgd_epsilon": eps_dp,
        "dpsgd_delta": delta_dp,
    }
    if paper_mia is not None:
        result["paper_mia_hilprecht"] = {
            "method": f"Paper-inspired/adapted implementation of {PAPER_REF}",
            "outputs": str(exp_dir / "paper_mia"),
            "separation_check": paper_mia["separation_check"],
            "attacks": {a: {k: v for k, v in res.items() if k != "_raw"} for a, res in paper_mia["attacks"].items()},
        }
    with open(exp_dir / "metrics" / f"eval_{label}.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=_to_py)
    result["_curve"] = (fpr, tpr)
    if paper_mia is not None:
        result["_paper_mia"] = paper_mia
    return result


# --------------------------------------------------------------------------- #
# comparison plots / summary
# --------------------------------------------------------------------------- #
def plot_comparison(results: list[dict[str, Any]], exp_dir: Path) -> None:
    alpha = np.linspace(0, 1, 501)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for i, r in enumerate(results):
        fpr, tpr = r["_curve"]
        c = f"C{i}"
        ax.plot(fpr, 1 - tpr, color=c, label=f"{r['label']} (AUC={r['attacks']['nn1_distance (paper)']['auc']:.3f})")
        if r["dpsgd_epsilon"] is not None:
            ax.plot(alpha, dp_tradeoff_curve(alpha, r["dpsgd_epsilon"], r["dpsgd_delta"]), "--", color=c,
                    label=f"{r['label']}: theory eps={r['dpsgd_epsilon']:.2f}")
    ax.plot([0, 1], [1, 0], ":", color="gray", label="random guess")
    ax.set_xlabel("Type I error (false alarm)")
    ax.set_ylabel("Type II error (missed member)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Monte Carlo MIA trade-off (OUR RESULT)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(exp_dir / "plots" / "mia_tradeoff_compare.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for i, r in enumerate(results):
        fpr, tpr = r["_curve"]
        ax.loglog(np.maximum(fpr, 1e-4), np.maximum(tpr, 1e-4), color=f"C{i}", label=r["label"])
    ax.plot([1e-4, 1], [1e-4, 1], ":", color="gray")
    ax.set_xlim(1e-3, 1)
    ax.set_ylim(1e-3, 1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Attack ROC, log-log (low-FPR regime)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(exp_dir / "plots" / "mia_roc_loglog.png", dpi=140)
    plt.close(fig)


def write_summary(results: list[dict[str, Any]], exp_dir: Path) -> None:
    rows = []
    for r in results:
        a = r["attacks"]["nn1_distance (paper)"]
        u = r["utility"] or {}
        o = r["outlier_audit"] or {}
        rows.append({
            "label": r["label"],
            "auc_nn1": a["auc"],
            "auc_ci_low": a["auc_ci_low"],
            "auc_ci_high": a["auc_ci_high"],
            "advantage": a["advantage_max_tpr_minus_fpr"],
            "tpr_at_fpr_1pct": a["tpr_at_fpr_1pct"],
            "eps_lb_heuristic": a["empirical_eps_lower_bound_heuristic"],
            "dpsgd_epsilon": r["dpsgd_epsilon"],
            "outlier_isolation_pct": o.get("isolation_percentile_vs_nonmembers"),
            "outlier_recon_bce": o.get("reconstruction_bce_sum_z_eq_mu"),
            "elbo_gap": r["elbo_generalisation"]["gap_nonmember_minus_member_neg_elbo"],
            "FD_clf": u.get("FD_clf_synthetic_vs_test"),
            "FD_clf_floor": u.get("FD_clf_real_train_vs_test_floor"),
            "class_entropy": u.get("class_entropy_normalised"),
            "mean_confidence": u.get("mean_confidence"),
            "lipschitz_p95_prior": (r["lipschitz_estimate"] or {}).get("prior_z", {}).get("p95"),
        })
    with open(exp_dir / "metrics" / "eval_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\n=== SUMMARY (OUR RESULT) ===")
    for row in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate saved VAE checkpoints (privacy + utility)")
    p.add_argument("--checkpoint", type=Path, nargs="+", default=None)
    p.add_argument("--replot-paper-mia", type=Path, default=None,
                   help="only re-draw summary plots from an existing results/<exp>/paper_mia folder, then exit")
    p.add_argument("--labels", nargs="*", default=None, help="one label per checkpoint")
    p.add_argument("--experiment-id", type=str, default="eval_run")
    p.add_argument("--n-synthetic", type=int, nargs="+", default=None,
                   help="first value is the primary pool; all values are used in the AUC sweep")
    p.add_argument("--postprocess", choices=["sigmoid", "threshold", "bernoulli"], default="sigmoid")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-bootstrap", type=int, default=200)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--delta", type=float, default=1e-4, help="delta used in the empirical-eps lower bound")
    p.add_argument("--no-utility", action="store_true", help="skip reference classifier / FD metrics")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--gen-batch-size", type=int, default=1024,
                   help="decoder batch size for the existing synthetic pool (same samples as before; "
                        "0 = legacy single-tensor generation)")

    g = p.add_argument_group("Hilprecht et al. 2019 attacks (arXiv:1906.03006, paper-inspired/adapted)")
    g.add_argument("--run-paper-attacks", action="store_true", help="run reconstruction + MC-PCA + MC-HOG")
    g.add_argument("--run-reconstruction-attack", action="store_true")
    g.add_argument("--run-mc-pca", action="store_true")
    g.add_argument("--run-mc-hog", action="store_true")
    g.add_argument("--mc-samples", type=int, default=100_000, help="Monte Carlo generated samples per model (paper: 1e6)")
    g.add_argument("--mc-batch-size", type=int, default=256, help="generation / decoding batch size")
    g.add_argument("--pca-components", type=int, default=40, help="paper: 40 for MNIST")
    g.add_argument("--epsilon", "--mc-epsilon", dest="epsilon", type=str, default="median",
                   help="MC neighbourhood radius: 'median' (paper default), 'p0.1' = 0.1th percentile of "
                        "all target-sample distances, or a fixed float radius")
    g.add_argument("--rec-samples", type=int, default=300, help="z draws per target in the reconstruction attack")
    g.add_argument("--hog-orientations", type=int, default=9)
    g.add_argument("--hog-pixels-per-cell", type=int, nargs=2, default=[7, 7])
    g.add_argument("--hog-cells-per-block", type=int, nargs=2, default=[2, 2])
    g.add_argument("--hog-block-norm", choices=["L1", "L1-sqrt", "L2", "L2-Hys"], default="L2-Hys")
    g.add_argument("--paper-m", type=int, default=100, help="M records per side in single/set MI trials (paper: 100)")
    g.add_argument("--paper-trials", type=int, default=100, help="random single/set MI trials")
    g.add_argument("--attack-seed", type=int, default=None, help="default: --seed, else first checkpoint's train seed")
    g.add_argument("--mc-feature-cache-mb", type=float, default=512.0,
                   help="cache generated features between the two MC passes up to this size, else regenerate")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.replot_paper_mia is not None:
        for path in replot_from_dir(_abs(args.replot_paper_mia)):
            print("wrote", path)
        return
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required")
    if args.labels and len(args.labels) != len(args.checkpoint):
        raise SystemExit("--labels must have one entry per --checkpoint")
    labels = args.labels or [c.resolve().parent.parent.name for c in args.checkpoint]

    device = get_device()
    first_cfg = load_checkpoint(args.checkpoint[0], device=torch.device("cpu"))[1]
    results_root = _abs(first_cfg["experiment"]["results_root"])
    seed0 = args.seed if args.seed is not None else int(first_cfg["train"]["seed"])
    set_seed(seed0)

    exp_dir = create_experiment_dir(results_root, args.experiment_id)
    with open(exp_dir / "logs" / "eval_args.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, default=_to_py)

    args._paper_attacks = enabled_paper_attacks(args)
    args._paper_cfg = None
    if args._paper_attacks:
        attack_seed = args.attack_seed if args.attack_seed is not None else seed0
        args._paper_cfg = build_paper_attack_config(args, attack_seed)
        print(f"[paper MIA] {args._paper_attacks} with {args._paper_cfg.to_dict()}")
    with open(exp_dir / "logs" / "environment.json", "w", encoding="utf-8") as handle:
        json.dump(collect_environment(), handle, indent=2)

    clf = None
    if not args.no_utility:
        ds = first_cfg["dataset"]
        clf = get_reference_classifier(results_root, _abs(ds["data_dir"]), float(ds["binarize_threshold"]),
                                       device, args.smoke, seed0)

    split_cache: dict = {}
    results = []
    for label, ckpt in zip(labels, args.checkpoint):
        print(f"--- evaluating {label}: {ckpt}")
        results.append(evaluate_one(label, ckpt, args, device, exp_dir, clf, split_cache))

    plot_comparison(results, exp_dir)
    write_summary(results, exp_dir)
    if args._paper_attacks:
        write_paper_attack_summary(results, exp_dir, args._paper_cfg, args._paper_attacks)
    print(f"\nWrote evaluation to {exp_dir}")


if __name__ == "__main__":
    main()
