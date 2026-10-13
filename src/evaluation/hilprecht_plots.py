"""Plots for the paper-inspired Hilprecht et al. (2019) attacks.

Every figure is labelled as an adaptation of arXiv:1906.03006 to our VAE
experiments (OUR RESULT), never as the paper's own numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.evaluation.hilprecht_mia import PAPER_LABEL  # noqa: E402

ATTACK_TITLES = {
    "reconstruction": "Reconstruction attack",
    "mc_pca": "MC attack (PCA)",
    "mc_hog": "MC attack (HOG)",
}
MEMBER_C, NONMEMBER_C = "#c0392b", "#2c7fb8"


def display_values(attack: str, scores: np.ndarray, n_samples: int | None) -> tuple[np.ndarray, str]:
    """Human-readable transform of the membership score for histograms."""
    if attack == "reconstruction":
        return -scores, "mean reconstruction error ||D(z)-x||_2  (lower = member)"
    counts = scores * (n_samples or 1)
    return np.log10(1.0 + counts), "log10(1 + #samples in U_eps(x))  (higher = member)"


def _hist(ax: plt.Axes, mem: np.ndarray, non: np.ndarray, xlabel: str) -> None:
    lo = float(min(mem.min(), non.min()))
    hi = float(max(mem.max(), non.max()))
    if hi <= lo:
        hi = lo + 1.0
    bins = np.linspace(lo, hi, 40)
    ax.hist(non, bins=bins, alpha=0.55, color=NONMEMBER_C, label=f"non-members (n={len(non)})", density=True)
    ax.hist(mem, bins=bins, alpha=0.55, color=MEMBER_C, label=f"members (n={len(mem)})", density=True)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel("density", fontsize=7)
    ax.tick_params(labelsize=7)


def plot_attack_results(
    attack: str,
    label: str,
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    fpr: np.ndarray,
    tpr: np.ndarray,
    metrics: dict[str, Any],
    n_samples: int | None,
    path: Path,
) -> None:
    """Per model x attack: member vs non-member score distribution + ROC."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mem, xl = display_values(attack, member_scores, n_samples)
    non, _ = display_values(attack, nonmember_scores, n_samples)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _hist(axes[0], mem, non, xl)
    axes[0].legend(fontsize=7)
    axes[0].set_title("Member vs non-member scores", fontsize=9)
    axes[1].plot(fpr, tpr, color="C0", label=f"AUC = {metrics['auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], ":", color="gray", label="random guess")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC", fontsize=9)
    axes[1].legend(fontsize=7)
    fig.suptitle(f"{label}: {ATTACK_TITLES[attack]}\n{PAPER_LABEL}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_paper_comparison(runs: list[dict[str, Any]], attacks: list[str], out_dir: Path) -> list[Path]:
    """Consolidated figures: rows = attacks, columns = models.

    `runs` items: {"label": str, "attacks": {attack: {"member": arr,
    "nonmember": arr, "fpr": arr, "tpr": arr, "metrics": dict, "n_samples": int|None}}}
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [r["label"] for r in runs]
    written: list[Path] = []

    # 1. score-distribution grid
    fig, axes = plt.subplots(len(attacks), len(runs), figsize=(3.6 * len(runs), 2.9 * len(attacks)), squeeze=False)
    for i, a in enumerate(attacks):
        for j, r in enumerate(runs):
            ax = axes[i][j]
            res = r["attacks"].get(a)
            if res is None:
                ax.axis("off")
                continue
            mem, xl = display_values(a, res["member"], res.get("n_samples"))
            non, _ = display_values(a, res["nonmember"], res.get("n_samples"))
            _hist(ax, mem, non, xl)
            m = res["metrics"]
            ax.set_title(
                f"{ATTACK_TITLES[a]} | {r['label']}\nAUC {m['auc']:.3f}  single {m['paper_protocol']['single_mi_accuracy_mean']:.2f}"
                f"  set {m['paper_protocol']['set_mi_accuracy_mean']:.2f}",
                fontsize=8,
            )
            if i == 0 and j == 0:
                ax.legend(fontsize=6)
    fig.suptitle(f"Member vs non-member membership scores\n{PAPER_LABEL}", fontsize=10)
    fig.tight_layout()
    p = out_dir / "paper_style_score_distributions.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)

    # 2. ROC grid (one panel per attack, one curve per model)
    fig, axes = plt.subplots(1, len(attacks), figsize=(4.4 * len(attacks), 4.2), squeeze=False)
    for i, a in enumerate(attacks):
        ax = axes[0][i]
        for j, r in enumerate(runs):
            res = r["attacks"].get(a)
            if res is None:
                continue
            ax.plot(res["fpr"], res["tpr"], color=f"C{j}", label=f"{r['label']} (AUC {res['metrics']['auc']:.3f})")
        ax.plot([0, 1], [0, 1], ":", color="gray", label="random guess")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(ATTACK_TITLES[a], fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"Attack ROC curves - {PAPER_LABEL}", fontsize=9)
    fig.tight_layout()
    p = out_dir / "paper_style_roc_curves.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)

    # 3. Figure-5-style accuracy bars (single MI / set MI, 50% = random guess)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.8 / max(len(runs), 1)
    x = np.arange(len(attacks))
    for k, (key, title) in enumerate((("single_mi", "Single membership inference"), ("set_mi", "Set membership inference"))):
        ax = axes[k]
        for j, r in enumerate(runs):
            means, stds = [], []
            for a in attacks:
                res = r["attacks"].get(a)
                pp = res["metrics"]["paper_protocol"] if res else None
                means.append(pp[f"{key}_accuracy_mean"] * 100 if pp else np.nan)
                stds.append(pp[f"{key}_accuracy_std"] * 100 if pp else 0.0)
            ax.bar(x + (j - (len(runs) - 1) / 2) * width, means, width, yerr=stds, capsize=3,
                   color=f"C{j}", label=r["label"])
        ax.axhline(50, ls="--", color="gray", lw=1, label="random guess (50%)")
        ax.set_xticks(x)
        ax.set_xticklabels([ATTACK_TITLES[a] for a in attacks], fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_ylabel("accuracy (%)")
        pp0 = runs[0]["attacks"][attacks[0]]["metrics"]["paper_protocol"]
        ax.set_title(f"{title} (M={pp0['M']}, {pp0['trials']} trials)", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"Paper-style attack accuracy (cf. Fig. 5 of arXiv:1906.03006)\n{PAPER_LABEL}", fontsize=9)
    fig.tight_layout()
    p = out_dir / "paper_style_accuracy_bars.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)

    # 4. consolidated AUC table (attacks x models)
    auc = np.array([[r["attacks"][a]["metrics"]["auc"] if a in r["attacks"] else np.nan for r in runs] for a in attacks])
    fig, ax = plt.subplots(figsize=(1.9 * len(runs) + 2.2, 0.75 * len(attacks) + 1.6))
    im = ax.imshow(auc, cmap="RdYlGn_r", vmin=0.4, vmax=1.0)
    for i in range(len(attacks)):
        for j in range(len(runs)):
            r = runs[j]["attacks"].get(attacks[i])
            if r is None:
                continue
            m = r["metrics"]
            ax.text(j, i, f"AUC {m['auc']:.3f}\nadv {m['advantage_max_tpr_minus_fpr']:.3f}\nset {m['paper_protocol']['set_mi_accuracy_mean']:.2f}",
                    ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(len(attacks)))
    ax.set_yticklabels([ATTACK_TITLES[a] for a in attacks], fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, label="ROC-AUC")
    ax.set_title(f"Attack comparison - {PAPER_LABEL}", fontsize=8)
    fig.tight_layout()
    p = out_dir / "paper_style_comparison_table.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)
    return written


# --------------------------------------------------------------------------- #
# results-summary figures (also usable from a saved summary.csv)
# --------------------------------------------------------------------------- #
def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _se(r: dict[str, Any], std_key: str) -> float:
    """Standard error of the mean accuracy over the random M-trials."""
    n = _f(r.get("paper_trials")) if r.get("paper_trials") not in (None, "") else 100.0
    return _f(r[std_key]) / np.sqrt(max(n, 1.0))


def plot_summary_rows(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    """Plot the attack x model results table (ROC-AUC, single-MI, set-MI accuracy).

    `rows` are the rows of paper_mia/summary.csv (dicts or csv.DictReader rows).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    models = list(dict.fromkeys(r["model"] for r in rows))
    attacks = [a for a in ATTACK_TITLES if any(r["attack"] == a for r in rows)]
    get = {(r["model"], r["attack"]): r for r in rows}
    written: list[Path] = []

    panels = [
        ("ROC-AUC (± 95% bootstrap CI)", "auc", lambda r: (_f(r["auc"]) - _f(r["auc_ci_low"]), _f(r["auc_ci_high"]) - _f(r["auc"])), 1.0),
        ("Single-MI accuracy (± s.e. over trials)", "single_mi_acc", lambda r: (_se(r, "single_mi_acc_std"),) * 2, 1.0),
        ("Set-MI accuracy (± s.e. over trials)", "set_mi_acc", lambda r: (_se(r, "set_mi_acc_std"),) * 2, 1.0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    width = 0.8 / max(len(models), 1)
    x = np.arange(len(attacks))
    for ax, (title, key, err, ymax) in zip(axes, panels):
        for j, m in enumerate(models):
            vals, lo, hi = [], [], []
            for a in attacks:
                r = get.get((m, a))
                vals.append(_f(r[key]) if r else np.nan)
                e = err(r) if r else (0.0, 0.0)
                lo.append(max(e[0], 0.0) if np.isfinite(e[0]) else 0.0)
                hi.append(max(e[1], 0.0) if np.isfinite(e[1]) else 0.0)
            pos = x + (j - (len(models) - 1) / 2) * width
            ax.bar(pos, vals, width, yerr=[lo, hi], capsize=3, color=f"C{j}", label=m)
            for px, v, e in zip(pos, vals, hi):
                if np.isfinite(v):
                    ax.text(px, v + e + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.axhline(0.5, ls="--", color="gray", lw=1, label="random guess")
        ax.set_xticks(x)
        ax.set_xticklabels([ATTACK_TITLES[a] for a in attacks], fontsize=8)
        ax.set_ylim(0, ymax + 0.12)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Membership inference attack results (higher = more leakage; 0.5 = random guess)\n" + PAPER_LABEL,
                 fontsize=9)
    fig.tight_layout()
    p = out_dir / "paper_style_results_summary.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # rendered version of the printed table: AUC (single / set), coloured by AUC
    cell_text, cell_col = [], []
    cmap = plt.get_cmap("RdYlGn_r")
    for a in attacks:
        row_t, row_c = [], []
        for m in models:
            r = get.get((m, a))
            if r is None:
                row_t.append("-")
                row_c.append("white")
                continue
            auc = _f(r["auc"])
            row_t.append(f"{auc:.3f}\n({_f(r['single_mi_acc']):.2f} / {_f(r['set_mi_acc']):.2f})")
            row_c.append(cmap(np.clip((auc - 0.4) / 0.6, 0, 1), alpha=0.55))
        cell_text.append(row_t)
        cell_col.append(row_c)
    fig, ax = plt.subplots(figsize=(2.3 * len(models) + 2.2, 0.6 * len(attacks) + 0.9))
    ax.axis("off")
    tab = ax.table(cellText=cell_text, cellColours=cell_col, rowLabels=[ATTACK_TITLES[a] for a in attacks],
                   colLabels=models, cellLoc="center", loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1.0, 2.6)
    ax.set_title("ROC-AUC  (single-MI acc / set-MI acc)\n" + PAPER_LABEL, fontsize=8)
    p = out_dir / "paper_style_results_table.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(p)
    return written


def replot_from_dir(paper_mia_dir: Path) -> list[Path]:
    """Re-draw the summary figures from <experiment>/paper_mia/summary.csv (no re-run)."""
    import csv

    with open(Path(paper_mia_dir) / "summary.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return plot_summary_rows(rows, Path(paper_mia_dir) / "plots")


if __name__ == "__main__":  # python -m src.evaluation.hilprecht_plots results/<exp>/paper_mia
    import sys

    for path in replot_from_dir(Path(sys.argv[1])):
        print("wrote", path)
