# Research Resources

## Primary Source

### Original Paper

File:

`papers/DP_VAE.pdf`

Bibliographic identity:

Benedikt Groß and Gerhard Wunder, *Differentially Private Synthetic Data Generation via Lipschitz-Regularised Variational Autoencoders*, arXiv:2304.11336v2 [cs.LG], 13 July 2023.

This is the primary source of truth for the implementation.

The complete paper (all 8 pages: abstract through references) was read for the methodology extraction in `docs/methodology.md`. There is no appendix and no official code release was found.

---

## Required Information to Extract

The paper was inspected for the items below. Details and classifications live in `docs/methodology.md`.

### Dataset

- Dataset name: MNIST — VERIFIED (Section V)
- Dataset source: Deng 2012 [23] — VERIFIED citation; host INFERRED as standard MNIST
- Number of samples: 10,000 train (9,999 random MNIST train + 1 all-white outlier) — VERIFIED (Section V-A)
- Number of features: \(28 \times 28 \times 1\) — VERIFIED
- Feature types: binarised pixels \(\{0,1\}\) — VERIFIED
- Missing values: not discussed
- Preprocessing: binarisation — VERIFIED; threshold rule ASSUMED
- Normalization: none beyond binarisation for experiments
- Encoding: none
- Train/validation/test split: no validation split; MIA uses 1,000 train members + 1,000 MNIST-test non-members — VERIFIED

### VAE Architecture

- Encoder: 4 conv (32, 64, 128, 32), \(3\times3\), stride 2; dense 100 and \(2 n_{\text{latent}}\) — VERIFIED (Section V-A)
- Decoder: dense 100 and \(7\cdot7\cdot32\); transposed conv 64, 128, 1; \(3\times3\), stride 2 — VERIFIED
- Activations: ReLU except output (none) — VERIFIED
- Latent dimension: symbol \(n_{\text{latent}}\) only — ASSUMED numeric value
- Latent distribution: Gaussian posterior, standard normal prior — VERIFIED
- Reparameterization: implied via [9] — INFERRED
- Output: unactivated decoder; BCE on binary MNIST — VERIFIED

### Mathematical Objective

- Reconstruction (experiments): binary cross-entropy — VERIFIED, Section V-A
- Reconstruction (theory): Gaussian MSE, eq. (9) — VERIFIED, theory only
- KL: \(\mathrm{KL}(q_\phi(z|x)\|p(z))\) in the unnumbered ELBO of Section III-A — VERIFIED
- VAE objective: average ELBO; constrained Lipschitz problem eq. (13) — VERIFIED
- Weighting \(\beta\): cited as literature, not used explicitly — ASSUMED \(\beta=1\)
- Extra term: gradient penalty like WGAN-GP [24], or spectral normalisation [25] — VERIFIED methods; coefficients ASSUMED

### Training

- Optimizer, learning rate, batch size, epochs, scheduler, init, seed: **not specified**
- Software: Python, TensorFlow 2.6, scikit-learn — VERIFIED (Section V-A)
- DP-SGD: optimiser as in Abadi et al. [8]; no \(\varepsilon,\delta\), noise, or clip — unspecified

### Generation

- \(z \sim p(z)\), decode — VERIFIED (Section III-A)
- Sample count and post-processing — unspecified

### Evaluation

- Monte Carlo MIA [17]; type I / type II trade-off vs threshold — VERIFIED (Sections II-C, V)
- Qualitative samples and outlier reconstruction — VERIFIED (Figures 1–3, Section V-B)
- Baselines: vanilla VAE and VAE-DPSGD — VERIFIED
- No numeric metric tables

---

# Source Priority

Use sources in this order:

1. `papers/DP_VAE.pdf`
2. Supplementary material, if available — **none found**
3. Official dataset documentation (MNIST / `torchvision`)
4. Official PyTorch documentation
5. Research papers cited by the original paper (listed below), used only to fill gaps the paper points to ([8], [9], [17], [24], [25])
6. Third-party implementations — **must not override the paper**. Unrelated “Lipschitz VAE” repos (e.g. Barrett / Anil lnets) implement a different paper.

---

# Research Rules

Do not assume that a standard VAE implementation matches the paper.

Do not copy hyperparameters from another implementation unless the paper itself does not specify them and the assumption is explicitly documented.

---

# External Resources

| Resource | Type | Purpose | Authority |
|---|---|---|---|
| `papers/DP_VAE.pdf` / arXiv:2304.11336v2 | Primary paper | Full methodology | Highest |
| [9] Kingma & Welling, *Auto-Encoding Variational Bayes*, arXiv:1312.6114 | Cited | Reparameterization, diagonal-Gaussian KL | High for unspecified VAE calculus |
| [8] Abadi et al., *Deep Learning with Differential Privacy*, CCS 2016 | Cited | DP-SGD baseline | High for DP-SGD mechanism; hparams still unspecified here |
| [17] Hilprecht, Härterich, Bernau, *Monte Carlo and Reconstruction Membership Inference Attacks*, PoPETs 2019 | Cited | MIA details beyond the paper’s one-paragraph description | Medium: paper summarises as nearest-synthetic thresholding |
| [23] Deng, MNIST, IEEE SPM 2012 | Cited | Dataset | High |
| [24] Gulrajani et al., Improved WGAN (gradient penalty), NeurIPS 2017 | Cited | Form of decoder gradient penalty | Medium: adapted, not copied as WGAN-GP \(\lambda=10\) unless documented ASSUMED |
| [25] Miyato et al., Spectral Normalization for GANs, arXiv:1802.05957 | Cited | Spectral norm on decoder layers | High for SN algorithm |
| [19] Higgins et al., \(\beta\)-VAE | Cited as literature | Not used unless paper had \(\beta\neq 1\) (it does not) | Do not adopt \(\beta\neq 1\) silently |
| TensorFlow 2.6 docs | Paper stack | Authors’ implementation framework | We use PyTorch (documented DEVIATION) |
| PyTorch / torchvision / Opacus | Implementation stack | Portable training, DP-SGD | Implementation only |
| scikit-learn | Paper eval stack | Monte Carlo MIA utilities | High for eval tooling |

---

# Open Questions

Unresolved after full-paper analysis. Each must stay labelled ASSUMED until evidence appears.

1. Numeric latent dimension \(n_{\text{latent}}\).
2. Convolution / transposed-convolution padding, output padding, and how three stride-2 decoder layers produce \(28\times28\) images.
3. Encoder Gaussian parameterisation (\(\log\sigma^2\) vs \(\log\sigma\) vs unconstrained \(\sigma\)).
4. Binarisation rule (hard threshold vs Bernoulli sampling; threshold value).
5. Optimiser, learning rate, batch size, epochs, scheduler, weight initialisation, seed.
6. Gradient-penalty coefficient \(\lambda\), target Lipschitz constant \(L\), and exact GP tensor reduction (\(\|\nabla_z \mu_\theta\|_2\)).
7. Number of spectral-normalisation power iterations; which layers count in \(n\) for \(L^{1/n}\).
8. DP-SGD noise multiplier, clipping norm, \((\varepsilon,\delta)\), lot size, and epoch budget used for Figure 2.
9. Number of synthetic samples for MIA and for the published grids.
10. MIA distance (“suitable norm”), \(k\) if more than one neighbour, and whether the full Hilprecht MC-\(\varepsilon\) estimator was used.
11. Whether generated images are sigmoid probabilities, thresholded bits, or Bernoulli samples.
12. Whether the all-white outlier is included in the 1,000 MIA member images.
13. Location of any official training code (none found).

Do not invent answers to these and present them as paper specifications.
