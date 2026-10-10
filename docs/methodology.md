# VAE Paper Methodology

Primary source: `papers/DP_VAE.pdf`

Paper: Benedikt Groß and Gerhard Wunder, *Differentially Private Synthetic Data Generation via Lipschitz-Regularised Variational Autoencoders*, arXiv:2304.11336v2, 13 July 2023 (8 pages).

This document records the methodology required to reproduce the **reported VAE experiments**. Lipschitz regularisation is part of the paper, not a future extension. Quantization, mixed-type models (VAEM), and UCI Adult (explicitly unreported) are out of scope.

Information is classified as:

- **VERIFIED** — explicitly stated in the paper
- **INFERRED** — strongly implied by the paper or a cited source used as the paper describes
- **ASSUMED** — implementation choice required because the paper is silent
- **DEVIATION** — will differ from the paper (must be justified)

No training, evaluation scores, or paper tables have been reproduced yet. Nothing below is an experimental result of this repository.

---

# 1. Research Problem

Status: VERIFIED (Abstract; Section I, pp. 1–2; Section VI, p. 8)

## Objective

Generate synthetic data with a variational autoencoder while protecting training records, especially outliers, without using DP-SGD noise that degrades sample quality and exhausts the privacy budget as a function of the number of epochs.

## Contribution Relevant to VAE

1. Constrain the **decoder Lipschitz constant** of a VAE (L-VAE / LVAE) instead of adding a noise mechanism on the optimiser.
2. Provide a theoretical privacy argument (Section IV) using bounded log-likelihood, posterior sampling, and an information-theoretic view of the ELBO.
3. Empirically compare three models with the **same encoder/decoder architecture** (Section V):
   - vanilla VAE (no privacy mechanism)
   - VAE trained with a differentially private optimiser (DP-SGD) [8]
   - Lipschitz-regularised VAE (gradient-penalty variant reported in Figure 3)

## What This Reproduction Must Cover

The paper’s VAE *component* is the full Section III + Section V experimental pipeline: architecture, ELBO, Lipschitz constraint (GP primary; spectral normalisation described), generation, Monte Carlo membership inference, and the two baselines.

Out of scope for this phase:

- Quantization / compression
- UCI Adult (Section VI: “not reported here”)
- VAEM / mixed categorical–continuous extensions (Section VI: ongoing work)
- Shadow-model MIA and GAN-based MIA (described, not used)
- Re-deriving or numerically verifying the Section IV privacy bounds beyond what the experiments plot

---

# 2. Dataset

Status: VERIFIED (Section V opening, p. 6; Section V-A “MNIST”, p. 7)

## Dataset

| Item | Paper | Status | Reference |
|---|---|---|---|
| Name | MNIST handwritten digits | VERIFIED | Section V; [23] Deng 2012 |
| Input shape | \(28 \times 28 \times 1\) | VERIFIED | Section V-A |
| Source | Standard MNIST (train / test split of the public dataset) | INFERRED | [23]; paper never names an alternate host |
| Pixel range before binarisation | Not stated | ASSUMED uint8 \(\{0,\ldots,255\}\) as in standard MNIST | — |
| Official train size | Not restated | INFERRED 60,000 train / 10,000 test | MNIST documentation |
| Labels | Not used in training or reported metrics | INFERRED | Experiments are unconditional generation + MIA |

## Training set construction

| Item | Paper | Status | Reference |
|---|---|---|---|
| Size | 10,000 images | VERIFIED | Section V-A |
| Sampling | 9,999 images drawn **uniformly at random** from the MNIST **training** set | VERIFIED | Section V-A |
| Outlier | One all-white image (every pixel value 1) inserted into the training set | VERIFIED | Section V, p. 6; Section V-A |
| Validation split | None described | ASSUMED none; train on the 10k set | — |
| Held-out MNIST train remainder | Unused except that 1,000 **members** for MIA are taken from the (constructed) training set | VERIFIED | Section V-A |

## Preprocessing

| Item | Paper | Status | Reference |
|---|---|---|---|
| Binarisation | Pixels mapped to \(\{0,1\}\) | VERIFIED | Section V-A |
| Binarisation rule | Not stated (hard threshold vs Bernoulli sampling) | ASSUMED hard threshold after scaling to \([0,1]\): \(x \leftarrow \mathbf{1}[x \ge 0.5]\) | Common MNIST-VAE practice; must be documented as ASSUMED |
| Feature encoding | None beyond binarisation | VERIFIED | Image data |
| Rescaling for \(R_x\) | Theory says \(R_x\) can be controlled by rescaling (p. 5) | Not used in MNIST experiments | Theory only; experiments use \(\{0,1\}\) pixels |

## Privacy-evaluation split

| Item | Paper | Status | Reference |
|---|---|---|---|
| Members | 1,000 images from the **training** set | VERIFIED | Section V-A |
| Non-members | 1,000 images from the MNIST **test** set | VERIFIED | Section V-A |
| Attack set size | 2,000 images, each attacked with the Monte Carlo MIA | VERIFIED | Section V-A |
| Whether the all-white outlier is among the 1,000 members | Not stated | ASSUMED include it in the member pool (it is the highlighted privacy case) | Section V, p. 6 |

## Reference

- Dataset and protocol: Section V, pp. 6–7; Section V-A.
- Citation: [23] Li Deng, *The MNIST database of handwritten digit images for machine learning research*, IEEE Signal Processing Magazine, 2012.

---

# 3. VAE Architecture

Status: partially VERIFIED (Section III-A, pp. 3–4; Section V-A, p. 7)

Shared by **all three** variants (vanilla, DP-SGD, L-VAE).

## 3.1 Encoder \(q_\phi(z \mid x)\)

VERIFIED (Section III-A; Section V-A):

- Parameterises a Gaussian on latent space given \(x\).
- Four 2D convolutions, then two dense layers.
- Convolution filters: **32, 64, 128, 32**.
- Kernel: **\(3 \times 3\)**.
- Stride: **2** on all four convolutions.
- Dense widths: **100**, then **\(2 \cdot n_{\text{latent}}\)**.
- Hidden activations: **ReLU**.
- Output layer: **no activation**.

INFERRED:

- The \(2 \cdot n_{\text{latent}}\) vector is split into mean \(\mu_\phi(x)\) and a scale parameter (standard VAE: \(\log \sigma^2\) or \(\log \sigma\)).
- Diagonal covariance (paper: “parameterized Gaussian”; later text contrasts this with more expressive posteriors as unused literature).

ASSUMED (not in the paper):

| Choice | Proposed default | Why it is an assumption |
|---|---|---|
| Convolution padding | `same` | Kernel 3, stride 2, no padding specified. `same` yields spatial sizes \(28 \to 14 \to 7 \to 4 \to 2\) |
| Bias terms | Enabled | Unspecified |
| Batch norm / dropout | None | Unspecified; do not add unless later required |
| Encoder scale parameter | \(\log \sigma^2\) (Kingma & Welling [9]) | Paper only says “parameters for a Gaussian” |
| Flatten before first dense | Flatten \(2 \times 2 \times 32 = 128\) features after `same` padding | Follows from padding assumption |

## 3.2 Latent representation

VERIFIED:

- \(z \in \mathbb{R}^d\), \(d = n_{\text{latent}}\) (Section III-A; Section V-A).
- Prior \(p(z) = \mathcal{N}(0, I_d)\) (Section III-A; Section III-B after (11)).
- Approximate posterior \(q_\phi(z \mid x)\) is Gaussian (Section III-A).

ASSUMED:

- **Numeric \(n_{\text{latent}}\) is never given.** Proposed recorded default for implementation: **16**, stored in config, not claimed as a paper value. Sensitivity to 8 / 32 should be noted if sample quality fails.

## 3.3 Reparameterization

VERIFIED in words, not as a numbered equation: encoder outputs Gaussian parameters; samples \(z \sim q_\phi(z \mid x)\) are used in the reconstruction term of the ELBO (Section III-A). Paper cites Auto-Encoding Variational Bayes [9].

INFERRED from [9] and the \(2 \cdot n_{\text{latent}}\) head:

\[
z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I_d)
\]

with \(\sigma_\phi(x) = \exp(\tfrac{1}{2} \log \sigma^2_\phi(x))\).

## 3.4 Decoder \(\mu_\theta(z)\) / \(p_\theta(x \mid z)\)

VERIFIED (Section V-A):

- Input: latent sample \(z\).
- Two dense layers of widths **100** and **\(7 \cdot 7 \cdot 32\)**.
- Then transposed convolutions with **64, 128, and 1** filters, kernel **\(3 \times 3\)**, **stride 2**.
- Hidden activations: **ReLU**.
- Output layer: **no activation**.

### Spatial-size conflict (must not be silently “fixed”)

If the \(7 \times 7 \times 32\) tensor is followed by **three** stride-2 transposed convolutions with `same` padding, spatial size is \(7 \to 14 \to 28 \to 56\), which is **not** \(28 \times 28\).

The paper does not specify padding, output padding, or a crop.

**Proposed implementation (ASSUMED, not VERIFIED):**

- Dense 100 → ReLU.
- Dense \(7 \cdot 7 \cdot 32\) → ReLU → reshape `(N, 32, 7, 7)` (channels-first in PyTorch).
- ConvTranspose2d 32→64, \(3 \times 3\), stride 2, padding 1, output_padding 1 → \(14 \times 14\).
- ConvTranspose2d 64→128, \(3 \times 3\), stride 2, padding 1, output_padding 1 → \(28 \times 28\).
- Final layer: Conv2d 128→1, \(3 \times 3\), stride 1, padding 1 → \(28 \times 28\) logits (no activation).

This matches filter counts and the first two stride-2 upsamples, but **deviates** from a literal reading that the last layer also has stride 2. Alternative (literal stride 2 then centre-crop 56→28) is worse scientifically. Record as **ASSUMED + potential DEVIATION** until a faithful spatial recipe is confirmed.

## 3.5 Output / likelihood used in experiments

VERIFIED:

- MNIST is binary \(\{0,1\}\).
- Reconstruction loss is **binary cross-entropy** (Section V-A).
- Decoder output has **no** nonlinearity.

INFERRED:

- Use **BCE with logits** (PyTorch `BCEWithLogitsLoss`), which is the consistent pairing with an unactivated output.
- For **display** of samples, apply \(\sigma(\mu_\theta(z))\) or a 0.5 threshold; the paper’s sample grids look like probabilities or thresholded digits, not raw logits.

Theory-only (Section III-B, eq. (9)): Gaussian likelihood with unit covariance and MSE. **Do not use MSE for the MNIST experiments.** That Gaussian model is the setting of the privacy proofs, not the reported MNIST loss.

---

# 4. Mathematical Formulation

## Encoder

\[
q_\phi(z \mid x) = \mathcal{N}\big(z; \mu_\phi(x), \operatorname{diag}(\sigma_\phi^2(x))\big)
\]

Status: VERIFIED as “parameterized Gaussian”; diagonal form INFERRED.

Reference: Section III-A, pp. 3–4.

## Latent sampling (generation)

\[
z \sim p(z) = \mathcal{N}(0, I_d), \qquad x \sim p_\theta(x \mid z)
\]

In experiments, the decoder is treated as producing Bernoulli logits rather than sampling \(x\) from a Gaussian (INFERRED from BCE + binary data).

Reference: Section III-A; prior statement after (11).

## Decoder (theory vs experiment)

**Theory** (eq. (9), Section III-B, p. 4):

\[
p_\theta(x \mid z) = (2\pi)^{-n/2} \exp\Big(-\tfrac{1}{2} \|x - \mu_\theta(z)\|^2\Big)
\]

**Experiment** (Section V-A): Bernoulli / BCE. With logits \(o = \mu_\theta(z) \in \mathbb{R}^{28 \times 28}\):

\[
\log p_\theta(x \mid z) = -\sum_{j=1}^{784} \Big[ x_j \log \sigma(o_j) + (1-x_j)\log(1-\sigma(o_j)) \Big]
\]

implemented stably as BCE-with-logits.

Status: VERIFIED that experiments use BCE; INFERRED logits form.

## Reconstruction loss

Experimental objective term: expected binary cross-entropy between \(x\) and decoder output.

Status: VERIFIED (Section V-A). Reduction (mean vs sum over pixels vs batch) is **ASSUMED**: use **sum over pixels, mean over batch**, which matches the ELBO \(\mathbb{E}[\log p_\theta(x \mid z)]\) for independent Bernoullis.

## KL divergence

Unnumbered formula in Section III-A (implementation basis):

\[
\operatorname{KL}\big(q_\phi(z \mid x_i) \,\|\, p(z)\big)
\]

with \(p(z)=\mathcal{N}(0,I_d)\).

Closed form (INFERRED from [9], not written in this paper):

\[
\operatorname{KL} = -\frac{1}{2} \sum_{k=1}^{d} \Big( 1 + \log \sigma_k^2 - \mu_k^2 - \sigma_k^2 \Big)
\]

Lemma 1 / eq. (8) decomposes the average KL for the privacy discussion; it is **not** an extra training loss.

## Total objective

**Standard VAE (implementation basis, Section III-A):** maximise

\[
\frac{1}{N} \sum_{i=1}^{N} \Big( \mathbb{E}_{z_i \sim q_\phi(z \mid x_i)}[\log p_\theta(x_i \mid z_i)] - \operatorname{KL}(q_\phi(z_i \mid x_i) \,\|\, p(z)) \Big)
\]

Equivalently minimise \(\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{KL}}\).

\(\beta\)-VAE is cited [19] as literature; **the paper does not say experiments use \(\beta \neq 1\)**. Status: ASSUMED \(\beta = 1\).

**L-VAE constrained problem (eq. (13), Section III-B):**

\[
\min_{\phi,\theta} -\frac{1}{N}\sum_{i=1}^{N} \mathcal{L}(x_i;\phi,\theta)
\quad \text{subject to } \mu_\theta \text{ is } L\text{-Lipschitz}.
\]

**L-VAE, gradient-penalty realisation (Section V-A, citing WGAN-GP [24]):** add a penalty on \(\|\nabla_z \mu_\theta(z)\|\) to the ELBO. Exact penalty formula and coefficient are **not given**.

Proposed training loss (ASSUMED, from [24] adapted to decoder Lipschitz \(L\)):

\[
\mathcal{L}_{\text{LVAE-GP}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{KL}} + \lambda \,\mathbb{E}_{z}\big[\big(\|\nabla_z \mu_\theta(z)\|_2 - L\big)^2\big]
\]

Probe locations: finite set of \(z\) (paper). Proposed: the reparameterized \(z\) of the current batch (ASSUMED). \(\lambda\) and \(L\) are **unspecified**.

**L-VAE, spectral-normalisation realisation (Section V-A, citing [25]):**

- Spectral-normalise each decoder layer (pre-activation) so each linear map is 1-Lipschitz.
- ReLU / sigmoid are 1-Lipschitz.
- Multiply each layer output by \(L^{1/n}\) where \(n\) is the number of decoder layers, so the composition is \(L\)-Lipschitz.

Reported MNIST figure: **gradient penalty worked better; Figure 3 is GP, not SN** (Section V-A). SN remains a paper method to implement, not the reported main plot.

## Lipschitz definition

Eq. (10), Assumption 1:

\[
\|\mu_\theta(z_1) - \mu_\theta(z_2)\| \le L \|z_1 - z_2\| \quad \forall z_1,z_2
\]

Norms on \(\mathcal{X}\) and \(\mathcal{Z}\) are not named (likely Euclidean). Status: VERIFIED inequality; ASSUMED Euclidean norms.

---

# 5. Training Procedure

Section V-A states the software stack and architecture. **It does not give optimiser, learning rate, batch size, epoch count, seed, or initialisation.**

| Item | Paper | Status | Proposed recorded default |
|---|---|---|---|
| Framework | Python, TensorFlow **2.6** | VERIFIED | **DEVIATION:** PyTorch (project standard). Same math, not same framework |
| Privacy / MIA library | scikit-learn for Monte Carlo MIA | VERIFIED | scikit-learn / NumPy |
| Optimizer | Unspecified | ASSUMED | Adam [Kingma & Welling convention, not stated here] |
| Learning rate | Unspecified | ASSUMED | \(10^{-3}\) |
| Batch size | Unspecified | ASSUMED | 64 |
| Epochs | Unspecified; paper argues L-VAE is **not** epoch-budgeted like DP-SGD | ASSUMED | Config value (e.g. 50); do not invent a paper number |
| Scheduler | Unspecified | ASSUMED | None |
| Weight init | Unspecified | ASSUMED | Framework default |
| Regularisation besides Lipschitz | Unspecified | ASSUMED | None (no weight decay) |
| Early stopping | Unspecified | ASSUMED | None |
| Random seed | Unspecified | ASSUMED | Record a seed (e.g. 42) for reproducibility |
| Device | Unspecified | ASSUMED | Auto-detect CUDA / MPS / CPU (project rule) |
| DP-SGD hyperparameters (\(\sigma\), clip \(C\), \(\varepsilon, \delta\)) | “differentially private optimiser as in [8]”; Figure 2 shows a **loose theoretical guarantee in green** | ASSUMED at implementation time from [8] / TF-Privacy-style defaults; **must not be labelled VERIFIED** | To be chosen and documented in experiment config |
| Gradient-penalty \(\lambda\), Lipschitz \(L\) | Unspecified | ASSUMED | Config hyperparameters; sweep if needed |
| SN power-iteration steps | “a few steps” | INFERRED from [25] | Typically 1 per forward (Miyato et al.) |

Vanilla VAE and L-VAE may be trained for many epochs; DP-SGD cannot, because privacy spend grows with steps (Section I, Section VI). The paper does not quantify either budget.

---

# 6. Generation Procedure

VERIFIED (Section III-A; sample grids in Figures 1–3):

1. Draw \(z \sim \mathcal{N}(0, I_d)\).
2. Pass \(z\) through the decoder.
3. Display synthetic images (and, for MIA, use a synthetic set as the generator’s public output).

Not specified:

- Number of images in the published grids
- Number of synthetic records used for MIA
- Whether generated pixels are Bernoulli-sampled or taken as \(\sigma(\text{logits})\)

ASSUMED:

- For **plots**: \(\sigma(\mu_\theta(z))\), optionally thresholded at 0.5.
- For **MIA**: a synthetic pool of size **10,000** (matched to training \(N\)), using probabilities or thresholded binaries consistently; record the choice.
- No class conditioning (architecture is unconditional).

Outlier check (VERIFIED, Section V-B): reconstruct the all-white training image; vanilla VAE reconstructs it (yellow square in Figure 1); both private variants do not.

---

# 7. Evaluation

## Metrics

| Metric | What the paper reports | Status | Reference |
|---|---|---|---|
| Monte Carlo MIA trade-off | Type I vs type II error while sweeping the attack threshold | VERIFIED | Section II-C; Section V-A; Figures 1–3 |
| Sample quality | Qualitative image grids | VERIFIED | Figures 1–3; Section V-B |
| Outlier reconstruction | Visual: white image reconstructed or not | VERIFIED | Section V-B |
| Theoretical DP curve | Green curve on the DP-SGD plot (“loose theoretical guarantee”) | VERIFIED as present in Fig. 2 | Section V-B |
| FID / IS / KNN accuracy / numeric tables | **None** | — | No tables in the paper |

Do **not** invent numeric paper scores. Compare OUR plots to PAPER figures qualitatively and keep them labelled separately.

## Baselines

VERIFIED (Section I; Section V opening):

1. **Vanilla VAE** — same architecture, no Lipschitz constraint, no DP optimiser (Figure 1).
2. **VAE-DPSGD** — same architecture, DP optimiser from Abadi et al. [8] (Figure 2).
3. **LVAE** — same architecture, Lipschitz decoder via **gradient penalty** for the reported figure (Figure 3). Spectral normalisation is a second L-VAE variant, empirically weaker on MNIST, not the published main curve.

## Evaluation procedure (Monte Carlo MIA)

Paper’s own wording (Section II-C, item 3):

> classification … based on **thresholding the distance of a target record to its nearest synthetic records** in a suitable norm.

Attack setting: **black-box** (parameters not available to the attacker). Shadow-model and GAN attacks are described and **not** used.

Protocol that is VERIFIED:

1. Train the target generative model on the 10k set (incl. white outlier).
2. Draw 1,000 members from training and 1,000 non-members from MNIST test.
3. For each of 2,000 images, run the Monte Carlo attack.
4. Plot type-I vs type-II error over **varying thresholds**.

Unspecified in this paper (partially specified in cited [17] Hilprecht et al.):

- Distance / “suitable norm” (Euclidean on pixels vs HOG vs other)
- Number of synthetic samples \(n\)
- Whether score is distance-to-nearest, MC-\(\varepsilon\) counts, or MC-\(d\) density
- How \(\varepsilon\) is chosen if the full MC-\(\varepsilon\) estimator is used

**Proposed reproduction (ASSUMED, aligned with the paper’s one-sentence description rather than the full [17] feature zoo):**

- Flatten images to vectors in \(\{0,1\}^{784}\) or \([0,1]^{784}\).
- Score(target) = Euclidean distance to the **nearest** synthetic sample (or mean of \(k=1\) nearest).
- Predict member if score \(< \tau\); sweep \(\tau\) to get the trade-off curve.
- Record that richer [17] features (HOG, colour histogram) are **not** claimed by this paper.

Implementation library: scikit-learn (VERIFIED).

---

# 8. Paper Reproduction Checklist

| Component | Paper | Implementation (planned) | Status |
|---|---|---|---|
| Dataset | MNIST \(28\times28\times1\) | `torchvision` MNIST | VERIFIED |
| Train subset | 9,999 random train + 1 white outlier = 10k | Same protocol, seed recorded | VERIFIED protocol / ASSUMED seed |
| Preprocessing | Binarise to \(\{0,1\}\) | Threshold at 0.5 after /255 | VERIFIED goal / ASSUMED rule |
| Validation split | None | None for paper runs | VERIFIED |
| Encoder | 4 conv (32,64,128,32), \(3\times3\), stride 2; dense 100 and \(2 n_{\text{lat}}\) | PyTorch modules | VERIFIED widths / ASSUMED padding |
| Latent dimension | \(n_{\text{latent}}\) unnamed | Config default 16 | ASSUMED numeric value |
| Posterior | Gaussian | Diagonal Gaussian, \(\log \sigma^2\) | VERIFIED family / ASSUMED parameterisation |
| Prior | \(\mathcal{N}(0,I_d)\) | Same | VERIFIED |
| Reparameterization | Implied via [9] | \(z=\mu+\sigma\odot\varepsilon\) | INFERRED |
| Decoder | Dense 100, \(7\cdot7\cdot32\); TConv 64,128,1; \(3\times3\) stride 2 | Last spatial layer stride 1 to keep \(28\times28\) | VERIFIED filters / ASSUMED last stride |
| Activations | ReLU; no output activation | Same | VERIFIED |
| Reconstruction (experiments) | Binary cross-entropy | `BCEWithLogitsLoss` | VERIFIED loss / INFERRED logits |
| Reconstruction (theory) | Gaussian / MSE eq. (9) | Not used for MNIST training | VERIFIED theory-only |
| KL | \(\mathrm{KL}(q_\phi(z\|x)\|p(z))\) | Closed-form diagonal Gaussian | VERIFIED term / INFERRED closed form |
| Total loss (vanilla) | Negative ELBO, \(\beta\) unspecified | \(\mathcal{L}_{\text{recon}}+\mathcal{L}_{\text{KL}}\) | VERIFIED structure / ASSUMED \(\beta=1\) |
| L-VAE constraint | Eq. (13), Lipschitz decoder | GP primary; SN optional | VERIFIED |
| GP formula / \(\lambda\) / \(L\) | “as in WGAN-GP [24]” | Penalty on \(\|\nabla_z \mu_\theta\|_2\) | INFERRED form / ASSUMED \(\lambda,L\) |
| Spectral normalisation | Each decoder layer + \(L^{1/n}\) | Implement as ablation | VERIFIED method |
| Optimizer | Unspecified | Adam (planned) | ASSUMED |
| Learning rate | Unspecified | \(10^{-3}\) (planned) | ASSUMED |
| Batch size | Unspecified | 64 (planned) | ASSUMED |
| Epochs | Unspecified | Config | ASSUMED |
| Scheduler | Unspecified | None | ASSUMED |
| Framework | TensorFlow 2.6 | PyTorch | DEVIATION (stack only) |
| Generation | \(z\sim p(z)\), decode | Same | VERIFIED |
| MIA | Monte Carlo, 1k+1k, threshold sweep | Nearest-synthetic Euclidean | VERIFIED protocol / ASSUMED metric |
| Baselines | Vanilla VAE, VAE-DPSGD | Same three runs | VERIFIED |
| DP-SGD hparams | Unspecified beyond [8] | Config + accountant curve | ASSUMED |
| Reported results | Figs 1–3 only | Reproduce plots; no numeric paper table | VERIFIED |

---

# 9. Deviations

| Component | Paper | Our implementation | Reason |
|---|---|---|---|
| Deep learning framework | TensorFlow 2.6 | PyTorch | Project environment; methodology unchanged |
| Decoder last upsample | Three stride-2 transposed convs as written | Two stride-2 TConvs then stride-1 conv to \(28\times28\) | Literal three stride-2 layers yield \(56\times56\); paper gives no padding/crop |
| (none yet for results) | — | — | No experiments run |

Framework and decoder-geometry choices must remain labelled DEVIATION / ASSUMED. They are not paper results.

---

# 10. Reproduction Notes

## Paper structure (read in full)

| Pages | Content |
|---|---|
| 1 | Abstract; Index terms; start of Introduction |
| 2 | Introduction continued; Section II DP Definitions 1–2 |
| 3 | Hypothesis-test DP, \(f\)-DP, GDP; Def. 5 generative DP; MIA taxonomy; start of VAEs |
| 4 | ELBO forms; Lemma 1; L-VAE; eq. (9)–(11) |
| 5 | \(R_x\); Assumptions 1–2; eq. (13); Section IV Lemmas 2–3 |
| 6 | Lemma 4; main generative DP argument; start of Experiments |
| 7 | Implementation details; MNIST protocol; Figures 1–3; Results |
| 8 | Section VI conclusion; acknowledgements; references [1]–[26] |

No appendix. No hyperparameter table. No official code found (checked arXiv:2304.11336; third-party “Lipschitz VAE” repos implement a **different** paper).

## Theory vs MNIST experiment (do not mix)

| | Theory (III-B, IV) | MNIST experiments (V) |
|---|---|---|
| Likelihood | Gaussian, MSE, eq. (9) | Bernoulli, BCE |
| Lipschitz | Assumption 1 on \(\mu_\theta\) | GP (reported) or SN |
| Privacy | Lemmas 2–4, \(\varepsilon=2C\) for encoder under consistency | Empirical MIA trade-off only |

## Experiments that must be reproduced

1. **Vanilla VAE** — train; generate sample grid; reconstruct white outlier; MIA trade-off (Figure 1).
2. **VAE-DPSGD** — same evaluation; optional theoretical trade-off overlay in green (Figure 2).
3. **LVAE-GP** — same evaluation (Figure 3).
4. **LVAE-SN** — implement and optionally run; paper says GP was better; do not treat SN as Figure 3.

Qualitative claims to check (PAPER RESULT, Section V-B), not to fabricate as OUR numbers:

- DP-SGD is strongest against MIA but has much worse samples.
- Vanilla VAE reconstructs the white outlier; both private methods do not.
- LVAE offers a better privacy–utility trade-off than DP-SGD on this setup.

## Open implementation questions

See `docs/resources.md` and `docs/implementation_plan.md`. The blocking unspecified items are: \(n_{\text{latent}}\), optimiser/LR/batch/epochs, GP \((\lambda, L)\), DP-SGD noise/clip/\((\varepsilon,\delta)\), decoder padding, binarisation rule, MIA distance and synthetic-set size.

**No code has been implemented yet. No tests have been executed.**
