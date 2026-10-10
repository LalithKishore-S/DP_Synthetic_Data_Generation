# Implementation Plan — Paper VAE Reproduction

Status: **analysis and plan only**. Full implementation has **not** started.

This plan turns `docs/methodology.md` into a reproducible PyTorch codebase and a single master notebook. It covers the **entire VAE methodology of** `papers/DP_VAE.pdf` (vanilla VAE, L-VAE, DP-SGD baseline, generation, Monte Carlo MIA). It does **not** cover quantization, VAEM, or UCI Adult.

Every unspecified paper detail must be read from a config file and recorded in experiment metadata as **ASSUMED**, never as a paper hyperparameter.

---

# 1. Scope of the first implementation phase

## In scope

- MNIST load, binarise, 9,999 + white outlier protocol
- Convolutional encoder / decoder as specified
- Standard ELBO (BCE reconstruction + KL)
- Reparameterization, prior \(\mathcal{N}(0,I)\)
- Lipschitz decoder via gradient penalty (reported L-VAE)
- Lipschitz decoder via spectral normalisation (paper method; not Figure 3)
- DP-SGD VAE baseline (same architecture)
- Ancestral sampling for synthetic images
- Outlier reconstruction check
- Monte Carlo MIA trade-off curves
- Checkpoints, logs, plots, metrics under `results/`
- Unit / smoke tests listed in Section 8
- Master notebook `src/notebooks/vae_paper_implementation.ipynb`

## Out of scope

- Quantization, compression, PTQ
- Mixed-type / VAEM architectures
- UCI Adult
- Shadow-model MIA, LOGAN/GAN MIA
- Re-implementing the full Section IV proof as executable math
- Claiming numerical equality with Figures 1–3 (figures are qualitative)

---

# 2. Proposed source-code structure

```text
DP_Synthetic_data_generation/
  configs/
    default.yaml              # ASSUMED hparams + VERIFIED architecture
    experiment_vanilla.yaml
    experiment_lvae_gp.yaml
    experiment_lvae_sn.yaml
    experiment_dpsgd.yaml
  src/
    __init__.py
    data/
      __init__.py
      mnist.py                # download, subset, outlier, binarise, loaders
      splits.py               # member / non-member MIA indices
    models/
      __init__.py
      encoder.py              # conv encoder → μ, log σ²
      decoder.py              # dense + conv-transpose decoder
      vae.py                  # vanilla VAE forward + reparameterize
      losses.py               # BCE recon, KL, ELBO, gradient penalty
      lipschitz.py            # spectral norm wrappers, L^{1/n} scale
    training/
      __init__.py
      device.py               # CUDA / MPS / CPU
      trainer.py              # loop, logging, no overwrite of results
      dpsgd.py                # DP optimiser wrapper (Opacus or equivalent)
    evaluation/
      __init__.py
      generate.py             # z ~ N(0,I), decode, postprocess
      mia.py                  # Monte Carlo nearest-synthetic attack
      plots.py                # sample grids, trade-off curves, reconstructions
    scripts/
      train.py                # CLI: config → train one variant
      evaluate.py             # CLI: checkpoint → samples + MIA + plots
      smoke_test.py           # tiny run for local Mac
    notebooks/
      vae_paper_implementation.ipynb
    tests/                    # or tests/ at repo root; prefer repo-root tests/
  tests/
    test_forward.py
    test_losses.py
    test_backward.py
    test_checkpoint.py
    test_generation.py
    test_data.py
  results/
    models/
    metrics/
    plots/
    samples/
    logs/
  docs/
    methodology.md
    implementation_plan.md
    resources.md
```

Notes:

- The notebook **imports** these modules; it does not reimplement the model.
- `src/training/` and `src/evaluation/` are allowed even though AGENTS.md names `data/`, `models/`, `scripts/`. Keeping train/eval out of `models/` avoids mixing nn.Module code with loops. Scripts remain thin CLIs.
- No generated artifacts under `src/`.
- `Requirements.txt` must list: `torch`, `torchvision`, `numpy`, `scikit-learn`, `matplotlib`, `pyyaml`, `opacus` (or documented DP-SGD alternative). Versions recorded at run time.

---

# 3. Configuration (machine-readable)

Each run writes a copy of the resolved config plus environment info to `results/<experiment_id>/logs/config.json`.

Proposed `configs/default.yaml` keys (paper vs assumption):

| Key | Source |
|---|---|
| `dataset.name: mnist` | VERIFIED |
| `dataset.n_train: 9999` | VERIFIED |
| `dataset.add_white_outlier: true` | VERIFIED |
| `dataset.binarize: true` | VERIFIED |
| `dataset.binarize_threshold: 0.5` | ASSUMED |
| `model.conv_filters: [32, 64, 128, 32]` | VERIFIED |
| `model.kernel_size: 3` | VERIFIED |
| `model.stride: 2` | VERIFIED |
| `model.encoder_hidden: 100` | VERIFIED |
| `model.decoder_hidden: 100` | VERIFIED |
| `model.decoder_reshape: [32, 7, 7]` | VERIFIED |
| `model.tconv_filters: [64, 128, 1]` | VERIFIED |
| `model.latent_dim: 16` | ASSUMED |
| `model.output_activation: none` | VERIFIED |
| `loss.reconstruction: bce_with_logits` | VERIFIED + INFERRED |
| `loss.beta_kl: 1.0` | ASSUMED |
| `train.optimizer: adam` | ASSUMED |
| `train.lr: 0.001` | ASSUMED |
| `train.batch_size: 64` | ASSUMED |
| `train.epochs: 50` | ASSUMED |
| `train.seed: 42` | ASSUMED |
| `lvae.method: gradient_penalty \| spectral_norm \| none` | VERIFIED options |
| `lvae.lipschitz_L` | ASSUMED |
| `lvae.gp_lambda` | ASSUMED |
| `dpsgd.*` | ASSUMED from [8] |
| `eval.n_synthetic: 10000` | ASSUMED |
| `eval.n_members: 1000` | VERIFIED |
| `eval.n_nonmembers: 1000` | VERIFIED |
| `eval.mia_metric: euclidean_nearest` | ASSUMED |

`experiment_id` must include timestamp or incrementing id so existing result directories are never overwritten.

---

# 4. Module responsibilities (mapped to the paper)

## 4.1 `src/data/mnist.py`

Paper: Section V-A.

1. Load MNIST train/test via `torchvision` (portable; works on Mac and Colab).
2. Scale to \([0,1]\), binarise with the configured threshold.
3. Sample 9,999 training indices uniformly without replacement (seeded).
4. Append a \(28\times28\) tensor of ones (all-white outlier).
5. Return `DataLoader`s; persist the chosen indices under `results/.../logs/` so MIA members are reproducible.

Do not download into `src/`. Use a project `data/` cache directory ignored by git if needed.

## 4.2 `src/models/encoder.py` / `decoder.py` / `vae.py`

Paper: Section III-A; Section V-A.

- Encoder: 4× Conv2d → flatten → Linear(100) → ReLU → Linear(\(2d\)) split into \(\mu, \log\sigma^2\).
- Reparameterize as in [9].
- Decoder: Linear(100) → ReLU → Linear(\(7\cdot7\cdot32\)) → ReLU → reshape → two stride-2 ConvTranspose2d → stride-1 Conv2d to 1 channel, **no** output activation.
- Document the last-layer stride assumption next to the module.

Forward returns `x_logits, mu, logvar, z`.

## 4.3 `src/models/losses.py`

Paper: unnumbered ELBO in III-A; BCE in V-A; GP in V-A / [24].

- `reconstruction_bce(logits, x)` — sum over pixels, mean over batch.
- `kl_standard_normal(mu, logvar)` — closed form.
- `elbo_loss = recon + beta * kl`.
- `gradient_penalty(decoder, z, L)` — `autograd.grad` of decoder output w.r.t. `z`; penalise \((\|\nabla_z \mu_\theta\|_2 - L)^2\). Use a scalar reduction of the image-valued decoder (e.g. sum of outputs) so the Jacobian–vector is well-defined, **or** compute the Frobenius norm of \(\partial \mathrm{vec}(\mu)/\partial z\). The paper does not specify the tensor reduction; the choice must be documented as ASSUMED.

## 4.4 `src/models/lipschitz.py`

Paper: Section V-A, spectral normalisation [25].

- Wrap every decoder linear/conv weight with spectral normalisation.
- After each layer (pre- or post-activation consistently), scale by \(L^{1/n}\).
- Count \(n\) as the number of weight layers in the decoder.

## 4.5 Training

- `device.py`: CUDA → MPS → CPU.
- `trainer.py`: seed; epoch loop; log recon/kl/gp; save `vae_best.pt` / `vae_final.pt` under the experiment folder; write `training_history.csv`.
- `dpsgd.py`: Opacus `PrivacyEngine` (or equivalent) around the **same** VAE. Record \(\varepsilon, \delta\), noise multiplier, max grad norm. Overlay the theoretical trade-off \(f_{\varepsilon,\delta}\) on the MIA plot when those values exist (Figure 2 green curve). If Opacus cannot attach to this architecture, document the blocker; do not fake a DP run.

## 4.6 Generation and evaluation

- `generate.py`: \(z \sim \mathcal{N}(0,I)\); decoder; sigmoid for continuous samples; optional Bernoulli/threshold for binary samples; save `generated_samples.npy` / grid PNG.
- Reconstruct the white outlier and a small real batch; save side-by-side PNG.
- `mia.py`: 1,000 members + 1,000 non-members; nearest-synthetic Euclidean scores; sweep thresholds; save `mia_tradeoff.csv` and plot type I vs type II (axes comparable to Figures 1–3).
- Do not write PAPER RESULT numbers into metric JSON. Write OUR RESULT only.

---

# 5. Proposed notebook structure

File: `src/notebooks/vae_paper_implementation.ipynb`

The notebook must run top to bottom on Colab and locally. Heavy training cells should honour a `SMOKE` / `FULL` flag so a Mac can execute a tiny path without claiming a paper reproduction.

| Section | Contents |
|---|---|
| 0. Title and paper citation | arXiv:2304.11336v2; scope reminder (no quantization) |
| 1. Environment setup | Install from `Requirements.txt` if needed; print Python / torch / numpy / sklearn versions; device detection |
| 2. Reproducibility | Seed; `experiment_id`; output roots under `results/<experiment_id>/{models,metrics,plots,samples,logs}` |
| 3. Configuration | Load YAML; display VERIFIED vs ASSUMED table |
| 4. Dataset | Load MNIST; show raw vs binarised examples; construct 10k set; show the white outlier |
| 5. Splits | Persist train indices; draw 1,000 members / 1,000 non-members |
| 6. Model | Instantiate encoder/decoder/VAE; print parameter counts; **forward-pass check** on one batch |
| 7. Losses | Compute recon, KL, total on one batch; **no training yet in this cell** |
| 8. Training | Call `trainer` for the selected variant (`vanilla` / `lvae_gp` / `lvae_sn` / `dpsgd`) |
| 9. Checkpoint | Save `vae_final.pt`; reload; assert parameter equality |
| 10. Generation | Sample grid; save to `results/.../samples/` and `plots/` |
| 11. Outlier reconstruction | Decode encoder(white); compare vanilla vs private when multiple ckpts exist |
| 12. Evaluation (MIA) | Trade-off curve; save metrics JSON/CSV |
| 13. Results reporting | OUR RESULT vs PAPER RESULT (qualitative: Figures 1–3). Never paste fabricated scores |
| 14. Experiment inventory | Which of the three reported runs have been executed in this session |

Optional later cells (still in the same notebook, gated): train remaining variants for the three-way comparison.

Do not create additional notebooks unless requested.

---

# 6. Output management

All artifacts:

```text
results/<experiment_id>/
  models/vae_best.pt
  models/vae_final.pt
  metrics/vae_metrics.json
  metrics/training_history.csv
  metrics/mia_tradeoff.csv
  plots/training_loss.png
  plots/generated_samples.png
  plots/outlier_reconstruction.png
  plots/mia_tradeoff.png
  samples/generated_samples.npy
  logs/config.json
  logs/environment.json
  logs/train.log
```

Rules:

- `experiment_id` example: `20261010_vanilla_mnist` or `exp_001_vanilla`.
- Refuse to write into an existing `results/<experiment_id>/` unless a new id is minted.
- Never save checkpoints inside `src/`.

Suggested ids for the paper trio:

- `exp_vanilla_mnist`
- `exp_dpsgd_mnist`
- `exp_lvae_gp_mnist`
- `exp_lvae_sn_mnist` (ablation)

---

# 7. Experiments to reproduce

| ID | Model | Paper figure | Required outputs |
|---|---|---|---|
| E1 | Vanilla VAE | Fig. 1 | Samples, MIA curve, white-outlier reconstruction (expected: visible) |
| E2 | VAE-DPSGD | Fig. 2 | Samples, MIA curve, green theoretical \(f_{\varepsilon,\delta}\) if hparams known, outlier recon (expected: not reconstructed) |
| E3 | LVAE-GP | Fig. 3 | Samples, MIA curve, outlier recon (expected: not reconstructed) |
| E4 | LVAE-SN | Text only (V-A) | Optional; paper reports GP worked better |

Success criteria (research integrity):

- Pipeline runs; losses finite; samples look digit-like for vanilla/LVAE.
- Qualitative agreement with Section V-B **or** a documented disagreement (DEVIATION / failed reproduction).
- No invented FID or accuracy matching the paper (the paper has none).

Local Mac: E1 smoke (few epochs, tiny subset) only. Full E1–E3 may run on Colab.

---

# 8. Validation plan (must be executed before claiming pass)

| Test | What it checks | When |
|---|---|---|
| Dataset load | Shapes `(N,1,28,28)`, values in `{0,1}`, length 10_000 including a unique all-ones image | After data module |
| Encoder / decoder dims | Latent `(B,d)`; recon logits `(B,1,28,28)` | After model |
| Reparameterization | `z` requires grad w.r.t. \(\mu,\log\sigma^2\) | After model |
| Forward | One batch, no exception | After model |
| Loss | Recon ≥ 0; KL ≥ 0 (up to numerical error); ELBO finite | After losses |
| Analytic KL | Compare closed form vs MC KL on a fixed Gaussian | After losses |
| Backward | All trainable params have non-None grad (vanilla); GP grads exist for decoder | After losses |
| Smoke train | 1–2 epochs on ~128 images; loss decreases or is logged honestly if it does not | After trainer |
| Checkpoint | Save → load → equal tensors; second forward matches | After trainer |
| Generation | `n` samples, finite, correct shape | After generate |
| MIA smoke | 16 members / 16 non-members produces a curve of the right length | After mia |

Tests live under `tests/` and are run with `pytest`. **Do not mark them passed until they have been run.**

This analysis phase executes **no** tests.

---

# 9. Implementation order (when implementation is authorised)

Do not start this list until the user asks to implement.

1. Project skeleton: packages, `configs/default.yaml`, `results/` subdirs, requirements.
2. Device + seed utilities.
3. Dataset pipeline + `test_data.py`.
4. Encoder, decoder, VAE + forward tests.
5. Losses + KL/recon unit tests + backward test.
6. Vanilla trainer + checkpoint + smoke train.
7. Generation + sample plots.
8. MIA evaluation + plots.
9. LVAE-GP penalty + SN decoder.
10. DP-SGD trainer.
11. Master notebook wiring E1, then E2–E3.
12. Full runs on Colab; write OUR RESULT vs PAPER RESULT in `docs/methodology.md` Section 10.

---

# 10. Risks

1. **Missing hyperparameters** dominate reproduction error more than architecture.
2. **Decoder spatial geometry** is internally inconsistent in the paper.
3. **BCE vs Gaussian theory** — training must follow Section V, not eq. (9).
4. **DP-SGD** may not match Figure 2 without unpublished noise/clip values.
5. **MIA** is under-specified; Euclidean nearest-neighbour is a documented proxy for the paper’s sentence, not a proof of identity with the authors’ scikit-learn script.
6. **No official code**; do not copy unrelated Lipschitz-VAE repositories.

---

# 11. Completion criterion for the VAE phase

From `AGENTS.md`, after implementation (not now):

- Methodology inspected (this document + `docs/methodology.md`)
- VAE implemented and tested with executed tests
- Master notebook runs end to end
- Checkpoints save/load
- Synthetic data generated
- Evaluation performed
- Results under `results/`
- Deviations documented
- Final config saved

Only then consider future extensions.
