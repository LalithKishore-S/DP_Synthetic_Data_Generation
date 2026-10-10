# Project Instructions — DP VAE Paper Reproduction

## 1. Current Project Scope

The current goal of this repository is to faithfully implement
and reproduce the VAE component of the research paper:

papers/DP_VAE.pdf

The current phase is ONLY the papers/DP_VAE.pdf paper implementation.

Do NOT implement or investigate:

- Quantization
- Model compression
- Post-training quantization
- Additional architectures
- Unrelated VAE variants
- Future project extensions

These will be considered only after the VAE implementation
has been completed, tested, evaluated, and frozen.

---

# 2. Primary Objective

The objective is not merely to create a VAE that runs.

The objective is to implement the methodology described in
the paper as faithfully as possible.

The implementation must cover the complete methodology
described in the paper, including where applicable:

1. Dataset acquisition/loading
2. Data preprocessing
3. Data splitting
4. VAE architecture
5. Encoder
6. Latent representation
7. Reparameterization
8. Decoder
9. Reconstruction loss
10. KL-divergence loss
11. Complete VAE objective
12. Training procedure
13. Hyperparameters
14. Optimizer
15. Learning-rate schedule
16. Checkpointing
17. Synthetic data generation
18. Evaluation
19. Baselines
20. Experiments reported in the paper

The exact list must be determined from the paper itself.

Do not assume that a standard VAE implementation is equivalent
to the paper.

---

# 3. Research Integrity

The original paper is the primary source of truth.

Before implementing a research-specific component:

1. Locate the relevant section/equation/algorithm in the paper.
2. Understand the mathematical formulation.
3. Record the interpretation in docs/methodology.md.
4. Implement the verified methodology.

Never invent:

- Architecture dimensions
- Loss functions
- Hyperparameters
- Dataset preprocessing
- Evaluation metrics
- Experimental results

If a detail is not specified in the paper:

1. Identify it as unspecified.
2. Determine whether it can be inferred from other parts
   of the paper.
3. If an assumption is required, document it explicitly.
4. Never present an assumption as if it were stated by the paper.

---

# 4. Read the Entire Paper

Before implementing the complete model, inspect the entire
papers/DP_VAE.pdf.

Do not rely only on:

- Abstract
- Introduction
- Method section
- Search snippets
- Secondary summaries

Read the complete paper including:

- Methodology
- Mathematical formulation
- Experimental setup
- Tables
- Figures
- Results
- Discussion
- Conclusion
- Appendices, if present

Use the paper to determine exactly what needs to be implemented.

---

# 5. Master Notebook

The complete experiment must be reproducible from:

src/notebooks/vae_paper_implementation.ipynb

This notebook is the master experiment notebook.

It should contain the complete end-to-end workflow:

1. Environment setup
2. Imports
3. Reproducibility configuration
4. Dataset loading
5. Dataset inspection
6. Preprocessing
7. Train/validation/test split
8. Model configuration
9. Model creation
10. Training
11. Checkpoint saving
12. Model loading
13. Synthetic data generation
14. Evaluation
15. Visualization
16. Results reporting

The notebook should execute from top to bottom in a clean
environment without requiring undocumented manual steps.

---

# 6. Source Code Organization

Reusable code should live under src/.

Use appropriate modules for:

src/data/
    Dataset loading and preprocessing.

src/models/
    Encoder, decoder, VAE and related model components.

src/scripts/
    Optional helper scripts for training, evaluation,
    preprocessing, or environment setup.

The notebook should orchestrate these components rather than
duplicating large amounts of implementation code.

However, if the paper's implementation is clearer when a
small mathematical component is shown directly in the
notebook, that is acceptable.

---

# 7. Results Directory

All generated experimental artifacts must be saved under:

results/

Use:

results/models/
    Model checkpoints and saved model weights.

results/metrics/
    JSON/CSV files containing evaluation metrics.

results/plots/
    Figures and plots generated during evaluation.

results/samples/
    Generated synthetic samples and representative outputs.

results/logs/
    Training logs and experiment metadata.

Do not scatter experiment outputs throughout the repository.

Do not save checkpoints directly into src/.

---

# 8. Naming Experimental Outputs

Use descriptive filenames.

Examples:

results/models/vae_best.pt
results/models/vae_final.pt

results/metrics/vae_metrics.json
results/metrics/training_history.csv

results/plots/training_loss.png
results/plots/reconstruction_quality.png

results/samples/generated_samples.csv

If multiple experiments are performed, create experiment-specific
subdirectories rather than overwriting previous results.

For example:

results/
    experiment_01/
        models/
        metrics/
        plots/
        samples/
        logs/

Never silently overwrite an existing experimental result.

---

# 9. Reproducibility

Set and record random seeds wherever possible.

Record:

- Python version
- PyTorch version
- NumPy version
- Dataset version/source
- Random seed
- Model configuration
- Batch size
- Learning rate
- Optimizer
- Number of epochs
- Hardware
- Training time
- Relevant package versions

Save the configuration used for each experiment.

---

# 10. Local Development and Colab

The project is developed locally using VS Code/Cursor.

The MacBook environment is primarily for:

- Code development
- Unit tests
- Debugging
- Small smoke tests
- Small experiments

Full training may be performed in Google Colab.

Do not assume CUDA is available locally.

The code must detect the available device rather than
hardcoding CUDA.

For example, the implementation may select between
CUDA, MPS, and CPU where appropriate.

The full experiment notebook must be capable of running
in Google Colab.

---

# 11. Validation Requirements

Before considering the implementation complete:

1. Verify dataset loading.
2. Verify preprocessing.
3. Verify encoder dimensions.
4. Verify decoder dimensions.
5. Verify latent sampling.
6. Verify VAE forward pass.
7. Verify reconstruction loss.
8. Verify KL divergence.
9. Verify total loss.
10. Verify gradient flow.
11. Perform a small training smoke test.
12. Verify the model actually learns.
13. Save and reload a checkpoint.
14. Generate synthetic samples.
15. Run the evaluation procedure.
16. Compare implementation details with the paper.

Tests must actually be executed.

Never claim that a test passed without running it.

---

# 12. Paper Reproduction Versus Assumptions

Every deviation from the paper must be documented.

Maintain a table in docs/methodology.md:

| Component | Paper | Implementation | Status |
|---|---|---|---|
| Dataset | | | |
| Preprocessing | | | |
| Encoder | | | |
| Decoder | | | |
| Latent dimension | | | |
| Loss | | | |
| Optimizer | | | |
| Learning rate | | | |
| Batch size | | | |
| Epochs | | | |

Use statuses such as:

- VERIFIED
- INFERRED
- ASSUMED
- DEVIATION

---

# 13. Code Quality

Use:

- Python
- PyTorch
- Modular code
- Type hints where useful
- Clear function/class names
- Docstrings for non-obvious functionality
- Configuration files for experiment parameters

Avoid:

- Hardcoded absolute paths
- Duplicated implementation
- Unnecessary abstractions
- Silent assumptions
- Untracked experimental changes

---

# 14. Important Agent Behaviour

Before making a major implementation decision:

Explain:

1. What the paper specifies.
2. What the implementation will do.
3. Why the implementation matches the paper.

When information is ambiguous, do not silently decide.

When an experiment fails:

- Report the failure.
- Explain the likely cause.
- Propose a fix.
- Do not fabricate successful results.

Do not modify the research methodology merely to make
the implementation easier.

---

# 15. Completion Criterion

The VAE phase is considered complete only when:

- The complete paper methodology has been inspected.
- The VAE has been implemented.
- The implementation has been tested.
- The master notebook runs end-to-end.
- The model can be trained.
- Checkpoints can be saved and loaded.
- Synthetic data can be generated.
- Evaluation has been performed.
- Results have been saved under results/.
- Deviations from the paper have been documented.
- The final configuration has been saved.
- The VAE baseline is reproducible.

Only after these conditions are satisfied should the project
move to a future extension.