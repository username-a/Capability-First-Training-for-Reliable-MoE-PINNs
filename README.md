# Numerical Qualification of Soft MoE-PINNs

Official reproducibility package for **“Numerical Qualification of Soft MoE-PINNs: Identifiability, Branch-Reliability Audits, and Routing Stress Tests.”**

This repository studies a distinction that aggregate solution error alone cannot resolve: a soft mixture can remain accurate while its complete expert branches become unreliable or while the prediction becomes sensitive to routing interventions. The package contains the model and training implementation, matched experimental protocols, source data used in the manuscript, and scripts for recomputing the reported statistics and figures.

## What is included

- `Burger2D/`: two-dimensional Burgers and Allen–Cahn implementations, soft MoE-PINN/APINN models, losses, training utilities, experiment scripts, and archived source outputs.
- `burger1D/`: one-dimensional Burgers and KdV implementations used for supporting experiments.
- `Burger2D/results/jcp_reference_rebuild_20260808/`: converged WENO5 reference solutions and fixed evaluation data used for the final Burgers re-evaluation.
- `Burger2D/results/` and `burger1D/results/`: machine-readable per-seed and summary outputs behind the manuscript tables and figures.
- `paper/`: submission-oriented English manuscript source and figure files.

Large neural-network checkpoints are intentionally excluded from Git history. Every reported checkpoint can be regenerated using the included experiment scripts. A separately versioned checkpoint archive can be attached to a GitHub release or an archival repository.

## Environment

The experiments were developed with Python 3.10 and PyTorch with CUDA support. A GeForce RTX 4060 is sufficient for the reported runs when experiments are executed sequentially.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the main evidence

The scripts expose `--help`; use the smoke modes first to verify the environment before launching the full multi-seed protocols.

```bash
# Matched 2×2 information × update-chronology experiment
python Burger2D/scripts/run_equal_information_2x2.py --help

# Staged versus mixture-mediated co-adaptation
python Burger2D/scripts/run_true_staged_vs_coadapt.py --help

# Parameter-matched two-subnetwork APINN control
python Burger2D/scripts/run_apinn_seed_suite.py --help

# Recompute paired statistics and the routing-counterfactual figure
python Burger2D/scripts/jcp_reviewed_stats_and_counterfactuals.py
```

The final statistical script reads `Burger2D/results/jcp_reference_rebuild_20260808/main_checkpoint_reevaluation.json`. Paths can be overridden through the command-line options documented by each script; the manuscript-analysis scripts retain their original project-relative defaults for provenance.

## Evidence map

| Manuscript component | Reproduction entry point | Archived source data |
|---|---|---|
| 2×2 information × chronology | `Burger2D/scripts/run_equal_information_2x2.py` | `Burger2D/results/equal_info_2x2_confirmatory_20260806/` |
| Staged vs co-adaptive paired study | `Burger2D/scripts/run_true_staged_vs_coadapt.py` | `Burger2D/results/true_staged_vs_coadapt_20260806/` |
| WENO5 checkpoint re-evaluation | `Burger2D/scripts/run_jcp_reference_rebuild_pipeline.py` | `Burger2D/results/jcp_reference_rebuild_20260808/` |
| Routing stress tests and paired statistics | `Burger2D/scripts/jcp_reviewed_stats_and_counterfactuals.py` | `Burger2D/results/jcp_reviewed_20260809/` |
| Allen–Cahn support | `Burger2D/scripts/run_allen_cahn_seeds.py` | `Burger2D/results/allen_cahn/` |
| Parameter-matched APINN | `Burger2D/scripts/run_apinn_seed_suite.py` | `Burger2D/results/apinn_reproduction/` |
| KdV schedule support | `burger1D/scripts/run_gate_intro_kdv.py` | `burger1D/results/gate_intro_kdv_20260805_005901/` |

## Reproducibility notes

- The main confirmatory comparison uses ten paired seeds and a common WENO5 evaluation reference.
- Descriptive values are mean ± sample standard deviation.
- Paired differences use deterministic paired bootstrap intervals and exact two-sided sign-flip tests; the nine primary endpoints are Holm-adjusted.
- The manuscript distinguishes mixture accuracy, complete-branch reliability, responsibility-region performance, capability-region performance, oracle behavior, and post-hoc routing interventions.
- Results should be interpreted at the level of the stated training protocols and PDE settings, not as a claim that every joint optimizer must fail.

## Citation

Citation metadata will be added when a DOI or journal record becomes available.

## License

No reuse license has been selected yet. Until a license file is added, copyright remains with the author.
