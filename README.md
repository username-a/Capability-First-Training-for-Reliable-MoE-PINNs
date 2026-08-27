# Capability-First Training for Reliable MoE-PINNs

This repository provides code, configurations, numerical data, and processed results for the manuscript:

> **From Aggregate Accuracy to Branch Reliability: A Capability-First Training Paradigm for Mixture-of-Experts Physics-Informed Neural Networks**  
> Xiang Li and Liping Huang  
> Prepared for submission to *Computers & Mathematics with Applications*

The study distinguishes accurate soft aggregation from reliable expert decomposition. It introduces a branch-reliability audit and capability-first staged training (CFST), then evaluates aggregate accuracy, complete-branch capability, gate--capability alignment, and robustness under routing interventions.

## Submission snapshot

The complete submission-aligned reproducibility snapshot is available from the [CAMWA submission reproducibility release](https://github.com/username-a/Capability-First-Training-for-Reliable-MoE-PINNs/releases/tag/camwa-submission-v1). The release asset includes the final high-budget ten-seed Burgers results, exact weighted-error cross-term analysis, configurations, numerical reference data, and machine-readable summaries used in the manuscript.

## Repository contents

- `Burger2D/`: two-dimensional Burgers and Allen--Cahn models, training and evaluation scripts, numerical references, configurations, and archived results
- `burger1D/`: one-dimensional Burgers and KdV implementations and supporting experiments
- `data/`: numerical data used by the included workflows
- `paper/`: CAMWA-formatted manuscript source, compiled manuscript, supplementary information, highlights, and figures

## Primary workflow

The main high-budget protocol is implemented through `Burger2D/scripts/run_true_staged_vs_coadapt.py`. The submission release additionally contains the final evaluation and exact cross-term analysis scripts together with their outputs.

The recorded primary environment is Python 3.10.10, PyTorch 2.6.0+cu124, NumPy 2.2.6, and SciPy 1.15.3. See `requirements.txt` for installation requirements.

Large neural-network checkpoints are excluded because they can be regenerated from the supplied scripts and configurations. No human or animal participant data are present.

## Citation

Please cite the associated manuscript. Formal bibliographic information will be added after a journal record becomes available.

## License

No reuse license has been selected. Copyright remains with the authors unless a license is added later.
