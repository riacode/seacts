# SEACTS

Sequential Evidence Acquisition for Cancer Target Selection

## Summary

Cancer target discovery requires integrating heterogeneous biological evidence while deciding which genes are worth perturbing. Most dependency-map methods treat expression, copy-number, and mutation data as simultaneously available, but real target-discovery pipelines face costs for acquiring, computing, or interpreting each evidence source. We introduce SEACTS, a sequential evidence-acquisition framework for cancer target selection. Each episode presents one DepMap cancer cell line and a small candidate gene set; an agent may query modality-specific evidence for gene-modality pairs before selecting a final target. Query actions incur costs, while selection is rewarded according to hidden CRISPR dependency. We train Double DQN policies with candidate-structured Q-networks. On held-out cell-line episodes, the tuned Structured DQN reaches mean total reward $1.035$ with $12.6$ queries per episode, beating fixed full-modality query baselines while using fewer queries. A SELECT-only cancer-context variant further improves reward to $1.043$ with $12.4$ queries by using OncoTree lineage only at final target scoring. Ablations show that candidate-structured action representation is critical, while context is most useful for interpreting acquired evidence rather than broadly reshaping acquisition.

## Setup

```bash
conda env create -f conda_env_modal.yml
conda activate seacts
modal setup
modal secret create wandb WANDB_API_KEY=<your-api-key>   # if W&B enabled in config
```

Settings: `configs/depmap_baselines.yaml`

## Data

DepMap files download to the `seacts-data` Modal volume (not in git):

```bash
modal run modal_data.py
```

## Run

```bash
modal run modal_data_baselines.py
modal run modal_environment_baselines.py
modal run modal_train_dqn.py
modal run modal_sweep_dqn.py
modal run modal_sweep_context_dqn.py
modal run modal_ablate_dqn.py
modal run modal_log_dqn_behavior.py
modal run modal_visualizations.py
```

Best Context DQN variant:

```bash
modal run modal_sweep_context_dqn.py --variant ctx_select_only_init_frozen
modal run modal_log_dqn_behavior.py --variant ctx_select_only_init_frozen
```

Results: `seacts-results` volume → `depmap_baselines/`. Pull locally with `scripts/pull_modal_results.sh`.

## Layout

`src/` library · `modal_*.py` entrypoints · `configs/` · `tests/`

Submission zip: `./scripts/zip_submission.sh seacts-submission.zip`
