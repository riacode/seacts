# CS224R Custom Final Project

SEACTS: Sequential Evidence Acquisition for Cancer Target Selection

## Objective

Modern cancer target discovery is incredibly difficult, with clinical drug development failure rates often exceeding 90% (Sun et al., 2022). This process relies on integrating diverse biological evidence sources like gene expression, copy-number alteration, and mutation data to identify genes whose perturbation impairs tumor viability. In practice, these modalities are highly heterogeneous, uneven in quality, and often redundant, yet most existing methods are designed under the assumption that all relevant data are available at decision time and should be used simultaneously. This obscures a key decision-making problem: which evidence is actually necessary for a given cancer context, and when does additional information cease to be worth its cost? In this project, we focus on cancer dependency prediction using DepMap data and approach target selection as a sequential decision-making problem under a budget constraint. Given a cancer cell line and a set of candidate genes, an agent must decide which biological evidence to acquire, in what order, and when to stop before selecting a gene to target. We model this process using deep reinforcement learning, where the agent is rewarded for selecting genes with strong dependency, using CRISPR dependency as a proxy for intervention effectiveness, while minimizing the cost of evidence acquisition. In our simulated environment, query costs serve as proxies for experimental, computational, or assay burden in real target-discovery pipelines. Our objective is to determine whether adaptive evidence acquisition can approach fixed full-modality policies while using fewer queries.

## Related Work

Cancer dependency prediction has been extensively studied using large-scale data such as DepMap, which integrates CRISPR gene knockout screens with genomic, transcriptomic, and copy-number data across hundreds of cancer cell lines (Broad Institute, n.d.; Tsherniak et al., 2017). These resources have enabled predictive models of gene essentiality and identification of therapeutic vulnerabilities, with recent work further constructing clinically informed dependency maps for target prioritization (Pacini et al., 2024) and extending these ideas to translational settings by learning predictors that generalize from cell lines to patient tumors (Shi et al., 2024). However, these approaches rely on static multi-omics integration, treating all modalities as simultaneously available rather than modeling how evidence should be acquired under constraints.

A closely related line of work is active feature acquisition (AFA), which formulates prediction as a sequential decision problem in which an agent selects features to observe while trading off predictive accuracy and acquisition cost. Reinforcement learning (RL) has been widely applied in this setting, including early deep RL approaches for cost-aware feature selection (Janisch et al., 2018) and more recent work on active modality selection in medical diagnosis (Bernardino et al., 2022). Recent advances have explored structured acquisition strategies and information-theoretic objectives, highlighting limitations of both RL-based policies and greedy approaches (Huang et al., 2026). While these methods capture sequential decision-making, they typically treat inputs as homogeneous features and focus on classification tasks. In contrast, biological evidence sources are heterogeneous and semantically distinct, and the downstream objective is often target ranking or intervention selection rather than simple prediction.

Another relevant direction is mixture-of-experts (MoE) and gating models, which learn to route inputs to specialized predictors (Shazeer et al., 2017). While these approaches capture modality specialization, they generally make single-step routing decisions and do not model sequential querying, stopping, or cost-aware decision-making. More broadly, Deep Q-Networks provide a standard framework for learning value functions over discrete actions with delayed rewards (Mnih et al., 2015), making them a natural fit for a finite-horizon evidence-acquisition environment. However, this framework has not been applied to settings where biological evidence acquisition and target selection are jointly optimized.

Our work bridges these areas by reformulating cancer target selection as a sequential, cost-aware decision problem. Unlike prior AFA methods, we explicitly model target selection as an action and evaluate performance using dependency-based outcomes. Unlike static multi-omics models, we allow the policy to adapt its evidence acquisition strategy to each candidate set and partial evidence state. To our knowledge, prior work has not explored jointly optimizing sequential evidence acquisition and target selection in cancer dependency maps under this query/select formulation.

## Technical Outline

We model cancer target selection as a finite-horizon decision process constructed from DepMap data, where each episode corresponds to a cancer cell line and a set of candidate genes. To keep the action space tractable, each episode uses a fixed-size candidate set containing at least one highly dependent gene and matched non-dependent genes, with dependency scores hidden from the agent until reward computation. The agent begins with limited information and must sequentially decide which biological evidence to query before selecting a gene to target. At each step, the state consists of a mask indicating which modalities have been queried for each candidate gene and the observed outputs of modality-specific evidence models for expression, copy number, damaging mutation, and hotspot mutation. In the supervised-score environment used for the main results, query actions reveal simple per-gene dependency evidence scores fit on training cell lines, providing calibrated partial signals while keeping CRISPR dependency hidden for reward and evaluation.

The agent can take two types of actions: querying a modality for a specific gene, or terminating and selecting a gene as the final target. Query actions reveal the corresponding modality-specific information deterministically from the dataset, simulating the process of acquiring biological evidence. The episode ends when the agent selects a gene, at which point it receives a reward based on a transformed dependency score of the chosen gene, where higher reward corresponds to stronger cancer dependency, with an additional penalty for the number of queries made. This creates a delayed reward setting in which the agent must balance gathering more information against the cost of doing so.

We train Deep Q-Network (DQN) policies over this discrete action space. The baseline DQN takes a vectorized representation of the partially observed state and outputs scores for each possible action; our main ablation also evaluates structured candidate-aware and structured dueling Q-networks. Training uses Double DQN updates, experience replay, target networks, action masking, and validation checkpointing. We evaluate against raw modality-ranking baselines, oracle/random upper and lower bounds, and fixed sequential policies that query a modality and rank candidates from the revealed scores. We report selected dependency, Hit@k, NDCG@k, MRR@k, query cost, number of queries, and total reward. We additionally analyze learned policies through per-episode behavior summaries, per-step action logs, modality-usage plots, cancer-context modality-usage heatmaps, query-efficiency plots, regret-vs-query plots, and example evidence-acquisition trajectories.

## Contributions

SEACTS provides an end-to-end framework for studying cost-aware, sequential
evidence acquisition for cancer target selection:

1. DepMap download and persistence are handled on Modal using `seacts-data` and
   `seacts-results` volumes.
2. The data pipeline loads CRISPR dependency, model metadata, expression, copy
   number, damaging mutation, and hotspot mutation matrices.
3. Candidate episodes are built from fixed-size cell-line/gene sets with hidden
   dependency scores used only for reward and evaluation.
4. Direct data baselines rank candidates from already available modality
   matrices.
5. RL environment baselines use the sequential `QUERY(gene, modality)` and
   `SELECT(gene)` API with normalized query costs, including fixed-budget
   expression and CNA baselines for query-budget comparison.
6. The RL environment can replace raw modality values with simple supervised
   per-gene modality scores so queries reveal dependency-prediction evidence
   rather than uncalibrated raw omics values.
7. A Double DQN policy trains on the same environment with action masking,
   replay, target networks, validation checkpointing, selection-aware
   exploration, optional expert replay seeding, and configurable cell-line
   train/validation/evaluation splits.
8. Metrics are logged to W&B and saved as CSVs, including selected dependency,
   Hit@k, NDCG@k, MRR@k, query cost, number of queries, modality usage, and
   total reward.
9. Visualization scripts compare baseline and DQN metrics, generate
   poster-style qualitative behavior figures, and log them to W&B.
10. DQN behavior logging records per-episode summaries and per-step actions,
    including selected genes, dependency regret, query counts, observed
    evidence values, and true dependency ranks for queried genes.
11. A Modal DQN sweep entrypoint runs targeted hyperparameter variants for
    model selection around the best structured 1-step DQN family.
12. A DQN ablation entrypoint compares the flat MLP Q-network against N-step,
    structured candidate-encoder, and structured dueling DQN variants under the
    same reward, query-cost, and minimum-evidence settings.

## Results

The main reported results use the held-out cell-line split, supervised
modality-score environment, equal normalized query costs, and minimum 8 queries
before selection. We report two baseline families: raw modality rankings, which
score candidates directly from already available matrices, and sequential
environment policies, which pay to query calibrated evidence before selecting a
target. In the sequential environment, queried evidence values are simple
per-gene supervised dependency scores fit on training cell lines, not
neural-network predictions.

### Raw Modality Ranking Baselines

| Policy | Selected dependency | Hit@k | NDCG@k | MRR@k | Query cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle dependency | -1.423 | 1.000 | 1.000 | 1.000 | 0.0 |
| Expression score | -0.665 | 0.875 | 0.527 | 0.432 | 1.0 |
| Average all modalities | -0.460 | 0.780 | 0.429 | 0.306 | 3.0 |
| CNA score | -0.264 | 0.558 | 0.267 | 0.160 | 1.0 |
| Random select | -0.164 | 0.483 | 0.221 | 0.104 | 0.0 |
| Hotspot mutation score | -0.160 | 0.460 | 0.213 | 0.109 | 1.0 |
| Damaging mutation score | -0.144 | 0.455 | 0.206 | 0.104 | 1.0 |

These baselines do not use the sequential environment; they rank each episode's
candidates directly from raw modality matrices. Expression is the strongest
single raw modality signal, but direct modality ranking remains far from the
hidden dependency oracle.

### Sequential Environment Policies

These policies interact with the query/select environment. Fixed policies query
evidence and rank candidates from the revealed scores, while DQN chooses which
candidate-modality pairs to query before selecting a final target.

| Policy | Total reward | Selected dependency | Queries | Hit@k | NDCG@k | MRR@k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle select | 1.423 | -1.423 | 0.0 | 1.000 | 1.000 | 1.000 |
| Structured DQN, 1-step larger | **1.035** | -1.287 | 12.6 | **1.000** | 0.847 | 0.807 |
| Query CNA full | 1.011 | -1.331 | 16.0 | 0.998 | 0.919 | 0.841 |
| Query expression full | 1.005 | -1.325 | 16.0 | 0.998 | 0.921 | 0.842 |
| Query CNA budget 12 | 0.927 | -1.167 | 12.0 | 0.975 | 0.760 | 0.648 |
| Random select | 0.164 | -0.164 | 0.0 | 0.483 | 0.221 | 0.104 |

The learned policy slightly outperforms full CNA and full expression baselines
on cost-adjusted reward while using about 21% fewer queries. The full-query
baselines still select slightly stronger raw dependencies, but the policy
achieves a better quality-cost tradeoff by stopping before exhaustive evidence
collection. This larger structured 1-step model is the best sweep variant; it
uses a 256-dimensional hidden layer instead of the 128-dimensional hidden layer
used by the default structured 1-step ablation model.

### DQN Ablation

| Model | Total reward | Selected dependency | Queries | Hit@k | NDCG@k | MRR@k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Structured DQN, 1-step | **1.023** | -1.261 | 11.90 | **1.000** | **0.829** | 0.760 |
| Structured dueling DQN, 3-step | 1.021 | -1.234 | 10.64 | 0.998 | 0.787 | 0.710 |
| Structured dueling DQN, 1-step | 0.991 | **-1.274** | 14.16 | 0.998 | 0.825 | **0.785** |
| Structured DQN, 3-step | 0.978 | -1.177 | 9.98 | 0.998 | 0.757 | 0.666 |
| MLP DQN, 1-step | 0.845 | -1.181 | 16.80 | 0.980 | 0.706 | 0.671 |
| MLP DQN, 3-step | 0.799 | -0.990 | 9.56 | 0.918 | 0.616 | 0.536 |

The ablation suggests that candidate-structured Q-functions are the main source
of improvement. The MLP DQN flattens the state and outputs one Q-value per
action. The structured DQN instead encodes each candidate gene with its observed
values and query mask, pools candidate encodings into an episode context, and
uses separate query/select heads. This explicitly shares information across
actions for the same gene and across repeated modality types. 3-step returns
alone hurt the MLP, likely because they add target variance without enough
action structure. Dueling helps most in the 3-step structured setting, where
the value stream can stabilize partially observed state estimates, but it does
not beat the simpler structured 1-step model on total reward.


## Setup

```bash
conda env create -f conda_env_local.yml
conda activate seacts
```

If the environment already exists after dependency changes:

```bash
conda env update -f conda_env_local.yml --prune
conda activate seacts
```

Run the local checks from the environment:

```bash
conda run -n seacts pytest -q
conda run -n seacts ruff check .
```

The current expected test result is `54 passed`.

## Modal

Install and authenticate the Modal client in the conda environment:

```bash
conda activate seacts
modal setup
```

Create a Modal secret for W&B so baseline runs log to the `seacts` team:

```bash
modal secret create wandb WANDB_API_KEY=<your-api-key>
```

Download the required DepMap files into the `seacts-data` Modal Volume:

```bash
modal run modal_data.py
```

Run the direct data baselines against the Modal data volume and write metrics to the `seacts-results` Modal Volume:

```bash
modal run modal_data_baselines.py
```

Run RL environment baselines through the sequential query/select API:

```bash
modal run modal_environment_baselines.py
```

Train and evaluate the first DQN policy against the same environment:

```bash
modal run modal_train_dqn.py
```

Log detailed DQN behavior for the six reported ablations and write
per-episode, per-step, and derived analysis CSVs:

```bash
modal run modal_log_dqn_behavior.py
```

Run targeted DQN hyperparameter variants and save a sweep summary:

```bash
modal run modal_sweep_dqn.py
```

Run Context DQN sweeps (context dim, dueling, structured checkpoint init, etc.):

```bash
modal run modal_sweep_context_dqn.py
```

Results are written to `depmap_baselines/dqn_context_sweeps/` on the `seacts-results` volume.

Run the DQN architecture/algorithm ablation:

```bash
modal run modal_ablate_dqn.py
```

Plot the saved comparison and ablation results and log the figures to W&B:

```bash
modal run modal_visualizations.py
```

The downloader fetches the DepMap manifest fresh from `https://depmap.org/portal/api/download/files`, selects the latest `DepMap Public` release by default, and downloads the dependency matrix, metadata, gene-aligned evidence matrices, and context files needed for the project:

- `CRISPRGeneEffect.csv`
- `Model.csv` or `sample_info.csv`
- `Gene.csv`
- `SubtypeMatrix.csv`
- `SubtypeTree.csv`
- `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`
- `PortalOmicsCNGeneLog2.csv`
- `OmicsSomaticMutationsMatrixDamaging.csv`
- `OmicsSomaticMutationsMatrixHotspot.csv`
- `OmicsGlobalSignatures.csv`
- `OmicsInferredMolecularSubtypes.csv`

## Codebase Structure

The code follows the course assignment style, with implementation code grouped under `src/` and runnable entry points kept separate:

```text
src/
├── behavior_analysis.py     # DQN behavior CSV summaries and context joins
├── config.py                # YAML config loading
├── data.py                  # DepMap-style matrix loading
├── data_baseline_runner.py  # Direct data baseline runner
├── data_baselines.py        # Direct data baseline policies
├── dqn.py                   # Double DQN network and optimization helpers
├── depmap_files.py          # DepMap manifest filtering and downloads
├── environment.py           # Sequential evidence-acquisition environment
├── environment_baselines.py # RL environment baseline policies
├── environment_runner.py    # RL environment baseline runner
├── episodes.py              # Candidate episode construction
├── metrics.py               # Ranking and selection metrics
├── replay_buffer.py         # Experience replay storage
├── rl_runner.py             # DQN training/evaluation runner
├── tracking.py              # W&B logging helpers
├── visualization.py         # Baseline and DQN behavior figures
└── state_encoder.py         # State vectorization and action indexing

scripts/
├── analyze_dqn_behavior.py
├── plot_baseline_results.py
├── run_data_baselines.py
├── run_environment_baselines.py
└── train_dqn.py

modal_ablate_dqn.py            # Modal DQN architecture/algorithm ablation runner
modal_data.py                  # DepMap download/prep launcher
modal_data_baselines.py        # Modal data-baseline runner
modal_environment_baselines.py # Modal environment-baseline runner
modal_log_dqn_behavior.py      # Modal DQN trajectory/behavior analysis
modal_sweep_dqn.py             # Modal DQN hyperparameter sweep runner
modal_sweep_context_dqn.py     # Modal Context DQN improvement sweep runner
modal_train_dqn.py             # Modal DQN training runner
modal_visualizations.py        # Modal plotting runner
src/modal_config.py            # Shared Modal app/image/volume setup
```

## Real Data

The project is designed to download real DepMap files on Modal rather than storing large raw data locally. The required raw files are persisted in the `seacts-data` Modal Volume.

For real DepMap filenames, use `configs/depmap_baselines.yaml`. The current baseline pipeline consumes gene-aligned cell-line-by-gene matrices:

- `data/raw/CRISPRGeneEffect.csv`
- `data/raw/Model.csv`
- `data/raw/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`
- `data/raw/OmicsSomaticMutationsMatrixDamaging.csv`
- `data/raw/OmicsSomaticMutationsMatrixHotspot.csv`
- `data/raw/PortalOmicsCNGeneLog2.csv`

The downloader also keeps context files for later analysis:

- `data/raw/Gene.csv`
- `data/raw/SubtypeMatrix.csv`
- `data/raw/SubtypeTree.csv`
- `data/raw/OmicsGlobalSignatures.csv`
- `data/raw/OmicsInferredMolecularSubtypes.csv`

Baseline modality matrices should have cell lines as rows and genes as columns. Long-form files with `cell_line_id`, `gene`, and `value` columns are also supported.

## Current Data Baselines

- `data_random_select`: random candidate gene selection.
- `data_oracle_dependency`: ranks by hidden dependency score as an upper bound.
- `data_{modality}_score`: ranks by one already-available modality score.
- `data_average_all_modalities`: averages within-episode standardized modality scores.

Metrics include selected dependency score, hit rate at k, NDCG at k, reciprocal rank at k, and query cost.

## Current RL Environment Baselines

- `rl_env_random_select`: selects a random candidate without querying evidence.
- `rl_env_oracle_select`: selects by hidden dependency as an upper bound.
- `rl_env_query_{modality}_then_select`: queries one modality for every candidate, then selects by the revealed evidence scores.
- `rl_env_query_expression_budget_{k}_then_select`: queries expression for a fixed budget of candidates, then selects by the revealed evidence scores.
- `rl_env_query_all_average_then_select`: queries all modalities for every candidate, then selects by the standardized average of revealed evidence scores.

Environment metrics include selected dependency score, hit rate at k, NDCG at k, reciprocal rank at k, query cost, number of queries, and total episode reward.

With `environment.use_supervised_modality_scores: true`, query actions reveal
simple per-gene supervised dependency evidence scores learned from the training
cell lines, rather than raw omics values. This makes modalities comparable while
keeping CRISPR dependency hidden from the agent until reward/evaluation.

The RL environment uses normalized query costs from `configs/depmap_baselines.yaml`. These costs are relative burden proxies on the same scale as dependency reward, not literal assay prices. The default values treat already available computational evidence as low-cost while still making exhaustive querying non-free.

Baseline runs log to the W&B project `seacts/seacts` when `tracking.wandb.enabled` is true in `configs/depmap_baselines.yaml`.

The DQN runner uses `rl_training.split_cell_lines: true` in the default config,
so train, validation, and evaluation episodes are sampled from disjoint eligible
cell-line pools. This gives a stricter estimate of whether the learned evidence
policy generalizes across cancer models rather than only across resampled
candidate sets from the same models. The default environment uses equal
normalized query costs and `selection_reward_scale: 1.0`, matching the reported
DQN ablation and sweep comparisons.

## DQN Behavior Analysis

After `modal_train_dqn.py` writes `dqn_policy.pt`, run
`modal_log_dqn_behavior.py` to reconstruct the evaluation environment, load the
checkpoint, and collect greedy-policy behavior. It writes:

- `dqn_episode_summary.csv`: one row per evaluation episode with selected gene,
  selected dependency, Hit@k, dependency regret, query cost, total reward, and
  per-modality query counts.
- `dqn_step_log.csv`: one row per DQN action with action type, gene, modality,
  observed evidence value, cumulative reward/cost, and true dependency rank of
  the acted-on gene.
- `behavior_analysis/dqn_failure_cases.csv`: high-regret or missed episodes for
  inspection.
- `behavior_analysis/dqn_success_cases.csv`: low-regret successful episodes.
- `behavior_analysis/dqn_query_efficiency_by_true_rank.csv`: how often query
  actions target genes at each true dependency rank.
- `behavior_analysis/dqn_modality_usage_by_context.csv`: average modality usage
  by inferred context column when `Model.csv` metadata is available.

W&B logs high-level behavior as charts where possible, including query-count
and dependency-regret histograms, modality usage, queried-gene rank
distributions, query-efficiency by true rank, regret-vs-query scatter plots,
trajectory strip plots, and cancer-context modality-usage heatmaps when
metadata includes a lineage-like column. Compact success and failure case
tables are logged for drill-down examples.

The same summaries can be generated locally from saved CSVs:

```bash
python scripts/analyze_dqn_behavior.py \
  --episodes outputs/depmap_baselines/dqn_episode_summary.csv \
  --steps outputs/depmap_baselines/dqn_step_log.csv \
  --metadata data/raw/Model.csv
```

## References

- Bernardino, G., Jonsson, A., Loncaric, F., Castellote, P.-M. M., Sitges, M., Clarysse, P., & Duchateau, N. (2022). Reinforcement learning for active modality selection during diagnosis. *Medical Image Computing and Computer Assisted Intervention - MICCAI 2022*, 592-601.
- Broad Institute. (n.d.). *DepMap Portal*. Retrieved May 19, 2026, from https://depmap.org/portal/
- Huang, H.-T., Dinh, D., & Oliva, J. B. (2026). *Information templates: A new paradigm for intelligent active feature acquisition*. arXiv. https://arxiv.org/abs/2508.18380
- Janisch, J., Pevný, T., & Lisý, V. (2018). *Classification with costly features using deep reinforcement learning*. arXiv. https://arxiv.org/abs/1711.07364
- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., et al. (2015). Human-level control through deep reinforcement learning. *Nature, 518*(7540), 529-533. https://doi.org/10.1038/nature14236
- Pacini, C., Dempster, J. M., Boyle, I., Goncalves, E., Najgebauer, H., et al. (2024). A comprehensive clinically informed map of dependencies in cancer cells and framework for target prioritization. *Cancer Cell, 42*, 301-316. https://doi.org/10.1016/j.ccell.2023.12.016
- Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *International Conference on Learning Representations*. https://arxiv.org/abs/1701.06538
- Shi, X., Gekas, C., Verduzco, D., Petiwala, S., Jeffries, C., Lu, C., Murphy, E., Anton, T., Vo, A. H., Xiao, Z., et al. (2024). Building a translational cancer dependency map for The Cancer Genome Atlas. *Nature Cancer, 5*, 1176-1194. https://doi.org/10.1038/s43018-024-00789-y
- Sun, D., Gao, W., Hu, H., & Zhou, S. (2022). Why 90% of clinical drug development fails and how to improve it? *Acta Pharmaceutica Sinica B, 12*(7), 3049-3062. https://doi.org/10.1016/j.apsb.2022.02.002
- Tsherniak, A., Vazquez, F., Montgomery, P. G., Weir, B. A., Kryukov, G., Cowley, G. S., Gill, S., Harrington, W. F., Pantel, S., Krill-Burger, J., et al. (2017). Defining a cancer dependency map. *Cell, 170*(3), 564-576. https://doi.org/10.1016/j.cell.2017.06.010
