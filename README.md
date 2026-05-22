# CS224R Custom Final Project

SEACTS: Sequential Evidence Acquisition for Cancer Target Selection

## Objective

Modern cancer target discovery is incredibly difficult, with clinical drug development failure rates often exceeding 90% (Sun et al., 2022). This process relies on integrating diverse biological evidence sources like gene expression, mutations, and pathway context to identify genes whose perturbation impairs tumor viability. In practice, these modalities are highly heterogeneous, uneven in quality, and often redundant, yet most existing methods are designed under the assumption that all relevant data are available at decision time and should be used simultaneously. This obscures a key decision-making problem: which evidence is actually necessary for a given cancer context, and when does additional information cease to be worth its cost? In this project, we focus on cancer dependency prediction using DepMap data and approach target selection as a sequential decision-making problem under a budget constraint. Given a cancer cell line and a set of candidate genes, an agent must decide which biological evidence to acquire, in what order, and when to stop before selecting a gene to target. We model this process using deep reinforcement learning, where the agent is rewarded for selecting genes with strong dependency, using CRISPR dependency as a proxy for intervention effectiveness, while minimizing the cost of evidence acquisition. In our simulated environment, query costs serve as proxies for experimental, computational, or assay burden in real target-discovery pipelines. Our objective is to determine whether context-conditioned, adaptive evidence acquisition can approach the performance of static multi-omics models while using fewer modalities, and to understand how optimal evidence strategies vary across cancer types.

## Related Work

Cancer dependency prediction has been extensively studied using large-scale data such as DepMap, which integrates CRISPR gene knockout screens with genomic, transcriptomic, and copy-number data across hundreds of cancer cell lines (Broad Institute, n.d.; Tsherniak et al., 2017). These resources have enabled predictive models of gene essentiality and identification of therapeutic vulnerabilities, with recent work further constructing clinically informed dependency maps for target prioritization (Pacini et al., 2024) and extending these ideas to translational settings by learning predictors that generalize from cell lines to patient tumors (Shi et al., 2024). However, these approaches rely on static multi-omics integration, treating all modalities as simultaneously available rather than modeling how evidence should be acquired under constraints.

A closely related line of work is active feature acquisition (AFA), which formulates prediction as a sequential decision problem in which an agent selects features to observe while trading off predictive accuracy and acquisition cost. Reinforcement learning (RL) has been widely applied in this setting, including early deep RL approaches for cost-aware feature selection (Janisch et al., 2018) and more recent work on active modality selection in medical diagnosis (Bernardino et al., 2022). Recent advances have explored structured acquisition strategies and information-theoretic objectives, highlighting limitations of both RL-based policies and greedy approaches (Huang et al., 2026). While these methods capture sequential decision-making, they typically treat inputs as homogeneous features and focus on classification tasks. In contrast, biological evidence sources are heterogeneous and semantically distinct, and the downstream objective is often target ranking or intervention selection rather than simple prediction.

Another relevant direction is mixture-of-experts (MoE) and gating models, which learn to route inputs to specialized predictors (Shazeer et al., 2017). While these approaches capture modality specialization, they generally make single-step routing decisions and do not model sequential querying, stopping, or cost-aware decision-making. More broadly, Deep Q-Networks provide a standard framework for learning value functions over discrete actions with delayed rewards (Mnih et al., 2015), making them a natural fit for a finite-horizon evidence-acquisition environment. However, this framework has not been applied to settings where biological evidence acquisition and target selection are jointly optimized.

Our work bridges these areas by reformulating cancer target selection as a sequential, cost-aware decision problem. Unlike prior AFA methods, we explicitly model target selection as an action and evaluate performance using dependency-based outcomes. Unlike static multi-omics models, we allow the policy to adapt its evidence acquisition strategy based on cancer context, enabling analysis of how different biological settings influence optimal decision-making. To our knowledge, prior work has not explored jointly optimizing sequential evidence acquisition and target selection in cancer dependency maps under cancer-context conditioning.

## Technical Outline

We model cancer target selection as a finite-horizon decision process constructed from DepMap data, where each episode corresponds to a cancer cell line and a set of candidate genes. To keep the action space tractable, each episode uses a fixed-size candidate set containing at least one highly dependent gene and matched non-dependent genes, with dependency scores hidden from the agent until reward computation. The agent begins with limited information and must sequentially decide which biological evidence to query before selecting a gene to target. At each step, the state consists of the cancer context, such as the cell line or lineage, a mask indicating which modalities have been queried for each candidate gene, and the observed outputs of modality-specific predictors such as expression-based, mutation/CNA-based, or pathway/network-based scores. These modality-specific predictors are trained using supervised learning and provide partial, complementary signals about gene dependency.

The agent can take two types of actions: querying a modality for a specific gene, or terminating and selecting a gene as the final target. Query actions reveal the corresponding modality-specific information deterministically from the dataset, simulating the process of acquiring biological evidence. The episode ends when the agent selects a gene, at which point it receives a reward based on a transformed dependency score of the chosen gene, where higher reward corresponds to stronger cancer dependency, with an additional penalty for the number of queries made. This creates a delayed reward setting in which the agent must balance gathering more information against the cost of doing so.

We train a Deep Q-Network (DQN) to learn a policy over this discrete action space. The model takes as input a vectorized representation of the partially observed state and outputs scores for each possible action. Training uses experience replay and target networks to stabilize learning, and the fully simulated environment allows efficient generation of training trajectories without additional data collection. We evaluate our approach against several baselines, including full multi-omics models that use all modalities, fixed modality subsets, random acquisition policies under the same budget, greedy strategies based on feature importance or uncertainty, and one-shot gating models that select modalities without sequential decision-making. For prediction, we report AUC/AUPRC and ranking metrics such as NDCG based on the model’s final dependency estimates over the candidate set. For the final target-selection action, we report the dependency score of the selected gene and hit rate for selecting a gene among the top-k most dependent candidates. We additionally report cost-performance curves showing how these metrics change as the average number of evidence queries varies. We analyze learned policies by visualizing modality usage across cancer types and examining decision trajectories, including a cancer-type-by-modality heatmap to assess whether the agent learns biologically interpretable, context-specific strategies. We also analyze failure cases, including settings where the learned policy over-queries redundant evidence, stops too early, or is misled by noisy modality-specific predictors. As a reach goal, we explore a multi-agent variant in which modality-specific experts learn to communicate compressed information to the central policy.

## Milestone Plan

The first milestone is an end-to-end baseline and environment pipeline:

1. Load DepMap-style dependency and modality matrices.
2. Build fixed-size candidate gene episodes for each cell line.
3. Run direct data baselines that rank candidates from already-available modality matrices.
4. Run RL environment baselines that must query evidence through the sequential query/select API.
5. Report ranking, selected-target, query-cost, and episode-reward metrics.

This gives us the shared data/evaluation surface and environment API needed before adding the DQN policy.

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
├── config.py                # YAML config loading
├── data.py                  # DepMap-style matrix loading
├── data_baseline_runner.py  # Direct data baseline runner
├── data_baselines.py        # Direct data baseline policies
├── depmap_files.py          # DepMap manifest filtering and downloads
├── environment.py           # Sequential evidence-acquisition environment
├── environment_baselines.py # RL environment baseline policies
├── environment_runner.py    # RL environment baseline runner
├── episodes.py              # Candidate episode construction
└── metrics.py               # Ranking and selection metrics

scripts/
├── run_data_baselines.py
└── run_environment_baselines.py

modal_data.py                  # DepMap download/prep launcher
modal_data_baselines.py        # Modal data-baseline runner
modal_environment_baselines.py # Modal environment-baseline runner
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

The downloader also keeps project context files for later cancer-context features and analysis:

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
- `rl_env_query_{modality}_then_select`: queries one modality for every candidate, then selects by the revealed values.
- `rl_env_query_all_average_then_select`: queries all modalities for every candidate, then selects by the standardized average of revealed values.

Environment metrics include selected dependency score, hit rate at k, NDCG at k, reciprocal rank at k, query cost, number of queries, and total episode reward.

The RL environment uses normalized query costs from `configs/depmap_baselines.yaml`. These costs are relative burden proxies on the same scale as dependency reward, not literal assay prices. The default values treat already available computational evidence as low-cost while still making exhaustive querying non-free.

Baseline runs log to the W&B project `seacts/seacts` when `tracking.wandb.enabled` is true in `configs/depmap_baselines.yaml`.

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
