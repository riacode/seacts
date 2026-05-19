# SEACTS

CS224R Custom Final Project: Sequential Evidence Acquisition for Cancer Target Selection

## Objective

Modern cancer target discovery is incredibly difficult, with clinical drug development failure rates often exceeding 90% [SUN20223049]. This process relies on integrating diverse biological evidence sources like gene expression, mutations, and pathway context to identify genes whose perturbation impairs tumor viability. In practice, these modalities are highly heterogeneous, uneven in quality, and often redundant, yet most existing methods are designed under the assumption that all relevant data are available at decision time and should be used simultaneously. This obscures a key decision-making problem: which evidence is actually necessary for a given cancer context, and when does additional information cease to be worth its cost? In this project, we focus on cancer dependency prediction using DepMap data and approach target selection as a sequential decision-making problem under a budget constraint. Given a cancer cell line and a set of candidate genes, an agent must decide which biological evidence to acquire, in what order, and when to stop before selecting a gene to target. We model this process using deep reinforcement learning, where the agent is rewarded for selecting genes with strong dependency, using CRISPR dependency as a proxy for intervention effectiveness, while minimizing the cost of evidence acquisition. In our simulated environment, query costs serve as proxies for experimental, computational, or assay burden in real target-discovery pipelines. Our objective is to determine whether context-conditioned, adaptive evidence acquisition can approach the performance of static multi-omics models while using fewer modalities, and to understand how optimal evidence strategies vary across cancer types.

## Related Work

Cancer dependency prediction has been extensively studied using large-scale data such as DepMap, which integrates CRISPR gene knockout screens with genomic, transcriptomic, and copy-number data across hundreds of cancer cell lines [depmap_portal; tsherniak2017dependency]. These resources have enabled predictive models of gene essentiality and identification of therapeutic vulnerabilities, with recent work further constructing clinically informed dependency maps for target prioritization [pacini2024dependency] and extending these ideas to translational settings by learning predictors that generalize from cell lines to patient tumors [shi2024tcga_dependency]. However, these approaches rely on static multi-omics integration, treating all modalities as simultaneously available rather than modeling how evidence should be acquired under constraints.

A closely related line of work is active feature acquisition (AFA), which formulates prediction as a sequential decision problem in which an agent selects features to observe while trading off predictive accuracy and acquisition cost. Reinforcement learning (RL) has been widely applied in this setting, including early deep RL approaches for cost-aware feature selection [janisch2019costly_features] and more recent work on active modality selection in medical diagnosis [bernardino2022active_modality]. Recent advances have explored structured acquisition strategies and information-theoretic objectives, highlighting limitations of both RL-based policies and greedy approaches [huang2026information_templates]. While these methods capture sequential decision-making, they typically treat inputs as homogeneous features and focus on classification tasks. In contrast, biological evidence sources are heterogeneous and semantically distinct, and the downstream objective is often target ranking or intervention selection rather than simple prediction.

Another relevant direction is mixture-of-experts (MoE) and gating models, which learn to route inputs to specialized predictors [shazeer2017moe]. While these approaches capture modality specialization, they generally make single-step routing decisions and do not model sequential querying, stopping, or cost-aware decision-making. More broadly, Deep Q-Networks provide a standard framework for learning value functions over discrete actions with delayed rewards [mnih2015dqn], making them a natural fit for a finite-horizon evidence-acquisition environment. However, this framework has not been applied to settings where biological evidence acquisition and target selection are jointly optimized.

Our work bridges these areas by reformulating cancer target selection as a sequential, cost-aware decision problem. Unlike prior AFA methods, we explicitly model target selection as an action and evaluate performance using dependency-based outcomes. Unlike static multi-omics models, we allow the policy to adapt its evidence acquisition strategy based on cancer context, enabling analysis of how different biological settings influence optimal decision-making. To our knowledge, prior work has not explored jointly optimizing sequential evidence acquisition and target selection in cancer dependency maps under cancer-context conditioning.

## Technical Outline

We model cancer target selection as a finite-horizon decision process constructed from DepMap data, where each episode corresponds to a cancer cell line and a set of candidate genes. To keep the action space tractable, each episode uses a fixed-size candidate set containing at least one highly dependent gene and matched non-dependent genes, with dependency scores hidden from the agent until reward computation. The agent begins with limited information and must sequentially decide which biological evidence to query before selecting a gene to target. At each step, the state consists of the cancer context, such as the cell line or lineage, a mask indicating which modalities have been queried for each candidate gene, and the observed outputs of modality-specific predictors such as expression-based, mutation/CNA-based, or pathway/network-based scores. These modality-specific predictors are trained using supervised learning and provide partial, complementary signals about gene dependency.

The agent can take two types of actions: querying a modality for a specific gene, or terminating and selecting a gene as the final target. Query actions reveal the corresponding modality-specific information deterministically from the dataset, simulating the process of acquiring biological evidence. The episode ends when the agent selects a gene, at which point it receives a reward based on a transformed dependency score of the chosen gene, where higher reward corresponds to stronger cancer dependency, with an additional penalty for the number of queries made. This creates a delayed reward setting in which the agent must balance gathering more information against the cost of doing so.

We train a Deep Q-Network (DQN) to learn a policy over this discrete action space. The model takes as input a vectorized representation of the partially observed state and outputs scores for each possible action. Training uses experience replay and target networks to stabilize learning, and the fully simulated environment allows efficient generation of training trajectories without additional data collection. We evaluate our approach against several baselines, including full multi-omics models that use all modalities, fixed modality subsets, random acquisition policies under the same budget, greedy strategies based on feature importance or uncertainty, and one-shot gating models that select modalities without sequential decision-making. For prediction, we report AUC/AUPRC and ranking metrics such as NDCG based on the model’s final dependency estimates over the candidate set. For the final target-selection action, we report the dependency score of the selected gene and hit rate for selecting a gene among the top-k most dependent candidates. We additionally report cost-performance curves showing how these metrics change as the average number of evidence queries varies. We analyze learned policies by visualizing modality usage across cancer types and examining decision trajectories, including a cancer-type-by-modality heatmap to assess whether the agent learns biologically interpretable, context-specific strategies. We also analyze failure cases, including settings where the learned policy over-queries redundant evidence, stops too early, or is misled by noisy modality-specific predictors. As a reach goal, we explore a multi-agent variant in which modality-specific experts learn to communicate compressed information to the central policy.

## Milestone Plan

The first milestone is an end-to-end baseline pipeline:

1. Load DepMap-style dependency and modality matrices.
2. Build fixed-size candidate gene episodes for each cell line.
3. Run target-selection baselines.
4. Report ranking and selected-target metrics.

This gives us the shared data/evaluation surface needed before adding the DQN evidence-acquisition environment.

## Setup

```bash
conda env create -f conda_env_local.yml
conda activate seacts-local
```

If the environment already exists after dependency changes:

```bash
conda env update -f conda_env_local.yml --prune
conda activate seacts-local
```

## Modal

Install and authenticate the Modal client in the conda environment:

```bash
conda activate seacts-local
modal setup
```

Download the required DepMap files into the `seacts-data` Modal Volume:

```bash
modal run modal_data.py
```

The downloader fetches the DepMap manifest fresh from `https://depmap.org/portal/api/download/files`, selects the latest `DepMap Public` release by default, and downloads:

- `CRISPRGeneEffect.csv`
- `Model.csv` or `sample_info.csv`
- `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`
- `PortalOmicsCNGeneLog2.csv`
- `OmicsSomaticMutationsMatrixDamaging.csv`

## Codebase Structure

The code follows the course assignment style, with implementation code grouped under `src/` and runnable entry points kept separate:

```text
src/
├── baselines.py      # Candidate target-selection baselines
├── config.py         # YAML config loading
├── data.py           # DepMap-style matrix loading
├── depmap_files.py   # DepMap manifest filtering and downloads
├── episodes.py       # Candidate episode construction
└── metrics.py        # Ranking and selection metrics

scripts/
└── run_baselines.py

modal_config.py       # Shared Modal app, image, volume, remote functions
modal_data.py         # DepMap download/prep launcher
```

## Real Data

The project is designed to download real DepMap files on Modal rather than storing large raw data locally. The required raw files are persisted in the `seacts-data` Modal Volume.

For real DepMap filenames, use `configs/depmap_baselines.yaml`:

- `data/raw/CRISPRGeneEffect.csv`
- `data/raw/Model.csv`
- `data/raw/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`
- `data/raw/OmicsSomaticMutationsMatrixDamaging.csv`
- `data/raw/PortalOmicsCNGeneLog2.csv`

Matrices should have cell lines as rows and genes as columns. Long-form files with `cell_line_id`, `gene`, and `value` columns are also supported.

## Current Baselines

- `random`: random candidate gene selection.
- `oracle_dependency`: ranks by hidden dependency score as an upper bound.
- `{modality}_score`: ranks by one modality score.
- `average_all_modalities`: averages all available modality scores.

Metrics include selected dependency score, hit rate at k, NDCG at k, reciprocal rank at k, and query cost.

## References

- [SUN20223049] Sun, D., Gao, W., Hu, H., & Zhou, S. (2022). Why 90% of clinical drug development fails and how to improve it? *Acta Pharmaceutica Sinica B*. https://doi.org/10.1016/j.apsb.2022.02.002
- [depmap_portal] Broad Institute. (2026). DepMap Portal. https://depmap.org/portal/
- [tsherniak2017dependency] Tsherniak, A., Vazquez, F., Montgomery, P. G., et al. (2017). Defining a Cancer Dependency Map. *Cell*. https://doi.org/10.1016/j.cell.2017.06.010
- [pacini2024dependency] Pacini, C., Dempster, J. M., Boyle, I., et al. (2024). A comprehensive clinically informed map of dependencies in cancer cells and framework for target prioritization. *Cancer Cell*. https://doi.org/10.1016/j.ccell.2023.12.004
- [shi2024tcga_dependency] Shi, J., Aguirre, A. J., et al. (2024). Building a translational cancer dependency map for The Cancer Genome Atlas. *Nature Cancer*. https://doi.org/10.1038/s43018-024-00789-y
- [janisch2019costly_features] Janisch, J., Pevný, T., & Lisý, V. (2018). Classification with Costly Features using Deep Reinforcement Learning. https://arxiv.org/abs/1711.07364
- [bernardino2022active_modality] Bernardino, G., Jonsson, A., Loncaric, F., et al. (2022). Reinforcement Learning for Active Modality Selection During Diagnosis. *MICCAI 2022*.
- [huang2026information_templates] Huang, H.-T., Dinh, D., & Oliva, J. B. (2026). Information Templates: A New Paradigm for Intelligent Active Feature Acquisition. https://arxiv.org/abs/2508.18380
- [shazeer2017moe] Shazeer, N., Mirhoseini, A., Maziarz, K., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR*. https://arxiv.org/abs/1701.06538
- [mnih2015dqn] Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. *Nature*. https://doi.org/10.1038/nature14236
