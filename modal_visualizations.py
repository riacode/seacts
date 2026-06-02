from __future__ import annotations

from pathlib import Path

from src.modal_config import app, image, results_volume, wandb_secret


RESULTS_ROOT = Path("/root/seacts/results/depmap_baselines")
CANCER_CONTEXT_DIR = RESULTS_ROOT / "dqn_cancer_context"


@app.function(
    image=image,
    volumes={"/root/seacts/results": results_volume},
    secrets=[wandb_secret],
    timeout=1800,
)
def plot_results_figures() -> list[str]:
    import wandb

    from src.visualization import (
        CANCER_CONTEXT_LABELS,
        CANCER_CONTEXT_VARIANT_ORDER,
        append_cancer_context_dqn_metrics,
        generate_ablation_figures,
        generate_baseline_figures,
        load_structured_dqn_poster_metrics,
    )

    results_volume.reload()

    dqn_results, dqn_trajectory_path = load_structured_dqn_poster_metrics(RESULTS_ROOT)
    print(f"Structured DQN metrics loaded ({len(dqn_results)} row(s))")
    dqn_results = append_cancer_context_dqn_metrics(dqn_results, CANCER_CONTEXT_DIR)
    print(f"Poster DQN policies: {', '.join(dqn_results['policy'].astype(str))}")

    figures = generate_baseline_figures(
        data_metrics_path=RESULTS_ROOT / "data_baseline_metrics.csv",
        environment_metrics_path=RESULTS_ROOT / "environment_baseline_metrics.csv",
        dqn_results=dqn_results,
        dqn_trajectory_path=dqn_trajectory_path,
        output_dir="/root/seacts/results/figures",
    )
    ablation_figures = generate_ablation_figures(
        ablation_summary_path=RESULTS_ROOT / "dqn_ablation" / "ablation_summary.csv",
        output_dir="/root/seacts/results/figures/dqn_ablation",
        ablation_dir=RESULTS_ROOT / "dqn_ablation",
    )
    figures.extend(ablation_figures)

    cancer_figures = generate_ablation_figures(
        ablation_summary_path=CANCER_CONTEXT_DIR / "ablation_summary.csv",
        output_dir="/root/seacts/results/figures/dqn_cancer_context",
        ablation_dir=CANCER_CONTEXT_DIR,
        variant_order=CANCER_CONTEXT_VARIANT_ORDER,
        ablation_labels=CANCER_CONTEXT_LABELS,
        title_prefix="Context DQN",
    )
    figures.extend(cancer_figures)

    with wandb.init(
        entity="seacts",
        project="seacts",
        name="results-visualizations",
        job_type="visualization",
    ) as run:
        run.log({f"figures/{figure.stem}": wandb.Image(str(figure)) for figure in figures})
    results_volume.commit()
    return [str(figure) for figure in figures]


@app.local_entrypoint(name="visualizations")
def main() -> None:
    for figure in plot_results_figures.remote():
        print(figure)
