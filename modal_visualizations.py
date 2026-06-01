from __future__ import annotations

from src.modal_config import app, image, results_volume, wandb_secret


@app.function(
    image=image,
    volumes={"/root/seacts/results": results_volume},
    secrets=[wandb_secret],
    timeout=1800,
)
def plot_results_figures() -> list[str]:
    import wandb

    from src.visualization import generate_ablation_figures, generate_baseline_figures

    results_volume.reload()
    figures = generate_baseline_figures(
        data_metrics_path="/root/seacts/results/depmap_baselines/data_baseline_metrics.csv",
        environment_metrics_path="/root/seacts/results/depmap_baselines/environment_baseline_metrics.csv",
        dqn_metrics_path=(
            "/root/seacts/results/depmap_baselines/dqn_sweeps/"
            "best_structured_1step_larger/dqn_eval_metrics.csv"
        ),
        dqn_trajectory_path=(
            "/root/seacts/results/depmap_baselines/dqn_sweeps/"
            "best_structured_1step_larger/dqn_trajectory_metrics.csv"
        ),
        output_dir="/root/seacts/results/figures",
    )
    ablation_figures = generate_ablation_figures(
        ablation_summary_path="/root/seacts/results/depmap_baselines/dqn_ablation/ablation_summary.csv",
        output_dir="/root/seacts/results/figures/dqn_ablation",
    )
    figures.extend(ablation_figures)
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
