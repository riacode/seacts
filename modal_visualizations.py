from __future__ import annotations

import modal


app = modal.App("seacts")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("src")
)

results_volume = modal.Volume.from_name("seacts-results", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/root/seacts/results": results_volume},
    secrets=[modal.Secret.from_name("wandb")],
    timeout=1800,
)
def plot_baseline_results() -> list[str]:
    import wandb

    from src.visualization import generate_baseline_figures

    results_volume.reload()
    figures = generate_baseline_figures(
        data_metrics_path="/root/seacts/results/depmap_baselines/data_baseline_metrics.csv",
        environment_metrics_path="/root/seacts/results/depmap_baselines/environment_baseline_metrics.csv",
        dqn_metrics_path="/root/seacts/results/depmap_baselines/dqn_eval_metrics.csv",
        dqn_trajectory_path="/root/seacts/results/depmap_baselines/dqn_trajectory_metrics.csv",
        output_dir="/root/seacts/results/figures",
    )
    with wandb.init(
        entity="seacts",
        project="seacts",
        name="baseline-visualizations",
        job_type="visualization",
    ) as run:
        run.log({figure.stem: wandb.Image(str(figure)) for figure in figures})
    results_volume.commit()
    return [str(figure) for figure in figures]


@app.local_entrypoint(name="visualizations")
def main() -> None:
    for figure in plot_baseline_results.remote():
        print(figure)
