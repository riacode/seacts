from __future__ import annotations

import modal


app = modal.App("seacts")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_file(
        "configs/depmap_baselines.yaml",
        remote_path="/root/seacts/configs/depmap_baselines.yaml",
    )
    .add_local_python_source("src")
)

data_volume = modal.Volume.from_name("seacts-data", create_if_missing=True)
results_volume = modal.Volume.from_name("seacts-results", create_if_missing=True)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("wandb")],
    volumes={
        "/root/seacts/data": data_volume,
        "/root/seacts/results": results_volume,
    },
    timeout=7200,
)
def run_depmap_baselines() -> list[dict[str, str | float]]:
    from src.baseline_runner import run_baseline_pipeline

    data_volume.reload()
    results, output_path = run_baseline_pipeline(
        config_path="/root/seacts/configs/depmap_baselines.yaml",
        raw_data_dir="/root/seacts/data/raw",
        output_dir="/root/seacts/results/depmap_baselines",
    )
    results_volume.commit()
    rows = results.to_dict(orient="records")
    return [
        {
            key: value.item() if hasattr(value, "item") else value
            for key, value in row.items()
        }
        for row in rows
    ] + [{"output_path": str(output_path)}]


@app.local_entrypoint(name="baselines")
def main() -> None:
    for row in run_depmap_baselines.remote():
        print(row)
