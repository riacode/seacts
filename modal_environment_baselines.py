from __future__ import annotations

from src.modal_config import REMOTE_CONFIG_PATH, app, configured_image, data_volume, results_volume
from src.modal_config import wandb_secret


@app.function(
    image=configured_image,
    secrets=[wandb_secret],
    volumes={
        "/root/seacts/data": data_volume,
        "/root/seacts/results": results_volume,
    },
    timeout=7200,
)
def run_environment_baselines() -> list[dict[str, str | float]]:
    from src.environment_runner import run_environment_baseline_pipeline

    data_volume.reload()
    results, output_path = run_environment_baseline_pipeline(
        config_path=REMOTE_CONFIG_PATH,
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


@app.local_entrypoint(name="environment_baselines")
def main() -> None:
    for row in run_environment_baselines.remote():
        print(row)
