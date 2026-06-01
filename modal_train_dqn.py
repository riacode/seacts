from __future__ import annotations

from src.modal_config import REMOTE_CONFIG_PATH, app, configured_image, data_volume, results_volume
from src.modal_config import wandb_secret


@app.function(
    image=configured_image,
    volumes={
        "/root/seacts/data": data_volume,
        "/root/seacts/results": results_volume,
    },
    secrets=[wandb_secret],
    timeout=7200,
)
def train_dqn() -> list[dict[str, float | str]]:
    from src.rl_runner import run_dqn_training_pipeline

    data_volume.reload()
    results_volume.reload()
    results, output_path = run_dqn_training_pipeline(
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


@app.local_entrypoint(name="train")
def main() -> None:
    for row in train_dqn.remote():
        print(row)
