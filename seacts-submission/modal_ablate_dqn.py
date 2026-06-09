from __future__ import annotations

from src.modal_config import REMOTE_CONFIG_PATH, app, configured_image, data_volume, results_volume
from src.modal_config import wandb_secret


LONG_TRAINING_OVERRIDES = {
    "train_episodes": 20_000,
    "replay_capacity": 50_000,
    "epsilon_decay_steps": 50_000,
    "validation_interval": 250,
}


ABLATION_VARIANTS = {
    "scale1_long_mlp_1step": {
        "rl_training": {
            **LONG_TRAINING_OVERRIDES,
            "q_network_type": "mlp",
            "n_step_returns": 1,
        },
        "environment": {
            "selection_reward_scale": 1.0,
        },
    },
    "scale1_long_mlp_3step": {
        "rl_training": {
            **LONG_TRAINING_OVERRIDES,
            "q_network_type": "mlp",
            "n_step_returns": 3,
        },
        "environment": {
            "selection_reward_scale": 1.0,
        },
    },
    "scale1_long_structured_1step": {
        "rl_training": {
            **LONG_TRAINING_OVERRIDES,
            "q_network_type": "structured",
            "n_step_returns": 1,
        },
        "environment": {
            "selection_reward_scale": 1.0,
        },
    },
    "scale1_long_structured_dueling_1step": {
        "rl_training": {
            **LONG_TRAINING_OVERRIDES,
            "q_network_type": "structured_dueling",
            "n_step_returns": 1,
        },
        "environment": {
            "selection_reward_scale": 1.0,
        },
    },
    "scale1_long_structured_3step": {
        "rl_training": {
            **LONG_TRAINING_OVERRIDES,
            "q_network_type": "structured",
            "n_step_returns": 3,
        },
        "environment": {
            "selection_reward_scale": 1.0,
        },
    },
    "scale1_long_structured_dueling_3step": {
        "rl_training": {
            **LONG_TRAINING_OVERRIDES,
            "q_network_type": "structured_dueling",
            "n_step_returns": 3,
        },
        "environment": {
            "selection_reward_scale": 1.0,
        },
    },
}


@app.function(
    image=configured_image,
    volumes={
        "/root/seacts/data": data_volume,
        "/root/seacts/results": results_volume,
    },
    secrets=[wandb_secret],
    timeout=21600,
)
def ablate_dqn() -> list[dict[str, float | str]]:
    from pathlib import Path
    import tempfile

    import pandas as pd
    import yaml

    from src.rl_runner import run_dqn_training_pipeline

    data_volume.reload()
    results_volume.reload()
    with Path(REMOTE_CONFIG_PATH).open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)

    rows: list[dict[str, float | str]] = []
    summary_path = Path("/root/seacts/results/depmap_baselines/dqn_ablation/ablation_summary.csv")
    for variant_name, overrides in ABLATION_VARIANTS.items():
        output_dir = Path("/root/seacts/results/depmap_baselines/dqn_ablation") / variant_name
        output_path = output_dir / "dqn_eval_metrics.csv"
        if output_path.exists():
            results = pd.read_csv(output_path)
            row = results.iloc[0].to_dict()
            row["variant"] = variant_name
            row["output_path"] = str(output_path)
            rows.append(_jsonable(row))
            continue

        config = dict(base_config)
        config["rl_training"] = dict(base_config.get("rl_training", {}))
        config["environment"] = dict(base_config.get("environment", {}))
        config["rl_training"].update(overrides.get("rl_training", {}))
        config["environment"].update(overrides.get("environment", {}))
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            yaml.safe_dump(config, handle)
            variant_config_path = handle.name

        results, output_path = run_dqn_training_pipeline(
            config_path=variant_config_path,
            raw_data_dir="/root/seacts/data/raw",
            output_dir=output_dir,
            wandb_run_name=f"dqn-ablation-{variant_name}",
        )
        row = results.iloc[0].to_dict()
        row["variant"] = variant_name
        row["output_path"] = str(output_path)
        rows.append(_jsonable(row))

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).sort_values("total_reward", ascending=False).to_csv(summary_path, index=False)
        results_volume.commit()

    if rows:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).sort_values("total_reward", ascending=False).to_csv(summary_path, index=False)
        results_volume.commit()
    return rows + [{"output_path": str(summary_path)}]


def _jsonable(row: dict) -> dict[str, float | str]:
    return {
        key: value.item() if hasattr(value, "item") else value
        for key, value in row.items()
    }


@app.local_entrypoint(name="ablate")
def main() -> None:
    for row in ablate_dqn.remote():
        print(row)
