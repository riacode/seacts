from __future__ import annotations

from src.modal_config import REMOTE_CONFIG_PATH, app, configured_image, data_volume, results_volume
from src.modal_config import wandb_secret


LONG_TRAINING_OVERRIDES = {
    "train_episodes": 20_000,
    "replay_capacity": 50_000,
    "epsilon_decay_steps": 50_000,
    "validation_interval": 250,
    "min_queries_before_select": 8,
    "n_step_returns": 1,
    "q_network_type": "context_structured",
    "cancer_context_column": "OncotreeLineage",
    "cancer_context_dim": 32,
}

CANCER_CONTEXT_VARIANTS = {
    "default": {
        "rl_training": {
            **LONG_TRAINING_OVERRIDES,
            "hidden_dim": 128,
        },
        "environment": {
            "selection_reward_scale": 1.0,
        },
    },
    "larger": {
        "rl_training": {
            **LONG_TRAINING_OVERRIDES,
            "hidden_dim": 256,
            "cancer_context_dim": 32,
        },
        "environment": {
            "selection_reward_scale": 1.0,
        },
    },
    "larger_init_structured": {
        "rl_training": {
            **LONG_TRAINING_OVERRIDES,
            "hidden_dim": 256,
            "cancer_context_dim": 32,
            "init_structured_checkpoint": (
                "/root/seacts/results/depmap_baselines/dqn_sweeps/"
                "best_structured_1step_larger/dqn_policy.pt"
            ),
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
def train_cancer_context_dqn() -> list[dict[str, float | str]]:
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
    summary_path = Path(
        "/root/seacts/results/depmap_baselines/dqn_cancer_context/ablation_summary.csv"
    )
    for variant_name, overrides in CANCER_CONTEXT_VARIANTS.items():
        output_dir = Path("/root/seacts/results/depmap_baselines/dqn_cancer_context") / variant_name
        output_path = output_dir / "dqn_eval_metrics.csv"
        if output_path.exists():
            results = pd.read_csv(output_path)
            row = results.iloc[0].to_dict()
            row["variant"] = f"cancer-context_{variant_name}"
            row["output_path"] = str(output_path)
            rows.append(_jsonable(row))
            continue

        config = dict(base_config)
        config["rl_training"] = dict(base_config.get("rl_training", {}))
        config["environment"] = dict(base_config.get("environment", {}))
        rl_overrides = dict(overrides.get("rl_training", {}))
        init_path = rl_overrides.get("init_structured_checkpoint")
        if init_path:
            primary = Path(str(init_path))
            fallback = Path(
                "/root/seacts/results/depmap_baselines/best_structured_1step_larger/dqn_policy.pt"
            )
            if not primary.exists() and fallback.exists():
                rl_overrides["init_structured_checkpoint"] = str(fallback)
        config["rl_training"].update(rl_overrides)
        config["environment"].update(overrides.get("environment", {}))
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            yaml.safe_dump(config, handle)
            variant_config_path = handle.name

        results, output_path = run_dqn_training_pipeline(
            config_path=variant_config_path,
            raw_data_dir="/root/seacts/data/raw",
            output_dir=output_dir,
            wandb_run_name=f"dqn-cancer-context-{variant_name}",
        )
        row = results.iloc[0].to_dict()
        row["variant"] = f"cancer-context_{variant_name}"
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


@app.local_entrypoint(name="train-cancer-context")
def main() -> None:
    for row in train_cancer_context_dqn.remote():
        print(row)
