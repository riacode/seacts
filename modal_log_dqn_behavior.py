from __future__ import annotations

from pathlib import Path

from src.modal_config import REMOTE_CONFIG_PATH, app, configured_image, data_volume, results_volume
from src.modal_config import wandb_secret


ABLATION_VARIANT_OVERRIDES = {
    "scale1_long_mlp_1step": {
        "rl_training": {"q_network_type": "mlp", "n_step_returns": 1},
        "environment": {"selection_reward_scale": 1.0},
    },
    "scale1_long_mlp_3step": {
        "rl_training": {"q_network_type": "mlp", "n_step_returns": 3},
        "environment": {"selection_reward_scale": 1.0},
    },
    "scale1_long_structured_1step": {
        "rl_training": {"q_network_type": "structured", "n_step_returns": 1},
        "environment": {"selection_reward_scale": 1.0},
    },
    "scale1_long_structured_dueling_1step": {
        "rl_training": {"q_network_type": "structured_dueling", "n_step_returns": 1},
        "environment": {"selection_reward_scale": 1.0},
    },
    "scale1_long_structured_3step": {
        "rl_training": {"q_network_type": "structured", "n_step_returns": 3},
        "environment": {"selection_reward_scale": 1.0},
    },
    "scale1_long_structured_dueling_3step": {
        "rl_training": {"q_network_type": "structured_dueling", "n_step_returns": 3},
        "environment": {"selection_reward_scale": 1.0},
    },
}


@app.function(
    image=configured_image,
    volumes={
        "/root/seacts/data": data_volume,
        "/root/seacts/results": results_volume,
    },
    secrets=[wandb_secret],
    timeout=1800,
)
def log_dqn_behavior(variant: str = "all_ablation") -> list[dict[str, str]]:
    variants = list(ABLATION_VARIANT_OVERRIDES) if variant == "all_ablation" else [variant]
    return [_log_single_dqn_behavior(name) for name in variants]


def _log_single_dqn_behavior(variant: str) -> dict[str, str]:
    import tempfile

    import torch
    import yaml

    from src.config import load_baseline_config
    from src.dqn import build_q_network
    from src.rl_runner import build_dqn_eval_env, collect_dqn_behavior_log, load_rl_training_config

    config_path = REMOTE_CONFIG_PATH
    if variant != "default":
        if variant not in ABLATION_VARIANT_OVERRIDES:
            raise ValueError(
                f"Unknown DQN behavior variant '{variant}'. "
                f"Expected one of: {', '.join(ABLATION_VARIANT_OVERRIDES)}"
            )
        with Path(REMOTE_CONFIG_PATH).open("r", encoding="utf-8") as handle:
            config_raw = yaml.safe_load(handle)
        overrides = ABLATION_VARIANT_OVERRIDES[variant]
        config_raw["rl_training"] = dict(config_raw.get("rl_training", {}))
        config_raw["environment"] = dict(config_raw.get("environment", {}))
        config_raw["rl_training"].update(overrides.get("rl_training", {}))
        config_raw["environment"].update(overrides.get("environment", {}))
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            yaml.safe_dump(config_raw, handle)
            config_path = handle.name

    data_volume.reload()
    results_volume.reload()

    base_results_dir = Path("/root/seacts/results/depmap_baselines")
    output_dir = base_results_dir if variant == "default" else base_results_dir / "dqn_ablation" / variant
    model_path = output_dir / "dqn_policy.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing DQN checkpoint for {variant}: {model_path}")

    config = load_baseline_config(config_path)
    rl_config = load_rl_training_config(config_path)
    env, encoder, eval_episodes = build_dqn_eval_env(
        config,
        raw_data_dir="/root/seacts/data/raw",
        rl_config=rl_config,
    )

    q_network = build_q_network(
        encoder.state_size,
        encoder.action_space.size,
        rl_config.hidden_dim,
        network_type=rl_config.q_network_type,
        n_genes=encoder.n_genes,
        n_modalities=encoder.n_modalities,
    )
    q_network.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    q_network.eval()

    episodes_df, steps_df = collect_dqn_behavior_log(
        q_network=q_network,
        env=env,
        episodes=eval_episodes,
        encoder=encoder,
        top_k=config.evaluation.top_k,
        max_steps_per_episode=rl_config.max_steps_per_episode,
        min_queries_before_select=rl_config.min_queries_before_select,
    )

    episodes_path = output_dir / "dqn_episode_summary.csv"
    steps_path = output_dir / "dqn_step_log.csv"
    episodes_df.to_csv(episodes_path, index=False)
    steps_df.to_csv(steps_path, index=False)

    from src.behavior_analysis import write_behavior_analysis_tables

    analysis_paths = write_behavior_analysis_tables(
        episode_summary_path=episodes_path,
        step_log_path=steps_path,
        metadata_path="/root/seacts/data/raw/Model.csv",
        output_dir=output_dir / "behavior_analysis",
    )
    from src.visualization import generate_behavior_figures

    behavior_figures = generate_behavior_figures(
        episode_summary_path=episodes_path,
        step_log_path=steps_path,
        output_dir=output_dir / "behavior_figures",
        context_usage_path=_first_analysis_path(
            analysis_paths,
            "dqn_modality_usage_by_context",
        ),
    )

    if config.tracking.wandb.enabled:
        import pandas as pd
        import wandb

        from src.tracking import log_dqn_behavior_results

        analysis_tables = {
            path.stem: pd.read_csv(path)
            for path in analysis_paths
        }
        with wandb.init(
            entity=config.tracking.wandb.entity,
            project=config.tracking.wandb.project,
            name=f"dqn-behavior-{variant}",
            job_type="analysis",
        ) as run:
            run.config.update({"variant": variant}, allow_val_change=True)
            log_dqn_behavior_results(run, episodes_df, steps_df, analysis_tables)
            run.log(
                {
                    f"behavior_figures/{figure.stem}": wandb.Image(str(figure))
                    for figure in behavior_figures
                }
            )

    results_volume.commit()
    return {
        "variant": variant,
        "episodes": str(episodes_path),
        "steps": str(steps_path),
        "analysis": ", ".join(str(path) for path in analysis_paths),
        "figures": ", ".join(str(path) for path in behavior_figures),
    }


def _first_analysis_path(paths: list[Path], stem: str) -> Path | None:
    return next((path for path in paths if path.stem == stem), None)


@app.local_entrypoint(name="log-behavior")
def main(variant: str = "all_ablation") -> None:
    for result in log_dqn_behavior.remote(variant):
        print(result)
