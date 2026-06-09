from __future__ import annotations

from pathlib import Path

from src.modal_config import REMOTE_CONFIG_PATH, app, configured_image, data_volume, results_volume
from src.modal_config import wandb_secret

behavior_image = configured_image.add_local_file(
    "modal_sweep_context_dqn.py",
    remote_path="/root/seacts/modal_sweep_context_dqn.py",
)


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

SWEEP_VARIANT_OVERRIDES = {
    "best_structured_1step_larger": {
        "rl_training": {
            "q_network_type": "structured",
            "n_step_returns": 1,
            "hidden_dim": 256,
            "min_queries_before_select": 8,
            "train_episodes": 20_000,
            "replay_capacity": 50_000,
            "epsilon_decay_steps": 50_000,
            "validation_interval": 250,
        },
        "environment": {"selection_reward_scale": 1.0},
    },
}

CANCER_CONTEXT_VARIANT_OVERRIDES = {
    "default": {
        "rl_training": {
            "q_network_type": "context_structured",
            "n_step_returns": 1,
            "hidden_dim": 128,
            "min_queries_before_select": 8,
            "cancer_context_column": "OncotreeLineage",
            "cancer_context_dim": 16,
        },
        "environment": {"selection_reward_scale": 1.0},
    },
    "larger": {
        "rl_training": {
            "q_network_type": "context_structured",
            "n_step_returns": 1,
            "hidden_dim": 256,
            "min_queries_before_select": 8,
            "cancer_context_column": "OncotreeLineage",
            "cancer_context_dim": 16,
        },
        "environment": {"selection_reward_scale": 1.0},
    },
}


@app.function(
    image=behavior_image,
    volumes={
        "/root/seacts/data": data_volume,
        "/root/seacts/results": results_volume,
    },
    secrets=[wandb_secret],
    timeout=1800,
)
def log_dqn_behavior(variant: str = "all_ablation") -> list[dict[str, str]]:
    if variant == "all_ablation":
        variants = list(ABLATION_VARIANT_OVERRIDES)
    elif variant == "all_cancer_context":
        variants = [f"cancer-context_{name}" for name in CANCER_CONTEXT_VARIANT_OVERRIDES]
    else:
        variants = [variant]
    return [_log_single_dqn_behavior(name) for name in variants]


def _log_single_dqn_behavior(variant: str) -> dict[str, str]:
    import tempfile

    import torch
    import yaml

    from src.config import load_baseline_config
    from src.dqn import build_q_network
    from src.rl_runner import (
        _hyperparameters_with_context,
        build_dqn_eval_env,
        collect_dqn_behavior_log,
        load_rl_training_config,
    )

    config_path = REMOTE_CONFIG_PATH
    if variant != "default":
        overrides, variant_key = _resolve_variant_overrides(variant)
        with Path(REMOTE_CONFIG_PATH).open("r", encoding="utf-8") as handle:
            config_raw = yaml.safe_load(handle)
        config_raw["rl_training"] = dict(config_raw.get("rl_training", {}))
        config_raw["environment"] = dict(config_raw.get("environment", {}))
        config_raw["rl_training"].update(overrides.get("rl_training", {}))
        config_raw["environment"].update(overrides.get("environment", {}))
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            yaml.safe_dump(config_raw, handle)
            config_path = handle.name
    else:
        variant_key = "default"

    data_volume.reload()
    results_volume.reload()

    base_results_dir = Path("/root/seacts/results/depmap_baselines")
    output_dir = _variant_output_dir(base_results_dir, variant, variant_key)
    model_path = output_dir / "dqn_policy.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing DQN checkpoint for {variant}: {model_path}")

    config = load_baseline_config(config_path)
    rl_config = load_rl_training_config(config_path)
    env, encoder, eval_episodes, context_encoder = build_dqn_eval_env(
        config,
        raw_data_dir="/root/seacts/data/raw",
        rl_config=rl_config,
    )
    hyperparameters = _hyperparameters_with_context(
        rl_config.hyperparameters,
        encoder,
        context_encoder,
    )
    q_network = build_q_network(
        encoder.state_size,
        encoder.action_space.size,
        hyperparameters.hidden_dim,
        network_type=hyperparameters.q_network_type,
        n_genes=hyperparameters.n_genes,
        n_modalities=hyperparameters.n_modalities,
        n_lineages=hyperparameters.n_lineages,
        cancer_context_dim=hyperparameters.cancer_context_dim,
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
        context_encoder=context_encoder,
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


def _resolve_variant_overrides(variant: str) -> tuple[dict, str]:
    import sys

    if "/root/seacts" not in sys.path:
        sys.path.insert(0, "/root/seacts")  # remote import path
    from modal_sweep_context_dqn import CONTEXT_SWEEP_VARIANTS, _resolve_structured_checkpoint

    if variant in CONTEXT_SWEEP_VARIANTS:
        overrides = CONTEXT_SWEEP_VARIANTS[variant]
        rl_training = _resolve_structured_checkpoint(dict(overrides.get("rl_training", {})))
        return (
            {
                "rl_training": rl_training,
                "environment": dict(overrides.get("environment", {})),
            },
            variant,
        )
    if variant in ABLATION_VARIANT_OVERRIDES:
        return ABLATION_VARIANT_OVERRIDES[variant], variant
    if variant in SWEEP_VARIANT_OVERRIDES:
        return SWEEP_VARIANT_OVERRIDES[variant], variant
    if variant.startswith("cancer-context_"):
        variant_key = variant.removeprefix("cancer-context_")
        if variant_key not in CANCER_CONTEXT_VARIANT_OVERRIDES:
            raise ValueError(f"Unknown cancer-context variant '{variant_key}'.")
        return CANCER_CONTEXT_VARIANT_OVERRIDES[variant_key], variant_key
    raise ValueError(
        f"Unknown DQN behavior variant '{variant}'. "
        f"Expected an ablation name, sweep variant (e.g. best_structured_1step_larger), "
        f"context sweep name (e.g. ctx_larger_init_structured), "
        f"cancer-context_<name>, all_ablation, or all_cancer_context."
    )


def _variant_output_dir(base_results_dir: Path, variant: str, variant_key: str) -> Path:
    if variant == "default":
        return base_results_dir
    if variant.startswith("cancer-context_"):
        return base_results_dir / "dqn_cancer_context" / variant_key
    try:
        from modal_sweep_context_dqn import CONTEXT_SWEEP_VARIANTS

        if variant in CONTEXT_SWEEP_VARIANTS or variant_key in CONTEXT_SWEEP_VARIANTS:
            return base_results_dir / "dqn_context_sweeps" / variant_key
    except ImportError:
        pass
    if variant_key in SWEEP_VARIANT_OVERRIDES or variant in SWEEP_VARIANT_OVERRIDES:
        for candidate in (
            base_results_dir / variant_key,
            base_results_dir / "dqn_sweeps" / variant_key,
        ):
            if (candidate / "dqn_policy.pt").exists():  # probe result layout
                return candidate
        return base_results_dir / variant_key
    return base_results_dir / "dqn_ablation" / variant_key


def _first_analysis_path(paths: list[Path], stem: str) -> Path | None:
    return next((path for path in paths if path.stem == stem), None)


@app.local_entrypoint(name="log-behavior")
def main(variant: str = "all_ablation") -> None:
    for result in log_dqn_behavior.remote(variant):
        print(result)
