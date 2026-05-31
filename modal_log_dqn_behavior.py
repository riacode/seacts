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
    volumes={
        "/root/seacts/data": data_volume,
        "/root/seacts/results": results_volume,
    },
    secrets=[modal.Secret.from_name("wandb")],
    timeout=1800,
)
def log_dqn_behavior() -> dict[str, str]:
    from pathlib import Path

    import torch

    from src.config import load_baseline_config
    from src.dqn import build_q_network
    from src.rl_runner import build_dqn_eval_env, collect_dqn_behavior_log, load_rl_training_config

    config_path = "/root/seacts/configs/depmap_baselines.yaml"
    output_dir = Path("/root/seacts/results/depmap_baselines")
    model_path = output_dir / "dqn_policy.pt"

    data_volume.reload()
    results_volume.reload()

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
    )

    episodes_path = output_dir / "dqn_episode_summary.csv"
    steps_path = output_dir / "dqn_step_log.csv"
    episodes_df.to_csv(episodes_path, index=False)
    steps_df.to_csv(steps_path, index=False)

    if config.tracking.wandb.enabled:
        import wandb

        from src.tracking import log_dqn_behavior_results

        with wandb.init(
            entity=config.tracking.wandb.entity,
            project=config.tracking.wandb.project,
            name="dqn-behavior",
            job_type="analysis",
        ) as run:
            log_dqn_behavior_results(run, episodes_df, steps_df)

    results_volume.commit()
    return {"episodes": str(episodes_path), "steps": str(steps_path)}


@app.local_entrypoint(name="log-behavior")
def main() -> None:
    print(log_dqn_behavior.remote())
