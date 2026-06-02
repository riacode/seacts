from __future__ import annotations

from src.modal_config import REMOTE_CONFIG_PATH, app, configured_image, data_volume, results_volume
from src.modal_config import wandb_secret

# Structured sweep winner on held-out cell lines (same training budget as modal_sweep_dqn.py).
STRUCTURED_DQN_TARGET_TOTAL_REWARD = 1.035

STRUCTURED_CHECKPOINT = (
    "/root/seacts/results/depmap_baselines/dqn_sweeps/best_structured_1step_larger/dqn_policy.pt"
)
STRUCTURED_CHECKPOINT_FALLBACK = (
    "/root/seacts/results/depmap_baselines/best_structured_1step_larger/dqn_policy.pt"
)
SWEEP_ROOT = "/root/seacts/results/depmap_baselines/dqn_context_sweeps"

# Matches structured DQN / modal_ablate_dqn.py: 20k episodes, CPU, one Modal container for all variants.
STRUCTURED_MATCHED_TRAINING = {
    "train_episodes": 20_000,
    "replay_capacity": 50_000,
    "epsilon_decay_steps": 50_000,
    "validation_interval": 250,
    "expert_seed_episodes": 0,
    "min_queries_before_select": 8,
    "n_step_returns": 1,
}

BASE_CONTEXT_OVERRIDES = {
    **STRUCTURED_MATCHED_TRAINING,
    "q_network_type": "context_fusion_structured",
    "cancer_context_column": "OncotreeLineage",
    "cancer_context_dim": 32,
    "hidden_dim": 256,
    "use_lineage_modality_scores": True,
    "lineage_min_samples": 6,
    "query_shaping_alpha": 0.05,
    "fusion_query_boost": 2.0,
    "fusion_select_weight": 1.0,
    "select_residual_weight": 0.25,
}

LEGACY_CONTEXT_OVERRIDES = {
    **STRUCTURED_MATCHED_TRAINING,
    "q_network_type": "context_structured",
    "cancer_context_column": "OncotreeLineage",
    "cancer_context_dim": 16,
    "hidden_dim": 256,
}

CONTEXT_SWEEP_VARIANTS = {
    "ctx_fusion_lineage_init_structured": {
        "rl_training": {
            **BASE_CONTEXT_OVERRIDES,
            "init_structured_checkpoint": STRUCTURED_CHECKPOINT,
        },
        "environment": {"selection_reward_scale": 1.0},
    },
    "ctx_fusion_lineage_init_frozen": {
        "rl_training": {
            **BASE_CONTEXT_OVERRIDES,
            "init_structured_checkpoint": STRUCTURED_CHECKPOINT,
            "freeze_shared_heads": True,
        },
        "environment": {"selection_reward_scale": 1.0},
    },
    "ctx_fusion_lineage_shaping_only": {
        "rl_training": {
            **BASE_CONTEXT_OVERRIDES,
            "init_structured_checkpoint": STRUCTURED_CHECKPOINT,
            "query_shaping_alpha": 0.08,
        },
        "environment": {"selection_reward_scale": 1.0},
    },
    "ctx_baseline_larger": {
        "rl_training": dict(LEGACY_CONTEXT_OVERRIDES),
        "environment": {"selection_reward_scale": 1.0},
    },
    "ctx_larger_ctx32": {
        "rl_training": {
            **LEGACY_CONTEXT_OVERRIDES,
            "cancer_context_dim": 32,
        },
        "environment": {"selection_reward_scale": 1.0},
    },
    "ctx_larger_ctx64": {
        "rl_training": {
            **LEGACY_CONTEXT_OVERRIDES,
            "cancer_context_dim": 64,
        },
        "environment": {"selection_reward_scale": 1.0},
    },
    "ctx_larger_dueling": {
        "rl_training": {
            **LEGACY_CONTEXT_OVERRIDES,
            "q_network_type": "context_structured_dueling",
        },
        "environment": {"selection_reward_scale": 1.0},
    },
    "ctx_larger_lower_lr": {
        "rl_training": {
            **LEGACY_CONTEXT_OVERRIDES,
            "learning_rate": 3e-5,
            "epsilon_decay_steps": 75_000,
        },
        "environment": {"selection_reward_scale": 1.0},
    },
    "ctx_larger_min_queries_6": {
        "rl_training": {
            **LEGACY_CONTEXT_OVERRIDES,
            "min_queries_before_select": 6,
        },
        "environment": {"selection_reward_scale": 1.0},
    },
    "ctx_larger_init_structured": {
        "rl_training": {
            **LEGACY_CONTEXT_OVERRIDES,
            "init_structured_checkpoint": STRUCTURED_CHECKPOINT,
        },
        "environment": {"selection_reward_scale": 1.0},
    },
    # Lineage on SELECT only (query path matches Structured); best shot at beating 1.035.
    "ctx_select_only_init_structured": {
        "rl_training": {
            **LEGACY_CONTEXT_OVERRIDES,
            "q_network_type": "context_select_structured",
            "init_structured_checkpoint": STRUCTURED_CHECKPOINT,
        },
        "environment": {"selection_reward_scale": 1.0},
    },
    "ctx_select_only_init_frozen": {
        "rl_training": {
            **LEGACY_CONTEXT_OVERRIDES,
            "q_network_type": "context_select_structured",
            "init_structured_checkpoint": STRUCTURED_CHECKPOINT,
            "freeze_shared_heads": True,
        },
        "environment": {"selection_reward_scale": 1.0},
    },
    "ctx_init_structured_30k": {
        "rl_training": {
            **LEGACY_CONTEXT_OVERRIDES,
            "init_structured_checkpoint": STRUCTURED_CHECKPOINT,
            "train_episodes": 30_000,
            "epsilon_decay_steps": 75_000,
        },
        "environment": {"selection_reward_scale": 1.0},
    },
    "ctx_init_structured_ft_lr2e5": {
        "rl_training": {
            **LEGACY_CONTEXT_OVERRIDES,
            "init_structured_checkpoint": STRUCTURED_CHECKPOINT,
            "learning_rate": 2e-5,
        },
        "environment": {"selection_reward_scale": 1.0},
    },
    # Cancer info in the environment: per-lineage Ridge query returns (no fusion / shaping).
    "struct_lineage_env_scores": {
        "rl_training": {
            **STRUCTURED_MATCHED_TRAINING,
            "q_network_type": "structured",
            "hidden_dim": 256,
            "cancer_context_column": "OncotreeLineage",
            "use_lineage_modality_scores": True,
            "lineage_min_samples": 6,
        },
        "environment": {"selection_reward_scale": 1.0},
    },
    "ctx_lineage_env_init_structured": {
        "rl_training": {
            **LEGACY_CONTEXT_OVERRIDES,
            "init_structured_checkpoint": STRUCTURED_CHECKPOINT,
            "use_lineage_modality_scores": True,
            "lineage_min_samples": 6,
        },
        "environment": {"selection_reward_scale": 1.0},
    },
    "ctx_larger_init_frozen": {
        "rl_training": {
            **LEGACY_CONTEXT_OVERRIDES,
            "init_structured_checkpoint": STRUCTURED_CHECKPOINT,
            "freeze_shared_heads": True,
        },
        "environment": {"selection_reward_scale": 1.0},
    },
    "ctx_init_structured_3step": {
        "rl_training": {
            **LEGACY_CONTEXT_OVERRIDES,
            "init_structured_checkpoint": STRUCTURED_CHECKPOINT,
            "n_step_returns": 3,
        },
        "environment": {"selection_reward_scale": 1.0},
    },
    "ctx_larger_init_structured_ctx32": {
        "rl_training": {
            **LEGACY_CONTEXT_OVERRIDES,
            "cancer_context_dim": 32,
            "init_structured_checkpoint": STRUCTURED_CHECKPOINT,
        },
        "environment": {"selection_reward_scale": 1.0},
    },
    "ctx_larger_init_structured_min_queries_6": {
        "rl_training": {
            **LEGACY_CONTEXT_OVERRIDES,
            "init_structured_checkpoint": STRUCTURED_CHECKPOINT,
            "min_queries_before_select": 6,
        },
        "environment": {"selection_reward_scale": 1.0},
    },
    "ctx_larger_init_structured_ctx32_min_queries_6": {
        "rl_training": {
            **LEGACY_CONTEXT_OVERRIDES,
            "cancer_context_dim": 32,
            "init_structured_checkpoint": STRUCTURED_CHECKPOINT,
            "min_queries_before_select": 6,
        },
        "environment": {"selection_reward_scale": 1.0},
    },
}


def _resolve_structured_checkpoint(rl_overrides: dict[str, object]) -> dict[str, object]:
    from pathlib import Path

    resolved = dict(rl_overrides)
    if resolved.get("init_structured_checkpoint") == STRUCTURED_CHECKPOINT:
        if not Path(STRUCTURED_CHECKPOINT).exists() and Path(STRUCTURED_CHECKPOINT_FALLBACK).exists():
            resolved["init_structured_checkpoint"] = STRUCTURED_CHECKPOINT_FALLBACK
    return resolved


@app.function(
    image=configured_image,
    volumes={
        "/root/seacts/data": data_volume,
        "/root/seacts/results": results_volume,
    },
    secrets=[wandb_secret],
    timeout=21600,
)
def sweep_context_dqn(variant_filter: str = "") -> list[dict[str, float | str]]:
    """Train Context DQN ablations in one container (same pattern as modal_ablate_dqn.py)."""
    from pathlib import Path
    import tempfile

    import pandas as pd
    import yaml

    from src.rl_runner import run_dqn_training_pipeline

    if variant_filter and variant_filter not in CONTEXT_SWEEP_VARIANTS:
        raise ValueError(
            f"Unknown variant {variant_filter!r}. "
            f"Choose from: {', '.join(CONTEXT_SWEEP_VARIANTS)}"
        )

    variants = (
        {variant_filter: CONTEXT_SWEEP_VARIANTS[variant_filter]}
        if variant_filter
        else CONTEXT_SWEEP_VARIANTS
    )

    data_volume.reload()
    results_volume.reload()
    with Path(REMOTE_CONFIG_PATH).open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)

    sweep_root = Path(SWEEP_ROOT)
    summary_path = sweep_root / "sweep_summary.csv"
    rows: list[dict[str, float | str]] = []

    for variant_name, overrides in variants.items():
        output_dir = sweep_root / variant_name
        output_path = output_dir / "dqn_eval_metrics.csv"
        if output_path.exists():
            row = pd.read_csv(output_path).iloc[0].to_dict()
            row["variant"] = variant_name
            row["output_path"] = str(output_path)
            row["status"] = "skipped_existing"
            rows.append(_jsonable(row))
            print(f"[sweep] {variant_name}: skipped (existing)", flush=True)
            continue

        print(f"[sweep] {variant_name}: training ...", flush=True)
        try:
            config = dict(base_config)
            config["rl_training"] = dict(base_config.get("rl_training", {}))
            config["environment"] = dict(base_config.get("environment", {}))
            rl_overrides = _resolve_structured_checkpoint(dict(overrides.get("rl_training", {})))
            config["rl_training"].update(rl_overrides)
            config["environment"].update(overrides.get("environment", {}))

            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
                yaml.safe_dump(config, handle)
                variant_config_path = handle.name

            results, saved_path = run_dqn_training_pipeline(
                config_path=variant_config_path,
                raw_data_dir="/root/seacts/data/raw",
                output_dir=output_dir,
                wandb_run_name=f"dqn-context-sweep-{variant_name}",
            )
            row = results.iloc[0].to_dict()
            row["variant"] = variant_name
            row["output_path"] = str(saved_path)
            row["status"] = "trained"
            rows.append(_jsonable(row))
            print(
                f"[sweep] {variant_name}: done total_reward={row.get('total_reward')}",
                flush=True,
            )
        except Exception as error:
            print(f"[sweep] {variant_name}: FAILED — {error}", flush=True)
            rows.append(
                {
                    "variant": variant_name,
                    "output_path": str(output_path),
                    "status": "failed",
                    "error": str(error),
                }
            )

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        _write_sweep_summary(rows, summary_path)
        results_volume.commit()

    if rows:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        _write_sweep_summary(rows, summary_path)
        results_volume.commit()
    return rows + [{"output_path": str(summary_path)}]


def _write_sweep_summary(rows: list[dict[str, float | str]], summary_path: Path) -> None:
    import pandas as pd

    frame = pd.DataFrame(rows)
    if "total_reward" in frame.columns:
        frame = frame.sort_values("total_reward", ascending=False, na_position="last")
    frame.to_csv(summary_path, index=False)


def _jsonable(row: dict) -> dict[str, float | str]:
    return {
        key: value.item() if hasattr(value, "item") else value
        for key, value in row.items()
    }


def _print_sweep_row(row: dict[str, float | str]) -> None:
    name = row.get("variant", row.get("output_path", "?"))
    reward = row.get("total_reward", "")
    status = row.get("status", "")
    beat = (
        isinstance(reward, (int, float))
        and float(reward) > STRUCTURED_DQN_TARGET_TOTAL_REWARD
    )
    flag = " (beats structured)" if beat else ""
    suffix = f" {status}" if status else ""
    line = f"{name}: total_reward={reward}{flag}{suffix}" if reward != "" else str(row)
    print(line, flush=True)


@app.local_entrypoint(name="sweep-context")
def main(variant: str = "") -> None:
    """Run Context DQN ablations (20k episodes each, CPU, one Modal job).

    **Full sweep must use detach** (otherwise Modal tears down the app when the
    local CLI exits and the job stops early):

        modal run --detach modal_sweep_context_dqn.py

    Single variant (~8–12 min) can run without detach:

        modal run modal_sweep_context_dqn.py --variant ctx_fusion_lineage_init_structured

    Target to beat: Structured DQN total_reward ~= 1.035.
    """
    if variant and variant not in CONTEXT_SWEEP_VARIANTS:
        raise SystemExit(
            f"Unknown variant {variant!r}. Options: {', '.join(CONTEXT_SWEEP_VARIANTS)}"
        )

    n_variants = 1 if variant else len(CONTEXT_SWEEP_VARIANTS)
    print(
        f"Context DQN sweep: {n_variants} variant(s), 20k episodes, CPU, "
        f"target total_reward > {STRUCTURED_DQN_TARGET_TOTAL_REWARD}",
        flush=True,
    )
    if not variant:
        print(
            "\nFull sweep: use  modal run --detach modal_sweep_context_dqn.py\n"
            "(without --detach the remote job often stops when this terminal exits)\n",
            flush=True,
        )

    for row in sweep_context_dqn.remote(variant):
        _print_sweep_row(row)
