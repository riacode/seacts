from __future__ import annotations

import modal


app = modal.App("seacts")

CONFIG_PATH = "configs/depmap_baselines.yaml"
REMOTE_CONFIG_PATH = "/root/seacts/configs/depmap_baselines.yaml"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("src")
)

configured_image = image.add_local_file(
    CONFIG_PATH,
    remote_path=REMOTE_CONFIG_PATH,
)

data_volume = modal.Volume.from_name("seacts-data", create_if_missing=True)
results_volume = modal.Volume.from_name("seacts-results", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb")
