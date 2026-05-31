from __future__ import annotations

import modal


app = modal.App("seacts")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("src")
)

data_volume = modal.Volume.from_name("seacts-data", create_if_missing=True)
results_volume = modal.Volume.from_name("seacts-results", create_if_missing=True)
