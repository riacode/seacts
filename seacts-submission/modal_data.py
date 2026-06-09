from __future__ import annotations

from src.modal_config import app, data_volume, image


@app.function(image=image, volumes={"/root/seacts/data": data_volume}, timeout=3600)
def download_depmap_data(release: str = "latest", overwrite: bool = False) -> list[dict[str, str]]:
    from src.depmap_files import download_selected_depmap_files

    downloads = download_selected_depmap_files(
        output_dir="/root/seacts/data",
        release=release,
        overwrite=overwrite,
    )
    data_volume.commit()
    return [
        {
            "group": download.group,
            "file_name": download.file_name,
            "release": download.release,
            "destination": str(download.destination),
            "downloaded": str(download.downloaded),
        }
        for download in downloads
    ]


@app.local_entrypoint(name="download")
def main(release: str = "latest", overwrite: bool = False) -> None:
    for row in download_depmap_data.remote(release=release, overwrite=overwrite):
        print(row)
