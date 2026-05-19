from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve
import re

import pandas as pd


MANIFEST_URL = "https://depmap.org/portal/api/download/files"

REQUIRED_FILE_GROUPS = {
    "dependency": ("CRISPRGeneEffect.csv",),
    "metadata": ("Model.csv", "sample_info.csv"),
    "expression": (
        "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv",
        "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    ),
    "copy_number": (
        "PortalOmicsCNGeneLog2.csv",
        "OmicsCNGene.csv",
        "OmicsCNGeneWGS.csv",
        "OmicsCNGeneMC_WES.csv",
    ),
    "mutation": ("OmicsSomaticMutationsMatrixDamaging.csv",),
}

FILE_COLUMN_CANDIDATES = ("file", "filename", "file_name", "name")
RELEASE_COLUMN_CANDIDATES = ("release", "releasename", "release_name", "dataset", "taiga_id")
URL_COLUMN_CANDIDATES = ("url", "download_url", "downloadUrl", "link")


@dataclass(frozen=True)
class DepMapDownload:
    group: str
    file_name: str
    release: str
    destination: Path
    downloaded: bool


def load_manifest(manifest_url: str = MANIFEST_URL) -> pd.DataFrame:
    return pd.read_csv(manifest_url)


def download_selected_depmap_files(
    output_dir: str | Path,
    release: str = "latest",
    overwrite: bool = False,
    manifest_url: str = MANIFEST_URL,
) -> list[DepMapDownload]:
    output_path = Path(output_dir)
    raw_dir = output_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_url)
    file_col = _find_column(manifest, FILE_COLUMN_CANDIDATES)
    release_col = _find_column(manifest, RELEASE_COLUMN_CANDIDATES)
    url_col = _find_column(manifest, URL_COLUMN_CANDIDATES)

    selected_release = _select_release(manifest, release_col, release)
    release_manifest = manifest[manifest[release_col].astype(str) == selected_release].copy()

    downloads: list[DepMapDownload] = []
    selected_rows = []
    for group, candidate_names in REQUIRED_FILE_GROUPS.items():
        row = _find_file_row(release_manifest, file_col, candidate_names)
        file_name = str(row[file_col])
        destination = raw_dir / file_name
        downloaded = False
        if overwrite or not destination.exists():
            try:
                urlretrieve(str(row[url_col]), destination)
            except URLError as error:
                if destination.exists():
                    destination.unlink()
                raise RuntimeError(f"Failed to download {file_name} from DepMap manifest URL.") from error
            downloaded = True

        selected_rows.append(row)
        downloads.append(
            DepMapDownload(
                group=group,
                file_name=file_name,
                release=selected_release,
                destination=destination,
                downloaded=downloaded,
            )
        )

    pd.DataFrame(selected_rows).to_csv(output_path / "depmap_selected_manifest.csv", index=False)
    return downloads


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    normalized = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    raise ValueError(f"Could not find any of columns {candidates} in manifest: {list(frame.columns)}")


def _select_release(manifest: pd.DataFrame, release_col: str, release: str) -> str:
    releases = sorted(manifest[release_col].dropna().astype(str).unique())
    if release != "latest":
        matches = [candidate for candidate in releases if candidate == release]
        if not matches:
            raise ValueError(f"Release {release!r} not found. Available examples: {releases[:10]}")
        return matches[0]

    public_releases = [candidate for candidate in releases if "DepMap Public" in candidate]
    if not public_releases:
        raise ValueError("Could not find any DepMap Public releases in the download manifest.")
    return max(public_releases, key=_release_sort_key)


def _release_sort_key(release: str) -> tuple[int, int, str]:
    match = re.search(r"(\d{2})Q([1-4])", release)
    if match is None:
        return (0, 0, release)
    return (2000 + int(match.group(1)), int(match.group(2)), release)


def _find_file_row(
    manifest: pd.DataFrame,
    file_col: str,
    candidate_names: tuple[str, ...],
) -> pd.Series:
    file_names = manifest[file_col].astype(str)
    for candidate in candidate_names:
        exact = manifest[file_names == candidate]
        if not exact.empty:
            return exact.iloc[0]

    lowered = file_names.str.lower()
    for candidate in candidate_names:
        fuzzy = manifest[lowered == candidate.lower()]
        if not fuzzy.empty:
            return fuzzy.iloc[0]

    raise ValueError(
        f"Could not find any of {candidate_names} in selected release. "
        f"Available examples: {file_names.head(20).tolist()}"
    )
