"""Download and verify the Altahullion turbine-data archive from Zenodo."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
RECORD_ID = "19948235"
ARCHIVE_NAME = "turbine_data.zip"
ARCHIVE_URL = (
    "https://zenodo.org/api/records/%s/files/%s/content"
    % (RECORD_ID, ARCHIVE_NAME)
)
EXPECTED_MD5 = "16c7c2e043590444d6584f7f27d6c967"


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the hexadecimal MD5 checksum of *path*."""

    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract a ZIP archive while rejecting path-traversal entries."""

    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if os.path.commonpath([str(destination), str(target)]) != str(destination):
                raise ValueError("unsafe ZIP member: %s" % member.filename)
        handle.extractall(destination)


def download(url: str, target: Path) -> None:
    """Download *url* to *target* via a temporary partial file."""

    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "censoring-aware-wind-power/0.1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, partial.open(
        "wb"
    ) as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    partial.replace(target)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_DIR / "data",
        help="directory in which the archive is extracted",
    )
    parser.add_argument("--force", action="store_true", help="redownload the archive")
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="retain the downloaded ZIP after successful extraction",
    )
    return parser


def main() -> int:
    """Download, validate, and extract the public turbine data."""

    arguments = build_parser().parse_args()
    data_dir = arguments.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / ARCHIVE_NAME

    if arguments.force or not archive.exists():
        download(ARCHIVE_URL, archive)

    observed = md5_file(archive)
    if observed != EXPECTED_MD5:
        raise RuntimeError(
            "archive checksum mismatch: expected %s, observed %s"
            % (EXPECTED_MD5, observed)
        )

    safe_extract(archive, data_dir)
    required = data_dir / "turbine_data" / (
        "fl_df_ALTA2_T11_20250904_to_20260228.parquet"
    )
    if not required.exists():
        raise RuntimeError("expected extracted file is missing: %s" % required)

    if not arguments.keep_archive:
        archive.unlink()
    print("Altahullion turbine data ready at %s" % (data_dir / "turbine_data"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
