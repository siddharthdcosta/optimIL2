"""Shared helpers for downloading raw data files."""

import hashlib
import json
import urllib.request
from datetime import date
from pathlib import Path

CHUNK_BYTES = 1 << 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, dest: Path) -> bool:
    """Download url to dest. Returns False if dest already existed.

    The download goes to a temporary file first, so an interrupted run never
    leaves a partial file at dest.
    """
    if dest.exists():
        print(f"[skip] {dest.name}: already downloaded")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    print(f"[download] {dest.name}")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as file:
        while chunk := response.read(CHUNK_BYTES):
            file.write(chunk)
    partial.replace(dest)
    return True


def write_manifest(folder: Path, source: str, files: dict[str, str]) -> None:
    """Record where each file in folder came from.

    files maps each file name to the URL it was downloaded from.
    """
    manifest = {
        "source": source,
        "written": date.today().isoformat(),
        "files": {
            name: {
                "url": url,
                "bytes": (folder / name).stat().st_size,
                "sha256": sha256_of(folder / name),
            }
            for name, url in files.items()
        },
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
