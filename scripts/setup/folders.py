"""Create the gitignored folders that a fresh clone of this repo is missing.

Usage:
    python scripts/setup/folders.py

Safe to re-run: existing folders are left untouched. When a new gitignored
folder is needed, add its path to optimil2/paths.py and to FOLDERS below.
"""

from optimil2.paths import DATA, PROCESSED_DATA, RAW_DATA, REPOS, ROOT

FOLDERS = [
    DATA,
    RAW_DATA,
    PROCESSED_DATA,
    REPOS,
]


def main() -> None:
    for folder in FOLDERS:
        existed = folder.exists()
        folder.mkdir(parents=True, exist_ok=True)
        status = "exists" if existed else "created"
        print(f"[{status}] {folder.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
