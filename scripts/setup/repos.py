"""Clone the external code repos this project depends on into repos/.

Only code is downloaded. Each clone is shallow (latest commit only), and
large data and model-weight files are left out, so the three repos take a
few MB instead of a few hundred.

Usage:
    python scripts/setup/repos.py

Safe to re-run: repos that are already cloned are skipped.
"""

import subprocess
import sys
from dataclasses import dataclass

from optimil2.paths import REPOS


@dataclass(frozen=True)
class Repo:
    name: str
    url: str
    # Paths to leave out of the checkout, in .gitignore pattern syntax.
    exclude: tuple[str, ...] = ()


REPOS_TO_CLONE = [
    Repo(
        name="FlowMol",
        url="https://github.com/Dunni3/FlowMol.git",
        exclude=("/examples/*.ipynb", "/images/"),
    ),
    # The ILGen datasets (database/) are fetched by scripts/data/, not here.
    Repo(
        name="ILGen-overall",
        url="https://github.com/fate1997/ILGen-overall.git",
        exclude=("/database/", "*.pt", "*.png"),
    ),
    Repo(
        name="ILGen-ion",
        url="https://github.com/fate1997/ILGen-ion.git",
        exclude=("/database/", "*.pt", "*.png"),
    ),
]


def git(*args: str, cwd=None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def clone(repo: Repo) -> None:
    dest = REPOS / repo.name
    if dest.exists():
        print(f"[skip] {repo.name}: already at {dest}")
        return

    print(f"[clone] {repo.name}")
    # --filter=blob:none downloads file contents only when they are checked
    # out, so the files excluded below are never fetched at all.
    git("clone", "--depth", "1", "--filter=blob:none", "--no-checkout", repo.url, str(dest))
    if repo.exclude:
        patterns = ["/*", *(f"!{pattern}" for pattern in repo.exclude)]
        git("sparse-checkout", "set", "--no-cone", *patterns, cwd=dest)
    git("checkout", cwd=dest)


def main() -> int:
    REPOS.mkdir(exist_ok=True)
    failed = []
    for repo in REPOS_TO_CLONE:
        try:
            clone(repo)
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            print(f"[fail] {repo.name}: {error}", file=sys.stderr)
            failed.append(repo.name)
    if failed:
        print(f"Failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"Done. Repos are in {REPOS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
