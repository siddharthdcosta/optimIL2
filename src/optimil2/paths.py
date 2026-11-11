"""Locations of the project's top-level folders.

Import these instead of building paths by hand, so scripts work from any
working directory and on any machine.
"""

from pathlib import Path

# src/optimil2/paths.py -> repo root is two levels above the package folder.
# This relies on the editable install (`pip install -e .`) described in CLAUDE.md.
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW_DATA = DATA / "raw"
PROCESSED_DATA = DATA / "processed"
REPOS = ROOT / "repos"
DOCS = ROOT / "docs"
