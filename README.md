# Ionic Liquid Generation

## Setup

Requires Python 3.12.

**With conda:**

```bash
conda env create -f environment.yml
conda activate optimIL2
```

**Without conda:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .   # installs dependencies plus this package
```

**Then, with the environment active, run from the repo root:**

```bash
python scripts/setup/folders.py   # creates the gitignored folders (data/, data/raw/, repos/)
python scripts/setup/repos.py     # clones FlowMol and ILGen (code only) into repos/
```

Both are safe to re-run. The cloned repos are for reading only: do not install them or their environments into `optimIL2`.

## Data

Download the datasets (run from the repo root, with the environment active):

```bash
python scripts/data/ilgen.py   # ILGen real ILs and PubChem ions, about 70 MB, under a minute
```

This saves untouched files to `data/raw/`, each folder with a `manifest.json`
recording the source URL and checksum of every file:

| Folder | Contents |
|---|---|
| `data/raw/ilgen/` | 7,505 real ILs (`collected_ils.csv`), their cations and anions, 2,206 melting points, and ILGen's predicted pairs (`expanded_ils.csv`, not real ILs) |
| `data/raw/pubchem_ions/` | 903,584 cations and 401,473 anions extracted from PubChem by the ILGen authors, used for pretraining |

Safe to re-run: files already downloaded are skipped. The ILGen repos have no
license, so keep these files local and do not redistribute them.

See `docs/data.md` for a survey of all data sources.
