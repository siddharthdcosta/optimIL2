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
