# CLAUDE.md

## Purpose

The purpose of this project is **ionic liquid generation**.

## Environment

This project uses the `optimIL2` conda environment (Python 3.12). Activate it before running any code:

```bash
conda activate optimIL2
```

Main dependencies: numpy, PyTorch, RDKit, and PyTorch Geometric (`torch_geometric`). These are installed with pip inside the conda environment, since PyTorch no longer publishes conda packages.

Dependencies are listed in `pyproject.toml` under `dependencies`, which is the single source of truth. `environment.yml` pins Python 3.12 and installs the project with `pip install -e .`. When adding a dependency, add it to `pyproject.toml` only, then rerun `pip install -e .`.

- With conda: `conda env create -f environment.yml`
- Without conda: create a Python 3.12 venv, then `pip install -e .`

The project itself is installed as an editable package (`optimil2`), so `import optimil2` works from any script or notebook without `sys.path` changes.

## Layout

- `src/optimil2/`: importable project code. Reusable logic (data processing, datasets, models) goes here.
  - `src/optimil2/data/`: code for downloading and processing data.
    - `src/optimil2/data/raw/`: one module per source that downloads untouched files into `data/raw/<source>/`, plus shared helpers in `download.py`. Scripts in `scripts/data/` only call these.
    - `src/optimil2/data/processing/`: code that reads `data/raw/` and writes cleaned files to `data/processed/`. Never modifies `data/raw/`.
      - `il_molecules.py` decides which ILs are kept and how they are split into train and val. Every representation (graphs in `il_graphs.py`, strings in `il_strings.py`) is built from it, so that models are compared on the same molecules. Add a new representation the same way; do not filter or split anywhere else.
  - `src/optimil2/model_primitive/`: two first, crude generative models. One is discrete diffusion on the IL graphs; the other is its controlled baseline, the same diffusion on SMILES characters (files named `string_*`). Both take their noise process from `noise.py`; extend that file rather than copying from it. Untrained; the design choices are recorded in the module docstrings.
  - `src/optimil2/evaluation.py`: metrics for generated molecules (validity, uniqueness, novelty). They are computed on SMILES, so every model is scored by the same code.
- `scripts/`: thin entry points that import from `optimil2` and run something.
  - `scripts/setup/`: one-time setup (cloning external repos, environment).
    - `folders.py` creates the gitignored folders (`data/`, `data/raw/`, `data/processed/`, `repos/`). Run it first on a fresh clone, and add new gitignored folders to it.
  - `scripts/data/`: download and process datasets.
  - `scripts/model/`: build and check models.
- `data/`: data files only, never code. Gitignored; recreate it by running `scripts/data/`.
- `repos/`: cloned external repos (e.g. FlowMol). Gitignored.
  - Clone them with `python scripts/setup/repos.py`.
  - They are for reading only. Never install them or their environments into `optimIL2`: they pin older, conflicting versions of Python, PyTorch and RDKit. To run one, use a separate conda environment.
- `investigate/`: one-off scripts for exploring a question. Not part of the pipeline; nothing imports them. Downloads are cached in `data/investigate/`.
- `docs/`, `research/`: notes and reference material.
