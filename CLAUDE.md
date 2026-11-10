# CLAUDE.md

## Purpose

The purpose of this project is **ionic liquid generation**.

## Environment

This project uses the `optimIL2` conda environment (Python 3.12). Activate it before running any code:

```bash
conda activate optimIL2
```

Main dependencies: numpy, PyTorch, RDKit, and PyTorch Geometric (`torch_geometric`). These are installed with pip inside the conda environment, since PyTorch no longer publishes conda packages.

Dependencies are listed in `requirements.txt`, which is the single source of truth. `environment.yml` pins Python 3.12 and installs `requirements.txt` through pip. When adding a dependency, add it to `requirements.txt` only.

- With conda: `conda env create -f environment.yml`
- Without conda: create a Python 3.12 venv, then `pip install -e .`

The project itself is installed as an editable package (`optimil2`), so `import optimil2` works from any script or notebook without `sys.path` changes.

## Layout

- `src/optimil2/`: importable project code. Reusable logic (data processing, datasets, models) goes here.
  - `src/optimil2/data/`: code for downloading and processing data.
- `scripts/`: thin entry points that import from `optimil2` and run something.
  - `scripts/setup/`: one-time setup (cloning external repos, environment).
  - `scripts/data/`: download and process datasets.
- `data/`: data files only, never code. Gitignored; recreate it by running `scripts/data/`.
- `repos/`: cloned external repos (e.g. FlowMol). Gitignored.
- `docs/`, `research/`: notes and reference material.
