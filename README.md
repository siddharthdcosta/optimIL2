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
python scripts/setup/folders.py   # creates the gitignored folders (data/, data/raw/, data/processed/, repos/)
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

Then process them:

```bash
python scripts/data/filter_charge.py   # keeps +1 cations and -1 anions, about 1 minute
```

This writes `data/processed/pubchem_ions/`: 834,945 cations and 364,311 anions,
plus a `summary.json` with how many ions were kept and dropped.

```bash
python scripts/data/build_graphs.py   # real ILs as graph tensors, about 15 seconds
```

This writes `data/processed/il_graphs/`. Each IL is one graph of heavy atoms
with two disconnected parts (cation and anion), padded to 56 atom slots:

| File | Contents |
|---|---|
| `train.pt`, `val.pt` | 6,107 and 744 ILs. `x` holds the atom class of each slot, `e` the bond class of each slot pair. The split holds out whole cations. |
| `vocab.json` | The 23 atom classes (element with charge, such as `N+`; class 0 is an empty slot) and 4 bond classes (none, single, double, triple) |
| `stats.json` | Rows kept and dropped, split sizes, class counts, and the class frequencies of the train split |
| `molecules.csv` | SMILES of every kept IL, with its split and its row in the tensors |
| `dropped.csv` | The 657 source rows that were dropped, each with the reason |

The choices behind these numbers are explained at the top of
`src/optimil2/data/processing/molgraph.py` (graph format) and
`src/optimil2/data/processing/il_graphs.py` (filters and split). Load a split
with `optimil2.data.graph_dataset.ILGraphs`.

## Model

`src/optimil2/model_primitive/` holds a first, crude generative model: discrete
diffusion on the IL graphs, in the style of DiGress. Noise replaces atom and
bond classes at random, a small graph transformer (about 1.3M parameters)
predicts the clean classes, and sampling walks back from pure noise. It is
not trained yet and there is no training loop.

```bash
python scripts/model/smoke_test.py   # builds it with random weights and checks it, about 40 seconds
```

This runs forward passes on real ILs, one backward pass and full sampling, on
CPU and on MPS if available. It needs `data/processed/il_graphs/`. Samples
from the untrained model are not valid molecules; that is expected.

```python
from optimil2.model_primitive.diffusion import build_model

model = build_model()            # random weights, sized from vocab.json and stats.json
parts = model.loss(x0, e0)       # x0: (B, 56) atom classes, e0: (B, 56, 56) bond classes
x, e = model.sample(8)           # 8 generated graphs; decode with molgraph.decode_il
```

See `docs/data.md` for a survey of all data sources.
