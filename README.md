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
`src/optimil2/data/processing/il_molecules.py` (which ILs are kept, and the
split) and `src/optimil2/data/processing/molgraph.py` (graph format). Load a
split with `optimil2.data.graph_dataset.ILGraphs`.

```bash
python scripts/data/build_strings.py   # the same ILs as character tokens, about 15 seconds
```

This writes `data/processed/il_strings/`, the data of the string baseline
model. Each IL is its SMILES (`cation.anion`) with one token per character,
padded to 128 positions:

| File | Contents |
|---|---|
| `train.pt`, `val.pt` | The same 6,107 and 744 ILs as the graphs, in the same row order. `tokens` holds the character class of each position. |
| `vocab.json` | The 31 character classes (30 characters; class 0 is padding) |
| `stats.json` | Rows kept and dropped, split sizes, character counts, and the class frequencies of the train split |
| `molecules.csv`, `dropped.csv` | Identical to the files in `il_graphs/` |

Both builders take their ILs and their split from `il_molecules.py`, so the two
folders hold the same molecules as long as both are built with the same
options. `build_strings.py` says so when it finishes, or warns if they differ.
The string format is explained at the top of
`src/optimil2/data/processing/molstring.py`. Load a split with
`optimil2.data.string_dataset.ILStrings`.

## Models

`src/optimil2/model_primitive/` holds two first, crude generative models.
Neither is trained yet and there is no training loop.

- **Graph model.** Discrete diffusion on the IL graphs, in the style of
  DiGress. Noise replaces atom and bond classes at random, a small graph
  transformer predicts the clean classes, and sampling walks back from pure
  noise.
- **String model.** The controlled baseline for it: the same diffusion on the
  characters of the SMILES. A small bidirectional transformer predicts the
  clean character of every position at once; nothing is generated left to
  right.

The two share everything except the representation and the denoiser. Both use
the one noise process in `noise.py`: classes are redrawn from their training
frequencies (marginal transitions), on a cosine schedule with 500 steps, and
sampling uses the same posterior. Both train on cross-entropy against the
clean classes, on the same ILs with the same split, and both are scored by
`optimil2.evaluation`. What differs:

| | Graph model | String model |
|---|---|---|
| One IL is | 56 atom slots and their 1,540 pairs | 128 character positions |
| Classes | 23 atom classes, 4 bond classes | 31 character classes |
| Padding, diffused like any other class | the empty atom slot | the padding character |
| Denoiser | dense graph transformer, 1.32M parameters, blind to atom order | bidirectional transformer, 1.27M parameters, learned position embeddings |
| Loss | atoms + 5 × bonds | characters |
| Code | `diffusion.py`, `denoiser.py`, `checks.py` | `string_diffusion.py`, `string_denoiser.py`, `string_checks.py` |

The design choices are recorded at the top of each of those files.

```bash
python scripts/model/smoke_test.py          # graph model, about 40 seconds
python scripts/model/smoke_test_string.py   # string model, about 25 seconds
```

Each builds its model with random weights and runs forward passes on real ILs,
one backward pass and full sampling, on CPU and on MPS if available. The
string test needs both `data/processed/` folders: it also checks that they
hold the same ILs row for row. Samples from the untrained models are not valid
molecules; that is expected.

```python
from optimil2.model_primitive.diffusion import build_model
from optimil2.model_primitive.string_diffusion import build_string_model

graph_model = build_model()            # random weights, sized from data/processed/il_graphs/
parts = graph_model.loss(x0, e0)       # x0: (B, 56) atom classes, e0: (B, 56, 56) bond classes
x, e = graph_model.sample(8)           # 8 generated graphs

string_model = build_string_model()    # random weights, sized from data/processed/il_strings/
parts = string_model.loss(tokens0)     # tokens0: (B, 128) character classes
tokens = string_model.sample(8)        # 8 generated strings
```

### Evaluation

`optimil2.evaluation.evaluate` scores samples given as SMILES, so both models
go through the same code:

```python
from optimil2.data.processing import molgraph, molstring
from optimil2.data.processing.il_graphs import IL_GRAPHS_DIR
from optimil2.data.processing.il_molecules import split_smiles
from optimil2.data.processing.il_strings import IL_STRINGS_DIR
from optimil2.evaluation import evaluate

train_smiles = split_smiles(IL_GRAPHS_DIR, "train")   # IL_STRINGS_DIR gives the same list
graph_vocab = molgraph.Vocab.load(IL_GRAPHS_DIR / "vocab.json")
string_vocab = molstring.CharVocab.load(IL_STRINGS_DIR / "vocab.json")

from_graphs = [molgraph.to_smiles(x[i], e[i], graph_vocab) for i in range(len(x))]
from_strings = [molstring.decode(row, string_vocab) for row in tokens]
print(evaluate(from_graphs, train_smiles))    # valid ..%, unique ..%, novel ..%, ion pairs ..%
print(evaluate(from_strings, train_smiles))
```

| Metric | Meaning |
|---|---|
| `validity` | Share of samples that RDKit parses and that have net charge 0 |
| `uniqueness` | Share of the valid samples that are distinct |
| `novelty` | Share of the distinct valid samples that are not in the training set |
| `ion_pair` | Share of samples that are one +1 ion and one -1 ion without radicals, which is what the training data calls an IL |

Validity alone is lenient: a neutral molecule with no ions passes. `ion_pair`
is the number that says whether a model makes ionic liquids.

See `docs/data.md` for a survey of all data sources.
