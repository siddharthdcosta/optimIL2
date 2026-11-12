"""Build graph tensors from the ILGen real ionic liquids.

Takes the cleaned, split ILs from il_molecules.py and writes
data/processed/il_graphs/:

    train.pt, val.pt   dict of two tensors, "x": uint8 (M, n_max) and
                       "e": uint8 (M, n_max, n_max)
    vocab.json         class numbering and n_max (molgraph.Vocab)
    stats.json         rows kept and dropped, split sizes, class frequencies,
                       and the class marginals of the train split
    molecules.csv      one row per kept IL: its split, its row in that split's
                       tensors, and its SMILES
    dropped.csv        one row per dropped source row, with the reason

Which ILs are kept and how they are split is decided in il_molecules.py, and
is shared with the string representation (il_strings.py). The graph format,
and what it cannot represent, is described in molgraph.py. Load the result
with optimil2.data.graph_dataset.ILGraphs.

Decisions made here:

- Storage is plain dense tensors, not PyG Data objects. Every graph has the
  same n_max slots and the model is a dense transformer, so two stacked
  tensors are all it needs. About 20 MB in total.
- The class marginals in stats.json (the noise distribution of the diffusion
  model) are measured on the train split only, over all n_max slots and all
  slot pairs, so they include the EMPTY and NO_BOND classes.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from optimil2.data.processing.il_molecules import COLLECTED_ILS, SPLITS, MoleculeConfig, load_molecules, read_table
from optimil2.data.processing.molgraph import BOND_TYPES, IL, Vocab, decode_il, encode
from optimil2.paths import PROCESSED_DATA

IL_GRAPHS_DIR = PROCESSED_DATA / "il_graphs"


def derive_vocab(ils: list[IL], n_max: int) -> Vocab:
    """Number the atom classes found in ils, most widespread first, and the bond types found."""
    in_ils = Counter(atom for il in ils for atom in set(il.atom_classes()))
    atoms = sorted(in_ils, key=lambda atom: (-in_ils[atom], atom))
    found = {name for il in ils for name in il.bond_names()}
    return Vocab(atoms, [name for name in BOND_TYPES if name in found], n_max)


def to_tensors(ils: list[IL], vocab: Vocab) -> dict[str, torch.Tensor]:
    """Stack the graphs of ils into "x" (M, n_max) and "e" (M, n_max, n_max)."""
    graphs = [encode(il, vocab) for il in ils]
    x = np.stack([x for x, _ in graphs]) if graphs else np.zeros((0, vocab.n_max), np.uint8)
    e = np.stack([e for _, e in graphs]) if graphs else np.zeros((0, vocab.n_max, vocab.n_max), np.uint8)
    return {"x": torch.from_numpy(x), "e": torch.from_numpy(e)}


def class_marginals(tensors: dict[str, torch.Tensor], vocab: Vocab) -> dict[str, list[float]]:
    """Share of each atom class over all slots, and of each bond class over all slot pairs."""
    rows, cols = torch.triu_indices(vocab.n_max, vocab.n_max, offset=1)
    atom = torch.bincount(tensors["x"].flatten().long(), minlength=vocab.num_atom_classes)
    bond = torch.bincount(tensors["e"][:, rows, cols].flatten().long(), minlength=vocab.num_bond_classes)
    return {"atom": (atom.double() / atom.sum()).tolist(), "bond": (bond.double() / bond.sum()).tolist()}


def verify(folder: Path) -> int:
    """Check that every saved graph decodes to the SMILES listed for it in molecules.csv.

    Returns the number of graphs checked. Raises RuntimeError on any mismatch.
    """
    vocab = Vocab.load(folder / "vocab.json")
    tensors = {split: torch.load(folder / f"{split}.pt") for split in SPLITS}
    rows = read_table(folder)
    if len(rows) != sum(len(t["x"]) for t in tensors.values()):
        raise RuntimeError("molecules.csv and the tensors disagree on the number of ILs")
    for row in rows:
        graph = tensors[row["split"]]
        index = int(row["row"])
        il = decode_il(graph["x"][index], graph["e"][index], vocab)
        if il is None or il.smiles != row["smiles"]:
            raise RuntimeError(f"{row['split']} row {index} decodes to {il and il.smiles}, expected {row['smiles']}")
    return len(rows)


def build_il_graphs(
    config: MoleculeConfig = MoleculeConfig(),
    source: Path = COLLECTED_ILS,
    out: Path = IL_GRAPHS_DIR,
) -> Path:
    """Run the whole pipeline and return the output folder."""
    molecules = load_molecules(config, source)
    print(f"[graphs] {molecules.num_source_rows:,} rows in {source.name}")
    ils = molecules.ils()
    vocab = derive_vocab(ils, config.n_max)

    out.mkdir(parents=True, exist_ok=True)
    vocab.save(out / "vocab.json")
    tensors = {split: to_tensors(molecules.ils(split), vocab) for split in SPLITS}
    for split in SPLITS:
        torch.save(tensors[split], out / f"{split}.pt")
    molecules.write_tables(out)

    marginals = class_marginals(tensors["train"], vocab)
    missing = [label for label, p in zip(vocab.atom_labels(), marginals["atom"]) if p == 0]
    if missing:
        raise RuntimeError(
            f"atom classes {missing} never occur in the train split, so their noise probability would be zero. "
            "Use another seed or raise min_class_molecules."
        )
    in_ils = Counter(atom for il in ils for atom in set(il.atom_classes()))
    atoms = Counter(atom for il in ils for atom in il.atom_classes())
    bonds = Counter(name for il in ils for name in il.bond_names())
    stats = {
        **molecules.summary(),
        "atom_classes": [
            {"index": vocab.atom_index(atom), "label": atom.label, "ils": in_ils[atom], "atoms": atoms[atom]}
            for atom in vocab.atoms[1:]
        ],
        "bond_classes": [
            {"index": vocab.bond_index(name), "label": name, "bonds": bonds[name]} for name in vocab.bonds[1:]
        ],
        "train_marginals": marginals,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    checked = verify(out)
    print(molecules.describe())
    print(f"  {vocab.num_atom_classes} atom classes: {' '.join(vocab.atom_labels())}")
    print(f"  {vocab.num_bond_classes} bond classes: {' '.join(vocab.bonds)}")
    print(f"  verified: all {checked:,} saved graphs decode back to their SMILES")
    return out
