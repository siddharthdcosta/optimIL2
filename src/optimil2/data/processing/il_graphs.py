"""Build graph tensors from the ILGen real ionic liquids.

Reads data/raw/ilgen/collected_ils.csv and writes data/processed/il_graphs/:

    train.pt, val.pt   dict of two tensors, "x": uint8 (M, n_max) and
                       "e": uint8 (M, n_max, n_max)
    vocab.json         class numbering and n_max (molgraph.Vocab)
    stats.json         rows kept and dropped, split sizes, class frequencies,
                       and the class marginals of the train split
    molecules.csv      one row per kept IL: its split, its row in that split's
                       tensors, and its SMILES
    dropped.csv        one row per dropped source row, with the reason

The graph format, and what it cannot represent, is described in molgraph.py.
Load the result with optimil2.data.graph_dataset.ILGraphs.

Decisions made here. Counts are for the default settings: of 7,508 source
rows, 6,851 are kept and 657 dropped. Each dropped row is counted once, under
the first rule below that it fails.

- Storage is plain dense tensors, not PyG Data objects. Every graph has the
  same n_max slots and the model is a dense transformer, so two stacked
  tensors are all it needs. About 20 MB in total.
- Unparsable SMILES are dropped (3) and so are radicals (53), see molgraph.py.
- Entries that are not one +1 ion and one -1 ion are dropped (131, mostly a
  dication written with a single anion). Fixing the ion counts instead would
  give graphs with three or more components, which breaks the "one cation,
  one anion" rule that generated samples are checked against.
- Duplicates are dropped (224): the same IL after removing stereo and
  isotope labels. Two resonance forms of one IL are NOT merged, because no
  cheap key separates them from true isomers. stats.json reports how many
  kept ILs share a skeleton with another one.
- n_max = 56 covers 99% of the ion pairs; larger ILs are dropped (70). The
  largest has 131 heavy atoms, and covering it would make every graph 5
  times larger.
- An atom class must occur in at least 40 kept ILs to enter the vocabulary;
  ILs with a rarer class are dropped (176). The check repeats until every
  remaining class passes, because dropping ILs lowers the counts of the
  other classes they contain. ILs containing each class before the cutoff:

      O+ 1, Cl+ 2, Cl+2 3, B 4, F- 5, Li+ 12, K+ 17, I 17, Fe- 24, Si 27,
      C+ 34, Na+ 34 | Al- 45, S+ 82, S- 82, Br 86, Cl+3 87, I- 112, ...

  So 40 removes the elements Si, Fe, Na, K and Li and a few rare charge
  states (C+ is cyclopropenium and similar cations), and keeps Al
  (chloroaluminates). The Na/K/Li entries are inorganic salts, not ILs. Any
  cutoff from 46 to 82 gives the same result as 40 minus the Al- class.
- The split holds out whole cations: all ILs that share a cation go to the
  same side. With a random split nearly every validation IL would be a new
  pairing of two ions seen in training, so validation loss could not show
  memorised ions. Cations are grouped by `skeleton_key`, so two resonance
  forms of one cation cannot land on different sides. Anions are not held
  out: there are too few, and a handful appear in most ILs.
- The class marginals in stats.json (the noise distribution of the diffusion
  model) are measured on the train split only, over all n_max slots and all
  slot pairs, so they include the EMPTY and NO_BOND classes.
"""

import csv
import json
import random
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger

from optimil2.data.processing.molgraph import BOND_TYPES, IL, Rejected, Vocab, decode_il, encode, read_il, skeleton_key
from optimil2.data.raw.ilgen import ILGEN_DIR
from optimil2.paths import PROCESSED_DATA, ROOT

COLLECTED_ILS = ILGEN_DIR / "collected_ils.csv"
IL_GRAPHS_DIR = PROCESSED_DATA / "il_graphs"
SPLITS = ("train", "val")


@dataclass(frozen=True)
class GraphConfig:
    n_max: int = 56  # atom slots per graph
    min_class_molecules: int = 40  # ILs an atom class must occur in to enter the vocabulary
    val_fraction: float = 0.1  # share of ILs held out, approximate because cations are kept whole
    seed: int = 0  # decides which cations are held out


@dataclass(frozen=True)
class Entry:
    source: str  # SMILES as written in the source file
    il: IL


def read_smiles(path: Path) -> list[str]:
    """Read the "smiles" column of a CSV."""
    with open(path, newline="") as f:
        return [row["smiles"] for row in csv.DictReader(f)]


def round_trips(il: IL) -> bool:
    """True if encoding the IL and decoding it again gives back the same IL."""
    vocab = Vocab(sorted(set(il.atom_classes())), list(BOND_TYPES), il.num_atoms)
    x, e = encode(il, vocab)
    back = decode_il(x, e, vocab)
    return back is not None and back.smiles == il.smiles


def clean(source: list[str], config: GraphConfig) -> tuple[list[Entry], list[tuple[str, str]]]:
    """Split source SMILES into kept entries and (SMILES, reason) pairs for the dropped ones."""
    kept: list[Entry] = []
    dropped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for smiles in source:
        try:
            il = read_il(smiles)
        except Rejected as reason:
            dropped.append((smiles, str(reason)))
            continue
        if il.smiles in seen:
            dropped.append((smiles, "duplicate"))
        elif il.num_atoms > config.n_max:
            dropped.append((smiles, "too_large"))
        elif not round_trips(il):
            dropped.append((smiles, "roundtrip_mismatch"))
        else:
            kept.append(Entry(smiles, il))
        seen.add(il.smiles)

    # Dropping ILs lowers the counts of the classes they contain, so repeat
    # until every remaining class occurs in enough ILs.
    while True:
        in_ils = Counter(atom for entry in kept for atom in set(entry.il.atom_classes()))
        rare = {atom for atom, n in in_ils.items() if n < config.min_class_molecules}
        if not rare:
            return kept, dropped
        has_rare = [bool(rare & set(entry.il.atom_classes())) for entry in kept]
        dropped += [(entry.source, "rare_class") for entry, flag in zip(kept, has_rare) if flag]
        kept = [entry for entry, flag in zip(kept, has_rare) if not flag]


def derive_vocab(ils: list[IL], n_max: int) -> Vocab:
    """Number the atom classes found in ils, most widespread first, and the bond types found."""
    in_ils = Counter(atom for il in ils for atom in set(il.atom_classes()))
    atoms = sorted(in_ils, key=lambda atom: (-in_ils[atom], atom))
    found = {name for il in ils for name in il.bond_names()}
    return Vocab(atoms, [name for name in BOND_TYPES if name in found], n_max)


def split_by_cation(ils: list[IL], val_fraction: float, seed: int) -> list[str]:
    """Assign "train" or "val" to each IL, keeping all ILs of one cation together.

    Cations are shuffled and moved to val one at a time until val holds
    val_fraction of the ILs, so val can overshoot by the size of one cation.
    """
    keys = [skeleton_key(Chem.MolFromSmiles(il.cation)) for il in ils]
    size = Counter(keys)
    cations = sorted(size)
    random.Random(seed).shuffle(cations)
    held_out: set[str] = set()
    n_val = 0
    for cation in cations:
        if n_val >= val_fraction * len(ils):
            break
        held_out.add(cation)
        n_val += size[cation]
    return ["val" if key in held_out else "train" for key in keys]


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
    with open(folder / "molecules.csv", newline="") as f:
        rows = list(csv.DictReader(f))
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
    config: GraphConfig = GraphConfig(),
    source: Path = COLLECTED_ILS,
    out: Path = IL_GRAPHS_DIR,
) -> Path:
    """Run the whole pipeline and return the output folder."""
    RDLogger.DisableLog("rdApp.*")  # silence per-molecule parse warnings
    smiles = read_smiles(source)
    print(f"[graphs] {len(smiles):,} rows in {source.name}")
    kept, dropped = clean(smiles, config)
    if not kept:
        raise RuntimeError("no ILs left after cleaning")
    ils = [entry.il for entry in kept]
    vocab = derive_vocab(ils, config.n_max)
    assignment = split_by_cation(ils, config.val_fraction, config.seed)

    out.mkdir(parents=True, exist_ok=True)
    vocab.save(out / "vocab.json")
    tensors = {}
    row_in_split = {}
    for split in SPLITS:
        members = [i for i, name in enumerate(assignment) if name == split]
        row_in_split.update({i: row for row, i in enumerate(members)})
        tensors[split] = to_tensors([ils[i] for i in members], vocab)
        torch.save(tensors[split], out / f"{split}.pt")

    with open(out / "molecules.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "row", "n_atoms", "smiles", "cation", "anion", "source_smiles"])
        for i, entry in enumerate(kept):
            il = entry.il
            writer.writerow([assignment[i], row_in_split[i], il.num_atoms, il.smiles, il.cation, il.anion, entry.source])
    with open(out / "dropped.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_smiles", "reason"])
        writer.writerows(dropped)

    marginals = class_marginals(tensors["train"], vocab)
    missing = [label for label, p in zip(vocab.atom_labels(), marginals["atom"]) if p == 0]
    if missing:
        raise RuntimeError(
            f"atom classes {missing} never occur in the train split, so their noise probability would be zero. "
            "Use another seed or raise min_class_molecules."
        )
    stats = summarise(len(smiles), kept, dropped, assignment, vocab, marginals, config, source)
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    checked = verify(out)
    reasons = ", ".join(f"{reason} {n}" for reason, n in stats["rows"]["dropped"].items())
    print(f"  kept {len(kept):,}; dropped {len(dropped):,} ({reasons})")
    print(f"  train {stats['splits']['train']['ils']:,}, val {stats['splits']['val']['ils']:,}; held-out cations {stats['splits']['val']['cations']:,}")
    print(f"  {vocab.num_atom_classes} atom classes: {' '.join(vocab.atom_labels())}")
    print(f"  {vocab.num_bond_classes} bond classes: {' '.join(vocab.bonds)}")
    print(f"  verified: all {checked:,} saved graphs decode back to their SMILES")
    return out


def summarise(
    n_source: int,
    kept: list[Entry],
    dropped: list[tuple[str, str]],
    assignment: list[str],
    vocab: Vocab,
    marginals: dict[str, list[float]],
    config: GraphConfig,
    source: Path,
) -> dict:
    """Collect the contents of stats.json."""
    ils = [entry.il for entry in kept]
    sizes = [il.num_atoms for il in ils]
    in_ils = Counter(atom for il in ils for atom in set(il.atom_classes()))
    atoms = Counter(atom for il in ils for atom in il.atom_classes())
    bonds = Counter(name for il in ils for name in il.bond_names())

    splits = {}
    for split in SPLITS:
        members = [il for il, name in zip(ils, assignment) if name == split]
        splits[split] = {
            "ils": len(members),
            "cations": len({il.cation for il in members}),
            "anions": len({il.anion for il in members}),
        }
    train_anions = {il.anion for il, name in zip(ils, assignment) if name == "train"}
    val_anions = {il.anion for il, name in zip(ils, assignment) if name == "val"}
    splits["val"]["anions_not_in_train"] = len(val_anions - train_anions)

    skeletons = Counter(skeleton_key(il.mol) for il in ils)
    return {
        "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "config": asdict(config),
        "rows": {
            "source": n_source,
            "kept": len(kept),
            "dropped": dict(Counter(reason for _, reason in dropped).most_common()),
        },
        "splits": splits,
        "atoms_per_il": {
            "min": min(sizes),
            "median": statistics.median(sizes),
            "mean": round(statistics.mean(sizes), 2),
            "max": max(sizes),
        },
        # ILs whose skeleton (see molgraph.skeleton_key) is shared with another
        # kept IL: mostly one IL drawn as two resonance forms.
        "ils_sharing_a_skeleton": sum(n for n in skeletons.values() if n > 1),
        "atom_classes": [
            {"index": vocab.atom_index(atom), "label": atom.label, "ils": in_ils[atom], "atoms": atoms[atom]}
            for atom in vocab.atoms[1:]
        ],
        "bond_classes": [
            {"index": vocab.bond_index(name), "label": name, "bonds": bonds[name]} for name in vocab.bonds[1:]
        ],
        "train_marginals": marginals,
    }
