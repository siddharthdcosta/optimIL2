"""The ionic liquids every model trains on: cleaning and the train/val split.

Reads data/raw/ilgen/collected_ils.csv, decides which ILs are kept, and
assigns each one to train or val. Every representation is built from this
one result (graphs in il_graphs.py, character strings in il_strings.py), so
the models being compared see the same molecules, in the same split, in the
same row order.

Each builder calls `load_molecules` and writes these two tables next to its
tensors:

    molecules.csv   one row per kept IL: its split, its row in that split's
                    tensors, and its SMILES
    dropped.csv     one row per dropped source row, with the reason

For the same options the tables are identical byte for byte, whichever
builder wrote them. `same_molecules` checks that for two output folders; it
is the test that two datasets really hold the same molecules.

Decisions made here. Counts are for the default settings: of 7,508 source
rows, 6,851 are kept and 657 dropped. Each dropped row is counted once, under
the first rule below that it fails.

- Unparsable SMILES are dropped (3) and so are radicals (53), see molgraph.py.
- Entries that are not one +1 ion and one -1 ion are dropped (131, mostly a
  dication written with a single anion). Fixing the ion counts instead would
  give graphs with three or more components, which breaks the "one cation,
  one anion" rule that generated samples are checked against.
- Duplicates are dropped (224): the same IL after removing stereo and
  isotope labels. Two resonance forms of one IL are NOT merged, because no
  cheap key separates them from true isomers. stats.json reports how many
  kept ILs share a skeleton with another one.
- n_max = 56 heavy atoms covers 99% of the ion pairs; larger ILs are dropped
  (70). The largest has 131 heavy atoms, and covering it would make every
  graph 5 times larger.
- An IL the graph format cannot reproduce exactly would be dropped
  ("roundtrip_mismatch"); at present there are none.
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
- The size and rare-class rules come from the graph format, but they define
  the molecule set for every model. A string model could represent a larger
  IL or a rarer element; it is given the same set anyway so that the
  comparison between models is about the representation, not the data.
- The split holds out whole cations: all ILs that share a cation go to the
  same side. With a random split nearly every validation IL would be a new
  pairing of two ions seen in training, so validation loss could not show
  memorised ions. Cations are grouped by `skeleton_key`, so two resonance
  forms of one cation cannot land on different sides. Anions are not held
  out: there are too few, and a handful appear in most ILs.
"""

import argparse
import csv
import random
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from rdkit import Chem, RDLogger

from optimil2.data.processing.molgraph import BOND_TYPES, IL, Rejected, Vocab, decode_il, encode, read_il, skeleton_key
from optimil2.data.raw.ilgen import ILGEN_DIR
from optimil2.paths import ROOT

COLLECTED_ILS = ILGEN_DIR / "collected_ils.csv"
SPLITS = ("train", "val")


@dataclass(frozen=True)
class MoleculeConfig:
    n_max: int = 56  # most heavy atoms an IL may have; also the number of atom slots of a graph
    min_class_molecules: int = 40  # ILs an atom class must occur in, or ILs containing it are dropped
    val_fraction: float = 0.1  # share of ILs held out, approximate because cations are kept whole
    seed: int = 0  # decides which cations are held out


def add_molecule_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options of MoleculeConfig to a script's parser.

    Every builder script takes these same options, so the molecule set can
    be changed for all representations with the same flags.
    """
    defaults = MoleculeConfig()
    parser.add_argument("--n-max", type=int, default=defaults.n_max, help="most heavy atoms per IL; larger ILs are dropped")
    parser.add_argument("--min-class-molecules", type=int, default=defaults.min_class_molecules,
                        help="ILs an atom class must occur in, or ILs containing it are dropped")
    parser.add_argument("--val-fraction", type=float, default=defaults.val_fraction, help="share of ILs held out")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="decides which cations are held out")


def config_from_arguments(args: argparse.Namespace) -> MoleculeConfig:
    return MoleculeConfig(args.n_max, args.min_class_molecules, args.val_fraction, args.seed)


@dataclass(frozen=True)
class Entry:
    source: str  # SMILES as written in the source file
    il: IL


@dataclass(frozen=True)
class Molecules:
    """The kept ILs with their split, plus what was dropped."""

    entries: list[Entry]  # kept ILs, in source order
    assignment: list[str]  # "train" or "val" for each entry
    dropped: list[tuple[str, str]]  # (source SMILES, reason)
    num_source_rows: int
    config: MoleculeConfig
    source: Path

    def ils(self, split: str | None = None) -> list[IL]:
        """The kept ILs, or those of one split. This order is the row order of every tensor built from them."""
        return [entry.il for entry, name in zip(self.entries, self.assignment) if split is None or name == split]

    def write_tables(self, out: Path) -> None:
        """Write molecules.csv and dropped.csv into out."""
        next_row = Counter()
        with open(out / "molecules.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "row", "n_atoms", "smiles", "cation", "anion", "source_smiles"])
            for entry, split in zip(self.entries, self.assignment):
                il = entry.il
                writer.writerow([split, next_row[split], il.num_atoms, il.smiles, il.cation, il.anion, entry.source])
                next_row[split] += 1
        with open(out / "dropped.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["source_smiles", "reason"])
            writer.writerows(self.dropped)

    def summary(self) -> dict:
        """The part of stats.json that describes the molecule set, whatever the representation."""
        sizes = [il.num_atoms for il in self.ils()]
        splits = {}
        for split in SPLITS:
            members = self.ils(split)
            splits[split] = {
                "ils": len(members),
                "cations": len({il.cation for il in members}),
                "anions": len({il.anion for il in members}),
            }
        train_anions = {il.anion for il in self.ils("train")}
        splits["val"]["anions_not_in_train"] = len({il.anion for il in self.ils("val")} - train_anions)
        skeletons = Counter(skeleton_key(il.mol) for il in self.ils())
        return {
            "source": str(self.source.relative_to(ROOT)) if self.source.is_relative_to(ROOT) else str(self.source),
            "config": asdict(self.config),
            "rows": {
                "source": self.num_source_rows,
                "kept": len(self.entries),
                "dropped": dict(Counter(reason for _, reason in self.dropped).most_common()),
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
        }

    def describe(self) -> str:
        """Two lines for a builder to print: what was kept and dropped, and the split sizes."""
        stats = self.summary()
        reasons = ", ".join(f"{reason} {n}" for reason, n in stats["rows"]["dropped"].items())
        splits = stats["splits"]
        return (
            f"  kept {len(self.entries):,}; dropped {len(self.dropped):,} ({reasons})\n"
            f"  train {splits['train']['ils']:,}, val {splits['val']['ils']:,}; held-out cations {splits['val']['cations']:,}"
        )


def read_smiles(path: Path) -> list[str]:
    """Read the "smiles" column of a CSV."""
    with open(path, newline="") as f:
        return [row["smiles"] for row in csv.DictReader(f)]


def round_trips(il: IL) -> bool:
    """True if encoding the IL as a graph and decoding it again gives back the same IL."""
    vocab = Vocab(sorted(set(il.atom_classes())), list(BOND_TYPES), il.num_atoms)
    x, e = encode(il, vocab)
    back = decode_il(x, e, vocab)
    return back is not None and back.smiles == il.smiles


def clean(source: list[str], config: MoleculeConfig) -> tuple[list[Entry], list[tuple[str, str]]]:
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


def load_molecules(config: MoleculeConfig = MoleculeConfig(), source: Path = COLLECTED_ILS) -> Molecules:
    """Read the source file, clean it and split it. Deterministic for a given config and source."""
    RDLogger.DisableLog("rdApp.*")  # silence per-molecule parse warnings
    smiles = read_smiles(source)
    kept, dropped = clean(smiles, config)
    if not kept:
        raise RuntimeError("no ILs left after cleaning")
    assignment = split_by_cation([entry.il for entry in kept], config.val_fraction, config.seed)
    return Molecules(kept, assignment, dropped, len(smiles), config, source)


def read_table(folder: Path) -> list[dict[str, str]]:
    """The rows of folder/molecules.csv."""
    with open(folder / "molecules.csv", newline="") as f:
        return list(csv.DictReader(f))


def split_smiles(folder: Path, split: str) -> list[str]:
    """SMILES of one split, in the row order of that split's tensors."""
    return [row["smiles"] for row in read_table(folder) if row["split"] == split]


def same_molecules(folder_a: Path, folder_b: Path) -> bool:
    """True if two output folders hold the same ILs, in the same split and the same row order."""

    def rows(folder: Path) -> list[tuple[str, str, str]]:
        return [(row["split"], row["row"], row["smiles"]) for row in read_table(folder)]

    return rows(folder_a) == rows(folder_b)
