"""Build character-token tensors from the ILGen real ionic liquids.

The string twin of il_graphs.py. Takes the same cleaned, split ILs from
il_molecules.py and writes data/processed/il_strings/:

    train.pt, val.pt   dict of one tensor, "tokens": uint8 (M, l_max)
    vocab.json         character numbering and l_max (molstring.CharVocab)
    stats.json         rows kept and dropped, split sizes, character
                       frequencies, and the class marginals of the train split
    molecules.csv      one row per kept IL: its split, its row in that split's
                       tensors, and its SMILES
    dropped.csv        one row per dropped source row, with the reason

Row i of a split here is the same IL as row i of that split in
data/processed/il_graphs/, provided both were built with the same options.
molecules.csv and dropped.csv are then identical in the two folders, and
il_molecules.same_molecules checks it. The token format is described in
molstring.py. Load the result with optimil2.data.string_dataset.ILStrings.

Decisions made here:

- No IL is dropped for its string. Which ILs are kept is decided only in
  il_molecules.py, so that the string and graph models train on the same
  molecules. If an IL does not fit in l_max the build stops with an error
  instead of quietly making the two training sets differ.
- l_max = 128 positions. The longest kept IL has 121 characters (median 42,
  99% within 92), so about 65% of all positions are PAD. For comparison,
  about 60% of the graph's atom slots are EMPTY.
- Characters are numbered by how many ILs contain them, most widespread
  first, as the graph's atom classes are.
- The class marginals in stats.json (the noise distribution of the diffusion
  model) are measured on the train split only, over all l_max positions, so
  they include PAD.
- Every character must occur in the train split, or the build stops: a
  character with zero noise probability breaks the diffusion posterior. The
  rarest ones are few enough for this to depend on the seed ("5" is in one
  IL, "o" in 14, "4" in 17). With the default seed all 30 are in train.
"""

import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from optimil2.data.processing.il_molecules import COLLECTED_ILS, SPLITS, MoleculeConfig, load_molecules, read_table
from optimil2.data.processing.molstring import CharVocab, decode, encode
from optimil2.paths import PROCESSED_DATA

IL_STRINGS_DIR = PROCESSED_DATA / "il_strings"
L_MAX = 128  # positions per string


def derive_vocab(smiles: list[str], l_max: int) -> CharVocab:
    """Number the characters found in smiles, most widespread first."""
    in_ils = Counter(char for text in smiles for char in set(text))
    return CharVocab(sorted(in_ils, key=lambda char: (-in_ils[char], char)), l_max)


def to_tensors(smiles: list[str], vocab: CharVocab) -> dict[str, torch.Tensor]:
    """Stack the token rows of smiles into "tokens" (M, l_max)."""
    rows = [encode(text, vocab) for text in smiles]
    tokens = np.stack(rows) if rows else np.zeros((0, vocab.l_max), np.uint8)
    return {"tokens": torch.from_numpy(tokens)}


def class_marginals(tensors: dict[str, torch.Tensor], vocab: CharVocab) -> dict[str, list[float]]:
    """Share of each character class over all positions."""
    counts = torch.bincount(tensors["tokens"].flatten().long(), minlength=vocab.num_classes)
    return {"char": (counts.double() / counts.sum()).tolist()}


def verify(folder: Path) -> int:
    """Check that every saved token row decodes to the SMILES listed for it in molecules.csv.

    Returns the number of rows checked. Raises RuntimeError on any mismatch.
    """
    vocab = CharVocab.load(folder / "vocab.json")
    tensors = {split: torch.load(folder / f"{split}.pt") for split in SPLITS}
    rows = read_table(folder)
    if len(rows) != sum(len(t["tokens"]) for t in tensors.values()):
        raise RuntimeError("molecules.csv and the tensors disagree on the number of ILs")
    for row in rows:
        index = int(row["row"])
        text = decode(tensors[row["split"]]["tokens"][index], vocab)
        if text != row["smiles"]:
            raise RuntimeError(f"{row['split']} row {index} decodes to {text}, expected {row['smiles']}")
    return len(rows)


def build_il_strings(
    config: MoleculeConfig = MoleculeConfig(),
    l_max: int = L_MAX,
    source: Path = COLLECTED_ILS,
    out: Path = IL_STRINGS_DIR,
) -> Path:
    """Run the whole pipeline and return the output folder."""
    molecules = load_molecules(config, source)
    print(f"[strings] {molecules.num_source_rows:,} rows in {source.name}")
    smiles = [il.smiles for il in molecules.ils()]
    lengths = [len(text) for text in smiles]
    too_long = sum(length > l_max for length in lengths)
    if too_long:
        raise RuntimeError(
            f"{too_long} ILs have more than l_max = {l_max} characters (the longest has {max(lengths)}). Raise l_max: "
            "dropping them here would give the string model a different training set from the graph model."
        )
    vocab = derive_vocab(smiles, l_max)

    out.mkdir(parents=True, exist_ok=True)
    vocab.save(out / "vocab.json")
    tensors = {split: to_tensors([il.smiles for il in molecules.ils(split)], vocab) for split in SPLITS}
    for split in SPLITS:
        torch.save(tensors[split], out / f"{split}.pt")
    molecules.write_tables(out)

    marginals = class_marginals(tensors["train"], vocab)
    missing = [label for label, p in zip(vocab.labels(), marginals["char"]) if p == 0]
    if missing:
        raise RuntimeError(
            f"characters {missing} never occur in the train split, so their noise probability would be zero. "
            "Use another seed, and rebuild the graphs with the same one."
        )
    in_ils = Counter(char for text in smiles for char in set(text))
    chars = Counter(char for text in smiles for char in text)
    stats = {
        **molecules.summary(),
        "l_max": l_max,
        "chars_per_il": {
            "min": min(lengths),
            "median": statistics.median(lengths),
            "mean": round(statistics.mean(lengths), 2),
            "max": max(lengths),
        },
        "char_classes": [
            {"index": vocab.index(char), "label": char, "ils": in_ils[char], "chars": chars[char]} for char in vocab.chars[1:]
        ],
        "train_marginals": marginals,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    checked = verify(out)
    print(molecules.describe())
    print(f"  {vocab.num_classes} character classes: {' '.join(vocab.labels())}")
    print(f"  length {min(lengths)} to {max(lengths)} characters, padded to {l_max}")
    print(f"  verified: all {checked:,} saved rows decode back to their SMILES")
    return out
