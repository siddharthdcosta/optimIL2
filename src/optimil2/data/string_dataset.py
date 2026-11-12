"""Load the IL character-token tensors written by scripts/data/build_strings.py."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from optimil2.data.processing.il_molecules import SPLITS
from optimil2.data.processing.il_strings import IL_STRINGS_DIR
from optimil2.data.processing.molstring import CharVocab


class ILStrings(Dataset):
    """One split ("train" or "val") of the IL strings.

    Item i is the (l_max,) row of character classes of one IL, int64, which
    is what indexing and one-hot encoding need. The default DataLoader
    collation stacks them to (B, l_max). Item i is the same IL as item i of
    graph_dataset.ILGraphs for the same split. The format is described in
    data/processing/molstring.py.
    """

    def __init__(self, split: str = "train", folder: Path = IL_STRINGS_DIR):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        path = folder / f"{split}.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: python scripts/data/build_strings.py")
        self.tokens = torch.load(path)["tokens"]  # uint8, converted per item to keep memory small
        self.vocab = CharVocab.load(folder / "vocab.json")

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.tokens[index].long()


def train_marginals(folder: Path = IL_STRINGS_DIR) -> torch.Tensor:
    """Class frequencies of the train split, over all positions, so index 0 (PAD) is included. Sums to 1."""
    return torch.tensor(json.loads((folder / "stats.json").read_text())["train_marginals"]["char"])
