"""Load the IL graph tensors written by scripts/data/build_graphs.py."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from optimil2.data.processing.il_graphs import IL_GRAPHS_DIR, SPLITS
from optimil2.data.processing.molgraph import Vocab


class ILGraphs(Dataset):
    """One split ("train" or "val") of the IL graphs.

    Item i is (x, e): x holds the atom class of each of the n_max slots and e
    the bond class of each slot pair, shapes (n_max,) and (n_max, n_max).
    Both are int64, which is what indexing and one-hot encoding need. The
    default DataLoader collation stacks them to (B, n_max) and
    (B, n_max, n_max). The format is described in data/processing/molgraph.py.
    """

    def __init__(self, split: str = "train", folder: Path = IL_GRAPHS_DIR):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        path = folder / f"{split}.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: python scripts/data/build_graphs.py")
        tensors = torch.load(path)
        self.x = tensors["x"]  # uint8, converted per item to keep memory small
        self.e = tensors["e"]
        self.vocab = Vocab.load(folder / "vocab.json")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index].long(), self.e[index].long()


def train_marginals(folder: Path = IL_GRAPHS_DIR) -> tuple[torch.Tensor, torch.Tensor]:
    """Class frequencies of the train split: (atom classes, bond classes).

    Measured over all slots and all slot pairs, so index 0 (EMPTY, NO_BOND)
    is included. Each sums to 1.
    """
    marginals = json.loads((folder / "stats.json").read_text())["train_marginals"]
    return torch.tensor(marginals["atom"]), torch.tensor(marginals["bond"])
