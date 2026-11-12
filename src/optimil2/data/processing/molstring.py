"""Convert an ionic liquid between a SMILES and a fixed-length row of character tokens.

This is the string twin of molgraph.py. An IL is the SMILES "cation.anion",
one token per character, padded on the right to l_max:

    tokens: (l_max,)   character class of each position; 0 (PAD) marks an
                       unused position

`encode` turns a SMILES into tokens and `decode` turns tokens back into a
SMILES. The class numbering lives in `CharVocab`. Nothing here needs RDKit:
whether a decoded string is a molecule is a question for
optimil2.evaluation.

Decisions. The numbers were measured on the 6,851 kept ILs.

- One token per character, the simplest tokenizer there is. There are 30
  characters. An atom written with two letters or in brackets takes several
  tokens ("Cl" is 2, "[N+]" is 4), so the model has to learn to spell atoms.
  Atom-level tokens ("Cl", "[N+]", ... as single tokens) would remove that
  burden but need 59 classes and a tokenizer that knows SMILES grammar; they
  would shorten the longest IL only from 121 to 115 tokens.
- The string is the canonical SMILES stored in IL.smiles: cation first, then
  ".", then anion, aromatic rings in lowercase, no stereo or isotope labels.
  It describes exactly the molecule the graph model is given. (The graph
  stores Kekule bonds and the string aromatic ones; each is the plainest
  form for its representation.)
- Each IL has one string. A molecule has many valid SMILES, and training on
  several of them is a common augmentation, but it is a training choice, not
  part of the format.
- PAD is an ordinary class, like EMPTY for the graph's atom slots. `decode`
  drops PAD wherever it appears, as graph decoding skips EMPTY slots, so
  every row of tokens decodes to some string and the length of a generated
  string is the number of positions that are not PAD.
"""

import json
from pathlib import Path

import numpy as np

PAD = 0  # character class of an unused position


class CharVocab:
    """Numbering of the characters, plus the number of positions.

    Index 0 is reserved for PAD. So `chars[k]` is the character with index k
    for k >= 1, and `chars[0]` is None.
    """

    def __init__(self, chars: list[str], l_max: int):
        self.chars: tuple[str | None, ...] = (None, *chars)
        self.l_max = l_max
        self._index = {char: i for i, char in enumerate(self.chars) if char is not None}

    @property
    def num_classes(self) -> int:
        """Number of character classes, counting PAD."""
        return len(self.chars)

    def index(self, char: str) -> int:
        return self._index[char]

    def labels(self) -> list[str]:
        return ["pad", *self.chars[1:]]

    def to_json(self) -> dict:
        return {"l_max": self.l_max, "char_classes": [{"index": i, "label": label} for i, label in enumerate(self.labels())]}

    @classmethod
    def from_json(cls, data: dict) -> "CharVocab":
        return cls([entry["label"] for entry in data["char_classes"][1:]], data["l_max"])

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "CharVocab":
        return cls.from_json(json.loads(path.read_text()))


def encode(smiles: str, vocab: CharVocab) -> np.ndarray:
    """Return the tokens of a SMILES, padded to vocab.l_max.

    Raises ValueError if the SMILES has more than l_max characters and
    KeyError if it has a character that is not in the vocabulary.
    """
    if len(smiles) > vocab.l_max:
        raise ValueError(f"{len(smiles)} characters do not fit in {vocab.l_max} positions")
    tokens = np.full(vocab.l_max, PAD, dtype=np.uint8)
    tokens[: len(smiles)] = [vocab.index(char) for char in smiles]
    return tokens


def decode(tokens, vocab: CharVocab) -> str:
    """Turn one row of tokens into a string: the characters in order, with every PAD skipped.

    tokens may be a numpy array or a torch tensor. The result can be any
    string, including the empty one; it need not be a valid SMILES.
    """
    if hasattr(tokens, "detach"):  # torch tensor, possibly on MPS
        tokens = tokens.detach().cpu().numpy()
    return "".join(vocab.chars[int(token)] for token in np.asarray(tokens) if token != PAD)
