"""Convert an ionic liquid between a SMILES and a dense heavy-atom graph.

An ionic liquid (IL) is ONE graph with two disconnected components, the cation
and the anion. Padded to n_max atom slots, a graph is two integer arrays:

    x: (n_max,)         atom class of each slot; 0 (EMPTY) marks an unused slot
    e: (n_max, n_max)   bond class of each pair; 0 (NO_BOND) for no bond,
                        symmetric, zero diagonal

`read_il` parses a SMILES, `encode` turns it into (x, e), `decode` turns
(x, e) back into an RDKit molecule. The class numbering lives in `Vocab`.

Decisions. The numbers were measured on the 7,505 parsable ILGen ILs by
investigate/il_graph_representation.py.

- Atom class is the combined (element, formal charge) pair, for example "N+"
  or "O-", not two separate features. One categorical variable per node is
  the simplest thing to diffuse, and the model can only ever produce pairs
  that occur in the data.
- Hydrogens are not stored. When decoding, RDKit fills them in from the
  element, charge and bond orders. This is exact for every IL without
  radicals (7,452 of 7,452).
- Bonds are Kekule: single, double or triple, with no "aromatic" class. With
  an aromatic class, 72 ILs cannot be decoded, because the hydrogen on an
  aromatic nitrogen (the H in [nH]) is lost. The cost: a ring has more than
  one Kekule form and we store whichever RDKit picks, so alternating
  single/double bonds around a ring is something the model has to learn.
- Dropped because a graph of this kind cannot hold them: stereochemistry
  (in 819 ILs) and isotope labels (7 deuterated ILs, which become the
  ordinary compound).
- Radicals cannot be represented either, so `read_il` rejects them (53 ILs).
  Decoding would silently add a hydrogen instead. 8 are real nitroxide
  radicals; the other 45 are N+ or C- written without their hydrogens in the
  source and could be repaired later by accepting the hydrogen-completed form.
- Atom order is cation first, then anion, each in RDKit's canonical order.
  The order carries no meaning (the model is permutation-equivariant); it
  only makes the output reproducible and the Kekule form deterministic.
- Two resonance forms of one ion (for example the + on either nitrogen of an
  imidazolium) are different graphs and are both kept. `skeleton_key`
  ignores the difference and is used to keep such pairs in the same split.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem

EMPTY = 0  # atom class of an unused slot
NO_BOND = 0  # bond class of an unbonded pair

BOND_TYPES = {
    "single": Chem.BondType.SINGLE,
    "double": Chem.BondType.DOUBLE,
    "triple": Chem.BondType.TRIPLE,
}
_BOND_NAMES = {bond_type: name for name, bond_type in BOND_TYPES.items()}


@dataclass(frozen=True, order=True)
class AtomClass:
    """An element with a formal charge."""

    symbol: str
    charge: int

    @property
    def label(self) -> str:
        """Short name such as "C", "N+", "O-" or "Cl+3"."""
        if self.charge == 0:
            return self.symbol
        size = abs(self.charge)
        return f"{self.symbol}{'+' if self.charge > 0 else '-'}{size if size > 1 else ''}"


class Vocab:
    """Numbering of atom and bond classes, plus the number of atom slots.

    Index 0 is reserved in both lists: EMPTY for atoms, NO_BOND for bonds.
    So `atoms[k]` is the class with index k for k >= 1, and `atoms[0]` is None.
    """

    def __init__(self, atoms: list[AtomClass], bonds: list[str], n_max: int):
        self.atoms: tuple[AtomClass | None, ...] = (None, *atoms)
        self.bonds: tuple[str, ...] = ("none", *bonds)
        self.n_max = n_max
        self._atom_index = {atom: i for i, atom in enumerate(self.atoms) if atom is not None}
        self._bond_index = {name: i for i, name in enumerate(self.bonds)}

    @property
    def num_atom_classes(self) -> int:
        """Number of atom classes, counting EMPTY."""
        return len(self.atoms)

    @property
    def num_bond_classes(self) -> int:
        """Number of bond classes, counting NO_BOND."""
        return len(self.bonds)

    def atom_index(self, atom: AtomClass) -> int:
        return self._atom_index[atom]

    def bond_index(self, name: str) -> int:
        return self._bond_index[name]

    def atom_labels(self) -> list[str]:
        return ["empty"] + [atom.label for atom in self.atoms[1:]]

    def to_json(self) -> dict:
        atoms = [{"index": 0, "label": "empty"}]
        for i, atom in enumerate(self.atoms[1:], start=1):
            atoms.append({"index": i, "label": atom.label, "symbol": atom.symbol, "charge": atom.charge})
        bonds = [{"index": i, "label": name} for i, name in enumerate(self.bonds)]
        return {"n_max": self.n_max, "atom_classes": atoms, "bond_classes": bonds}

    @classmethod
    def from_json(cls, data: dict) -> "Vocab":
        atoms = [AtomClass(entry["symbol"], entry["charge"]) for entry in data["atom_classes"][1:]]
        bonds = [entry["label"] for entry in data["bond_classes"][1:]]
        return cls(atoms, bonds, data["n_max"])

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls.from_json(json.loads(path.read_text()))


class Rejected(ValueError):
    """A SMILES that cannot be turned into an IL graph. The message is a short reason code."""


@dataclass(frozen=True)
class IL:
    """One ionic liquid, cleaned and ready to encode."""

    smiles: str  # canonical "cation.anion", without stereo or isotope labels
    cation: str
    anion: str
    mol: Chem.Mol  # Kekule bonds; cation atoms first, then anion atoms

    @property
    def num_atoms(self) -> int:
        return self.mol.GetNumAtoms()

    def atom_classes(self) -> list[AtomClass]:
        return [_atom_class(atom) for atom in self.mol.GetAtoms()]

    def bond_names(self) -> list[str]:
        return [_BOND_NAMES[bond.GetBondType()] for bond in self.mol.GetBonds()]


def _atom_class(atom: Chem.Atom) -> AtomClass:
    return AtomClass(atom.GetSymbol(), atom.GetFormalCharge())


def _canonical_order(mol: Chem.Mol) -> Chem.Mol:
    """Renumber atoms into RDKit's canonical order."""
    ranks = list(Chem.CanonicalRankAtoms(mol))
    return Chem.RenumberAtoms(mol, sorted(range(mol.GetNumAtoms()), key=ranks.__getitem__))


def split_ions(mol: Chem.Mol) -> tuple[Chem.Mol, Chem.Mol] | None:
    """Return (cation, anion) if mol is exactly one +1 ion and one -1 ion, else None.

    This is also the check to run on a generated molecule: is it an ion pair?
    """
    ions = Chem.GetMolFrags(mol, asMols=True)
    charges = [Chem.GetFormalCharge(ion) for ion in ions]
    if sorted(charges) != [-1, 1]:
        return None
    return ions[charges.index(1)], ions[charges.index(-1)]


def il_from_mol(mol: Chem.Mol) -> IL:
    """Build an IL from a sanitized molecule that has no stereo, isotopes or explicit H atoms.

    Raises Rejected("not_one_cation_one_anion") or Rejected("radical").
    """
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        raise Rejected("radical")
    ions = split_ions(mol)
    if ions is None:
        raise Rejected("not_one_cation_one_anion")
    cation, anion = (_canonical_order(ion) for ion in ions)
    kekule = Chem.CombineMols(cation, anion)
    Chem.Kekulize(kekule, clearAromaticFlags=True)
    cation_smiles, anion_smiles = Chem.MolToSmiles(cation), Chem.MolToSmiles(anion)
    return IL(f"{cation_smiles}.{anion_smiles}", cation_smiles, anion_smiles, kekule)


def read_il(smiles: str) -> IL:
    """Parse a salt SMILES, dropping stereo and isotope labels.

    Raises Rejected with reason "unparsable", "radical" or
    "not_one_cation_one_anion".
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise Rejected("unparsable")
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    mol = Chem.RemoveHs(mol)  # deuterium atoms are ordinary hydrogens now
    Chem.RemoveStereochemistry(mol)
    return il_from_mol(mol)


def encode(il: IL, vocab: Vocab) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, e) for an IL, padded to vocab.n_max.

    Raises ValueError if the IL has more than n_max atoms and KeyError if it
    has an atom or bond class that is not in the vocabulary.
    """
    if il.num_atoms > vocab.n_max:
        raise ValueError(f"{il.num_atoms} atoms do not fit in {vocab.n_max} slots")
    x = np.full(vocab.n_max, EMPTY, dtype=np.uint8)
    e = np.full((vocab.n_max, vocab.n_max), NO_BOND, dtype=np.uint8)
    for atom in il.mol.GetAtoms():
        x[atom.GetIdx()] = vocab.atom_index(_atom_class(atom))
    for bond in il.mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        e[i, j] = e[j, i] = vocab.bond_index(_BOND_NAMES[bond.GetBondType()])
    return x, e


def decode(x, e, vocab: Vocab) -> Chem.Mol | None:
    """Turn (x, e) into a sanitized RDKit molecule, or None if RDKit rejects it.

    x and e may be numpy arrays or torch tensors. EMPTY slots are skipped, and
    so is any bond that touches one. Only the upper triangle of e is read.
    The result can be any molecule; use `split_ions` to test for an ion pair.
    """
    x, e = _as_array(x), _as_array(e)
    slots = np.flatnonzero(x != EMPTY)
    mol = Chem.RWMol()
    for slot in slots:
        atom_class = vocab.atoms[int(x[slot])]
        atom = Chem.Atom(atom_class.symbol)
        atom.SetFormalCharge(atom_class.charge)
        mol.AddAtom(atom)
    for a, i in enumerate(slots):
        for b in range(a + 1, len(slots)):
            bond = int(e[i, slots[b]])
            if bond != NO_BOND:
                mol.AddBond(a, b, BOND_TYPES[vocab.bonds[bond]])
    mol = mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Chem.MolSanitizeException:
        return None
    return mol


def decode_il(x, e, vocab: Vocab) -> IL | None:
    """Decode (x, e) into an IL, or None if it is not a valid ion pair."""
    mol = decode(x, e, vocab)
    if mol is None:
        return None
    try:
        return il_from_mol(mol)
    except Rejected:
        return None


def skeleton_key(mol: Chem.Mol) -> str:
    """Key that is the same for resonance forms and tautomers of one species.

    It keeps the heavy-atom connectivity, the elements, the net charge and the
    total hydrogen count, and ignores bond orders and where the charges sit.
    It can also merge true isomers that differ only in the position of a
    double bond, so use it to group molecules, not to deduplicate them.
    """
    hydrogens = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
    charge = Chem.GetFormalCharge(mol)
    flat = Chem.RWMol(mol)
    for atom in flat.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
    for bond in flat.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    return f"{Chem.MolToSmiles(flat.GetMol())}|q={charge:+d}|H={hydrogens}"


def _as_array(values) -> np.ndarray:
    if hasattr(values, "detach"):  # torch tensor, possibly on MPS
        values = values.detach().cpu().numpy()
    return np.asarray(values)
