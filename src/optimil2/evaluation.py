"""Score generated molecules: validity, uniqueness, novelty.

`evaluate` takes samples as SMILES and nothing else, so it scores any model
the same way. A string model's output is already a SMILES
(molstring.decode); a graph model's is turned into one first
(molgraph.to_smiles).

    validity     share of samples that RDKit parses and that have net charge 0
    uniqueness   share of the valid samples that are distinct
    novelty      share of the distinct valid samples that are not in the
                 training set
    ion_pair     share of samples that are an ionic liquid as the training
                 data defines one: one +1 ion and one -1 ion, no radicals

Decisions:
- A sample is a SMILES, or None for an output that has no SMILES at all (a
  graph that RDKit cannot sanitize). None counts as invalid.
- A sample with no atoms counts as invalid. RDKit parses the empty string
  without complaint, as a molecule with no atoms and net charge 0, so by the
  letter of the definition it would be valid. An untrained or collapsed
  model can produce it: a string of only PAD, a graph of only EMPTY slots.
- Two samples are the same molecule if their canonical SMILES are equal
  after dropping stereo and isotope labels. This is the rule the data
  pipeline uses to remove duplicates (molgraph.read_mol), so a sample that
  equals a training IL is always recognised, however it is written. The
  training SMILES passed in go through the same function.
- Two resonance forms of one IL count as different molecules here, as they
  do in the data (see molgraph.py). So novelty can count a training IL drawn
  with its charge on another atom as new.
- Validity is lenient: a neutral molecule that is no salt ("CCO"), three
  ions, or a radical ("[N]") all pass. `ion_pair` is the strict version,
  and the one that says whether the model makes ionic liquids. It is a share
  of all samples and is reported next to the other three; uniqueness and
  novelty are still measured on the valid samples.
- With no valid sample, uniqueness and novelty are undefined and reported as
  None, not as 0.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from rdkit import Chem, rdBase

from optimil2.data.processing.molgraph import Rejected, il_from_mol, read_mol


@dataclass(frozen=True)
class Metrics:
    num_samples: int
    validity: float
    uniqueness: float | None  # None if no sample is valid
    novelty: float | None  # None if no sample is valid
    ion_pair: float

    def __str__(self) -> str:
        def percent(value: float | None) -> str:
            return "n/a" if value is None else f"{100 * value:.1f}%"

        return (
            f"valid {percent(self.validity)}, unique {percent(self.uniqueness)}, "
            f"novel {percent(self.novelty)}, ion pairs {percent(self.ion_pair)}"
        )


def _valid_mol(smiles: str | None) -> Chem.Mol | None:
    """The molecule of a valid sample (parses, has atoms, net charge 0), else None."""
    if not smiles:
        return None
    mol = read_mol(smiles)
    if mol is None or mol.GetNumAtoms() == 0 or Chem.GetFormalCharge(mol) != 0:
        return None
    return mol


def _is_ion_pair(mol: Chem.Mol) -> bool:
    try:
        il_from_mol(mol)
    except Rejected:
        return False
    return True


def canonical(smiles: str | None) -> str | None:
    """Canonical SMILES of a valid sample, or None if the sample is not valid."""
    with rdBase.BlockLogs():  # RDKit prints an error for every string it cannot parse
        mol = _valid_mol(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def evaluate(samples: Iterable[str | None], train: Iterable[str]) -> Metrics:
    """Score samples against the SMILES of the training set."""
    samples = list(samples)
    if not samples:
        raise ValueError("no samples to evaluate")
    with rdBase.BlockLogs():
        mols = [mol for mol in map(_valid_mol, samples) if mol is not None]
        ion_pairs = sum(_is_ion_pair(mol) for mol in mols)
    valid = [Chem.MolToSmiles(mol) for mol in mols]
    distinct = set(valid)
    known = {canonical(smiles) for smiles in train}
    return Metrics(
        num_samples=len(samples),
        validity=len(valid) / len(samples),
        uniqueness=len(distinct) / len(valid) if valid else None,
        novelty=len(distinct - known) / len(distinct) if distinct else None,
        ion_pair=ion_pairs / len(samples),
    )
