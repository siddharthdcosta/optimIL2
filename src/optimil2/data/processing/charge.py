"""Keep only singly charged ions: cations at exactly +1, anions at exactly -1.

Reads the PubChem ions in data/raw/pubchem_ions/ and writes the ions that
pass to data/processed/pubchem_ions/. SMILES are copied unchanged; RDKit is
only used to read the net charge. SMILES that RDKit cannot parse are dropped.
"""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rdkit import Chem, RDLogger

from optimil2.data.raw.ilgen import PUBCHEM_IONS_DIR
from optimil2.paths import PROCESSED_DATA

PUBCHEM_IONS_OUT = PROCESSED_DATA / "pubchem_ions"


@dataclass
class FilterCounts:
    total: int = 0
    kept: int = 0
    wrong_charge: int = 0
    unparsable: int = 0


def net_charge(smiles: str) -> int | None:
    """Net formal charge of a SMILES, or None if RDKit cannot parse it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.GetFormalCharge(mol)


def filter_by_charge(src: Path, dest: Path, charge: int) -> FilterCounts:
    """Copy rows of src whose SMILES has net charge equal to charge into dest.

    Both files are CSVs with a "smiles" column.
    """
    counts = FilterCounts()
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    with open(src, newline="") as fin, open(partial, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["smiles"])
        for row in csv.DictReader(fin):
            counts.total += 1
            value = net_charge(row["smiles"])
            if value is None:
                counts.unparsable += 1
            elif value != charge:
                counts.wrong_charge += 1
            else:
                counts.kept += 1
                writer.writerow([row["smiles"]])
    partial.replace(dest)
    return counts


def filter_pubchem_ions() -> Path:
    """Write +1 cations and -1 anions from the PubChem ions, plus a summary."""
    RDLogger.DisableLog("rdApp.*")  # silence per-molecule parse warnings
    jobs = {
        "pubchem_cations.csv": +1,
        "pubchem_anions.csv": -1,
    }
    summary = {}
    for name, charge in jobs.items():
        print(f"[filter] {name}: keeping net charge {charge:+d}")
        counts = filter_by_charge(PUBCHEM_IONS_DIR / name, PUBCHEM_IONS_OUT / name, charge)
        print(f"  kept {counts.kept:,} of {counts.total:,}")
        summary[name] = {"source": str(Path("data/raw/pubchem_ions") / name), "charge": charge, **asdict(counts)}
    (PUBCHEM_IONS_OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return PUBCHEM_IONS_OUT
