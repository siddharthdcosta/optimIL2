"""What does a heavy-atom graph lose on the real ILs, and how should it be built?

Question: before writing the SMILES -> graph converter, which choices does the
data force? Specifically, for data/raw/ilgen/collected_ils.csv:

  1. What do the SMILES contain that a graph of (element, formal charge) nodes
     and bond-order edges cannot hold (isotopes, stereo, radicals)?
  2. Hydrogens are dropped. Can RDKit put them back from element, charge and
     bond orders alone? Compared for two bond encodings: Kekule bonds
     (single/double/triple) and an extra "aromatic" bond class.
  3. How many entries are the same IL written twice (after dropping stereo,
     and as two resonance forms)?
  4. How many molecules contain each (element, charge) class, to pick a
     rare-class cutoff, and how many fit under each candidate N_max?

Usage:
    python scripts/data/ilgen.py                      # once, to download
    python investigate/il_graph_representation.py

The answers are summarised in the docstring of
src/optimil2/data/processing/il_graphs.py, which is where the decisions live.
"""

import csv
from collections import Counter, defaultdict

from rdkit import Chem, RDLogger

from optimil2.data.raw.ilgen import ILGEN_DIR

N_MAX_CANDIDATES = (40, 48, 56, 60, 64, 72, 80)


def strip_labels(mol: Chem.Mol) -> Chem.Mol:
    """Drop isotope labels (deuterium becomes ordinary H) and stereo marks."""
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    mol = Chem.RemoveHs(mol)
    Chem.RemoveStereochemistry(mol)
    return mol


def rebuild_without_h(mol: Chem.Mol, kekulize: bool) -> str | None:
    """Rebuild mol from element, charge and bond type only; None if RDKit rejects it."""
    mol = Chem.Mol(mol)
    if kekulize:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    out = Chem.RWMol()
    for atom in mol.GetAtoms():
        new = Chem.Atom(atom.GetAtomicNum())
        new.SetFormalCharge(atom.GetFormalCharge())
        new.SetIsAromatic(atom.GetIsAromatic())
        out.AddAtom(new)
    for bond in mol.GetBonds():
        out.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
        out.GetBondBetweenAtoms(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()).SetIsAromatic(bond.GetIsAromatic())
    out = out.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return Chem.MolToSmiles(out)


def skeleton_key(mol: Chem.Mol) -> str:
    """Key that ignores bond orders and where charges sit.

    Two resonance forms (or tautomers) of one species share it. It can also
    merge true isomers that differ only in where a double bond is.
    """
    hydrogens = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
    charge = Chem.GetFormalCharge(mol)
    flat = Chem.RWMol(mol)
    for atom in flat.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
        atom.SetNumRadicalElectrons(0)
    for bond in flat.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    return f"{Chem.MolToSmiles(flat.GetMol())}|q={charge:+d}|H={hydrogens}"


def resonance_key(mol: Chem.Mol) -> str:
    """Smallest canonical SMILES over RDKit's resonance structures, per fragment."""
    parts = []
    for frag in Chem.GetMolFrags(mol, asMols=True):
        forms = {Chem.MolToSmiles(form) for form in Chem.ResonanceMolSupplier(frag) if form is not None}
        parts.append(min(forms) if forms else Chem.MolToSmiles(frag))
    return ".".join(sorted(parts))


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    with open(ILGEN_DIR / "collected_ils.csv", newline="") as f:
        raw = [row["smiles"] for row in csv.DictReader(f)]
    mols = [Chem.MolFromSmiles(s) for s in raw]
    print(f"rows {len(raw):,}; unparsable {sum(m is None for m in mols)}")
    mols = [m for m in mols if m is not None]

    # 1. What the graph cannot hold.
    def count(pred) -> int:
        return sum(any(pred(a) for a in m.GetAtoms()) for m in mols)

    print("\n[1] molecules containing something a heavy-atom graph drops")
    print(f"  isotope labels       {count(lambda a: a.GetIsotope() != 0):5d}")
    print(f"  tetrahedral stereo   {count(lambda a: a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED):5d}")
    print(f"  radical electrons    {count(lambda a: a.GetNumRadicalElectrons() > 0):5d}")
    radical_kinds = Counter(
        (a.GetSymbol(), a.GetFormalCharge())
        for m in mols
        for a in m.GetAtoms()
        if a.GetNumRadicalElectrons()
    )
    print(f"    radical atoms by (element, charge): {dict(radical_kinds)}")

    clean = [strip_labels(m) for m in mols]
    target = [Chem.MolToSmiles(m) for m in clean]

    # 2. Can hydrogens be recovered?
    print("\n[2] rebuild from (element, charge, bond type) with hydrogens inferred by RDKit")
    for name, kekulize in (("Kekule bonds", True), ("aromatic bond class", False)):
        rebuilt = [rebuild_without_h(m, kekulize) for m in clean]
        rejected = sum(r is None for r in rebuilt)
        changed = [(t, r) for t, r in zip(target, rebuilt) if r is not None and r != t]
        radical_changed = sum(any(a.GetNumRadicalElectrons() for a in m.GetAtoms()) for m, t, r in zip(clean, target, rebuilt) if r != t)
        print(f"  {name:20s} identical {len(clean) - rejected - len(changed):5d}   rejected {rejected:4d}   changed {len(changed):4d}"
              f"   (of the non-identical, {radical_changed} have radicals)")
        for t, r in changed[:3]:
            print(f"      {t}  ->  {r}")

    # 3. Duplicates.
    print("\n[3] duplicates")
    exact = Counter(target)
    print(f"  distinct after dropping stereo/isotopes: {len(exact):,} ({len(target) - len(exact)} duplicate rows)")
    by_skeleton, by_resonance = defaultdict(set), defaultdict(set)
    for mol, smiles in zip(clean, target):
        by_skeleton[skeleton_key(mol)].add(smiles)
        by_resonance[resonance_key(mol)].add(smiles)
    print(f"  distinct by skeleton key:  {len(by_skeleton):,}")
    print(f"  distinct by resonance key: {len(by_resonance):,}")
    resonance_of = {s: key for key, group in by_resonance.items() for s in group}
    split = [group for group in by_skeleton.values() if len({resonance_of[s] for s in group}) > 1]
    print(f"  skeleton groups holding more than one resonance key (isomers the skeleton key merges): {len(split)}")
    for group in split[:5]:
        print(f"      {sorted(group)}")

    # 4. Class frequencies and sizes, on charge-balanced entries only.
    balanced = [m for m in clean if sorted(Chem.GetFormalCharge(f) for f in Chem.GetMolFrags(m, asMols=True)) == [-1, 1]]
    print(f"\n[4] one +1 ion and one -1 ion: {len(balanced):,} of {len(clean):,}")
    in_molecules = Counter()
    for m in balanced:
        in_molecules.update({(a.GetSymbol(), a.GetFormalCharge()) for a in m.GetAtoms()})
    print("  molecules containing each (element, charge) class:")
    for (symbol, charge), n in sorted(in_molecules.items(), key=lambda kv: -kv[1]):
        print(f"    {symbol:2s} {charge:+d}  {n:5d}")
    sizes = [m.GetNumAtoms() for m in balanced]
    for n_max in N_MAX_CANDIDATES:
        fit = sum(s <= n_max for s in sizes)
        print(f"  heavy atoms <= {n_max}: {fit:,} ({100 * fit / len(sizes):.2f}%)")


if __name__ == "__main__":
    main()
