"""Keep only +1 cations and -1 anions from the PubChem ions.

Usage:
    python scripts/data/filter_charge.py

Needs data/raw/pubchem_ions/ (run scripts/data/ilgen.py first). Writes
data/processed/pubchem_ions/ with the filtered CSVs and a summary.json of
how many ions were kept and dropped. Takes a few minutes.
"""

from optimil2.data.processing.charge import filter_pubchem_ions


def main() -> None:
    folder = filter_pubchem_ions()
    print(f"Saved to {folder}")


if __name__ == "__main__":
    main()
