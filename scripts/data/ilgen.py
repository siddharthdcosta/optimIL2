"""Download the ILGen datasets into data/raw/.

Usage:
    python scripts/data/ilgen.py

Creates data/raw/ilgen/ (real ILs, about 15 MB) and data/raw/pubchem_ions/
(PubChem cations and anions, about 55 MB). Safe to re-run: files that are
already downloaded are skipped.
"""

from optimil2.data.raw.ilgen import download_ilgen, download_pubchem_ions


def main() -> None:
    for download in (download_ilgen, download_pubchem_ions):
        folder = download()
        print(f"Saved to {folder}")


if __name__ == "__main__":
    main()
