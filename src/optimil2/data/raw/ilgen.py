"""Download the datasets published in the two ILGen repos.

Two separate sets of files come from these repos:

- Real ionic liquids and their melting points, saved to data/raw/ilgen/.
- General charged molecules extracted from PubChem by the ILGen authors,
  saved to data/raw/pubchem_ions/. Most of these are not ionic liquid ions.

Files are fetched at fixed commits, so every download gives identical data.
The ILGen repos have no license: keep these files local, do not redistribute.
"""

from dataclasses import dataclass
from pathlib import Path

from optimil2.data.raw.download import download_file, write_manifest
from optimil2.paths import RAW_DATA

ILGEN_DIR = RAW_DATA / "ilgen"
PUBCHEM_IONS_DIR = RAW_DATA / "pubchem_ions"

ILGEN_OVERALL = "fate1997/ILGen-overall"
ILGEN_OVERALL_COMMIT = "4c79a0662db5cd0c11c8e27cb2b61f000c8dee1f"
ILGEN_ION = "fate1997/ILGen-ion"
ILGEN_ION_COMMIT = "c3af5752c403adad9d9a3c6499ae3cb8eac9f11f"


@dataclass(frozen=True)
class RemoteFile:
    repo: str
    commit: str
    path: str  # path inside the repo
    name: str  # file name to save as

    @property
    def url(self) -> str:
        return f"https://raw.githubusercontent.com/{self.repo}/{self.commit}/{self.path}"


# Real ILs collected from ILThermo and the literature, plus melting points.
# Left out: collected-ILs.txt (duplicate of the CSV) and
# exhausted-caion-anion-combinations.csv (93 MB, every cation paired with
# every anion; regenerate it from the two ion files if needed).
ILGEN_FILES = [
    RemoteFile(ILGEN_OVERALL, ILGEN_OVERALL_COMMIT, "database/collected-ILs.csv", "collected_ils.csv"),
    RemoteFile(ILGEN_OVERALL, ILGEN_OVERALL_COMMIT, "database/IL-cations.csv", "il_cations.csv"),
    RemoteFile(ILGEN_OVERALL, ILGEN_OVERALL_COMMIT, "database/IL-anions.csv", "il_anions.csv"),
    RemoteFile(ILGEN_OVERALL, ILGEN_OVERALL_COMMIT, "database/IL-melting-point.csv", "il_melting_points.csv"),
    # Virtual pairs predicted by ILGen's link-prediction model. Not real ILs.
    RemoteFile(ILGEN_OVERALL, ILGEN_OVERALL_COMMIT, "database/expanded-ILs.csv", "expanded_ils.csv"),
]

PUBCHEM_ION_FILES = [
    RemoteFile(ILGEN_ION, ILGEN_ION_COMMIT, "database/pubchem/pubchem_cations.csv", "pubchem_cations.csv"),
    RemoteFile(ILGEN_ION, ILGEN_ION_COMMIT, "database/pubchem/pubchem_anions.csv", "pubchem_anions.csv"),
]


def _download(files: list[RemoteFile], folder: Path, source: str) -> Path:
    for file in files:
        download_file(file.url, folder / file.name)
    write_manifest(folder, source, {file.name: file.url for file in files})
    return folder


def download_ilgen() -> Path:
    """Download the real IL structures, melting points and expanded pairs."""
    return _download(ILGEN_FILES, ILGEN_DIR, "ILGen-overall (github.com/fate1997/ILGen-overall)")


def download_pubchem_ions() -> Path:
    """Download the PubChem cations and anions prepared by the ILGen authors."""
    return _download(PUBCHEM_ION_FILES, PUBCHEM_IONS_DIR, "ILGen-ion (github.com/fate1997/ILGen-ion)")
