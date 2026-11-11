"""How many organic salts with a measured melting point can we get?

Question: is there enough melting-point data on organic salts in general
(not only known ionic liquids) to train a generator conditioned on melting
point?

Sources checked:
  1. ILGen IL melting points      (data/raw/ilgen/il_melting_points.csv)
  2. ILGen general salt set       (data/raw/ilgen/general_melting_points.csv)
  3. PubChem experimental values  (fetched here, cached in data/investigate/)

Usage:
    python scripts/data/ilgen.py                  # once, for sources 1 and 2
    python investigate/salt_melting_points.py

A "salt" here is a SMILES with at least one positively charged fragment and
one negatively charged fragment, and net charge zero. Salts that PubChem
writes as neutral acid + base (for example an amine plus HCl) are counted
separately, because their SMILES carry no charges.
"""

import csv
import json
import re
import statistics
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

from rdkit import Chem, RDLogger

from optimil2.data.raw.ilgen import ILGEN_DIR
from optimil2.paths import DATA

CACHE = DATA / "investigate" / "salt_melting_points"
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest"
LIQUID_BELOW_K = 373.15  # the usual ionic liquid cutoff, 100 °C
SMILES_BATCH = 200


# ---------- classifying a SMILES ----------

def classify(smiles: str) -> tuple[str, str | None]:
    """Return (kind, canonical SMILES).

    kind is one of:
      organic cation       salt whose cation contains carbon (IL-like)
      inorganic cation     salt with a metal or ammonium cation and an organic anion
      inorganic salt       salt with no carbon in any charged fragment
      neutral mixture      several uncharged fragments (acid + base, hydrate, ...)
      single molecule      one fragment
      unparsable
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unparsable", None
    canonical = Chem.MolToSmiles(mol)
    fragments = Chem.GetMolFrags(mol, asMols=True)
    if len(fragments) < 2:
        return "single molecule", canonical

    cations = [f for f in fragments if Chem.GetFormalCharge(f) > 0]
    anions = [f for f in fragments if Chem.GetFormalCharge(f) < 0]
    if not cations or not anions or Chem.GetFormalCharge(mol) != 0:
        return "neutral mixture", canonical

    def has_carbon(fragment) -> bool:
        return any(atom.GetSymbol() == "C" for atom in fragment.GetAtoms())

    if any(has_carbon(f) for f in cations):
        return "organic cation", canonical
    if any(has_carbon(f) for f in anions):
        return "inorganic cation", canonical
    return "inorganic salt", canonical


def is_one_to_one(canonical: str) -> bool:
    """True for exactly one +1 fragment and one -1 fragment."""
    fragments = Chem.GetMolFrags(Chem.MolFromSmiles(canonical), asMols=True)
    return sorted(Chem.GetFormalCharge(f) for f in fragments) == [-1, 1]


# ---------- local sources ----------

def read_csv_melting_points(name: str, smiles_col: str, mp_col: str) -> dict[str, float]:
    """SMILES -> melting point in kelvin, from a file in data/raw/ilgen/."""
    path = ILGEN_DIR / name
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run: python scripts/data/ilgen.py")
    with open(path, newline="") as file:
        return {row[smiles_col]: float(row[mp_col]) for row in csv.DictReader(file)}


# ---------- PubChem ----------

def fetch_json(url: str, data: bytes | None = None) -> dict:
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=120) as response:
                return json.load(response)
        except Exception as error:  # PubChem returns occasional 503s under load
            if attempt == 4:
                raise
            print(f"  retrying after error: {error}")
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def pubchem_annotations() -> list[dict]:
    """Every PubChem compound annotation under the heading 'Melting Point'."""
    cache = CACHE / "pubchem_melting_point_annotations.json"
    if cache.exists():
        return json.loads(cache.read_text())
    base = f"{PUBCHEM}/pug_view/annotations/heading/JSON?heading=Melting%20Point&heading_type=Compound"
    annotations, page, total_pages = [], 1, 1
    while page <= total_pages:
        block = fetch_json(f"{base}&page={page}")["Annotations"]
        total_pages = block["TotalPages"]
        annotations.extend(block["Annotation"])
        print(f"  PubChem annotations: page {page} of {total_pages}")
        page += 1
        time.sleep(0.3)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(annotations))
    return annotations


NUMBER = r"-?\d+(?:\.\d+)?"
TEMPERATURE = re.compile(rf"({NUMBER})(?:\s*(?:-|to|–)\s*({NUMBER}))?\s*°?\s*(C|F|K)\b")


def parse_kelvin(text: str) -> float | None:
    """First temperature in a free-text melting point, in kelvin.

    A range such as '120-122 °C' gives its midpoint. Returns None when no
    temperature with a unit is found.
    """
    match = TEMPERATURE.search(text)
    if match is None:
        return None
    low, high, unit = match.groups()
    value = float(low)
    if high is not None and float(high) > value:  # guards against '5-10' read as 5 and -10
        value = (value + float(high)) / 2
    if unit == "C":
        value += 273.15
    elif unit == "F":
        value = (value - 32) * 5 / 9 + 273.15
    if not 50 < value < 3500:
        return None
    return value


def pubchem_melting_points() -> tuple[dict[int, float], int]:
    """CID -> median parsed melting point in kelvin, plus the count of unparsed values."""
    values, unparsed = defaultdict(list), 0
    for annotation in pubchem_annotations():
        cids = annotation.get("LinkedRecords", {}).get("CID", [])
        for item in annotation.get("Data", []):
            value = item.get("Value", {})
            texts = [part.get("String", "") for part in value.get("StringWithMarkup", [])]
            if "Number" in value:
                texts.append(f"{value['Number'][0]} {value.get('Unit', '')}")
            for text in texts:
                kelvin = parse_kelvin(text)
                if kelvin is None:
                    unparsed += 1
                    continue
                for cid in cids:
                    values[cid].append(kelvin)
    return {cid: statistics.median(v) for cid, v in values.items()}, unparsed


def pubchem_smiles(cids: list[int]) -> dict[int, str]:
    """CID -> SMILES, looked up in batches."""
    cache = CACHE / "pubchem_smiles.json"
    known = {int(k): v for k, v in json.loads(cache.read_text()).items()} if cache.exists() else {}
    missing = [cid for cid in cids if cid not in known]
    url = f"{PUBCHEM}/pug/compound/cid/property/SMILES/JSON"
    for start in range(0, len(missing), SMILES_BATCH):
        batch = missing[start:start + SMILES_BATCH]
        body = urllib.parse.urlencode({"cid": ",".join(map(str, batch))}).encode()
        for row in fetch_json(url, body)["PropertyTable"]["Properties"]:
            smiles = next((v for k, v in row.items() if k != "CID"), None)
            if smiles:
                known[row["CID"]] = smiles
        print(f"  PubChem SMILES: {min(start + SMILES_BATCH, len(missing)):,} of {len(missing):,}")
        time.sleep(0.3)
    if missing:
        CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known))
    return known


# ---------- report ----------

def summarise(name: str, melting_points: dict[str, float]) -> dict[str, tuple[str, float]]:
    """Print counts for one source. Returns canonical SMILES -> (kind, kelvin) for salts."""
    kinds = Counter()
    salts: dict[str, tuple[str, float]] = {}
    for smiles, kelvin in melting_points.items():
        kind, canonical = classify(smiles)
        kinds[kind] += 1
        if kind in ("organic cation", "inorganic cation"):
            salts[canonical] = (kind, kelvin)

    print(f"\n=== {name} ===")
    print(f"entries with a melting point: {len(melting_points):,}")
    for kind, count in kinds.most_common():
        print(f"  {kind:18s} {count:>7,}")
    describe("organic-cation salts", [k for kind, k in salts.values() if kind == "organic cation"])
    describe("inorganic-cation salts", [k for kind, k in salts.values() if kind == "inorganic cation"])
    return salts


def describe(label: str, kelvins: list[float]) -> None:
    if not kelvins:
        print(f"  {label}: none")
        return
    below = sum(k < LIQUID_BELOW_K for k in kelvins)
    quartiles = statistics.quantiles(kelvins, n=4) if len(kelvins) > 1 else [kelvins[0]] * 3
    print(
        f"  {label}: n={len(kelvins):,}; melting below 100 °C: {below:,} ({100 * below / len(kelvins):.0f}%); "
        f"quartiles {quartiles[0] - 273.15:.0f} / {quartiles[1] - 273.15:.0f} / {quartiles[2] - 273.15:.0f} °C"
    )


def main() -> None:
    RDLogger.DisableLog("rdApp.*")

    il = read_csv_melting_points("il_melting_points.csv", "smiles", "Melting point (K)")
    general = read_csv_melting_points("general_melting_points.csv", "smiles", "mpK")

    print("Fetching PubChem melting points (cached after the first run)...")
    by_cid, unparsed = pubchem_melting_points()
    smiles_by_cid = pubchem_smiles(sorted(by_cid))
    pubchem = {smiles_by_cid[cid]: kelvin for cid, kelvin in by_cid.items() if cid in smiles_by_cid}
    print(f"PubChem: {len(by_cid):,} compounds with a parsed value; {unparsed:,} values could not be parsed")

    sources = {
        "ILGen IL melting points": summarise("ILGen IL melting points", il),
        "ILGen general salt melting points": summarise("ILGen general salt melting points", general),
        "PubChem experimental melting points": summarise("PubChem experimental melting points", pubchem),
    }

    union: dict[str, tuple[str, float]] = {}
    for salts in sources.values():
        for canonical, entry in salts.items():
            union.setdefault(canonical, entry)
    organic = {s: k for s, (kind, k) in union.items() if kind == "organic cation"}
    one_to_one = [k for s, k in organic.items() if is_one_to_one(s)]

    print("\n=== All sources combined, duplicates removed ===")
    print(f"salts with a melting point: {len(union):,}")
    describe("organic-cation salts", list(organic.values()))
    describe("...of which one +1 cation and one -1 anion", one_to_one)
    names = list(sources)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            print(f"  overlap, {a} and {b}: {len(sources[a].keys() & sources[b].keys()):,}")


if __name__ == "__main__":
    main()
