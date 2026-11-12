"""Turn the real ILs into graph tensors with a train/val split.

Usage:
    python scripts/data/build_graphs.py
    python scripts/data/build_graphs.py --n-max 64 --min-class-molecules 20

Needs data/raw/ilgen/ (run scripts/data/ilgen.py first). Writes
data/processed/il_graphs/: train.pt, val.pt, vocab.json, stats.json,
molecules.csv and dropped.csv. Takes about 15 seconds. Safe to re-run: the
output is overwritten and is identical for the same options.

The options choose which ILs are kept and how they are split; what each one
trades off is explained in src/optimil2/data/processing/il_molecules.py.
scripts/data/build_strings.py takes the same options. To compare the graph
and string models, build both with the same ones.
"""

import argparse

from optimil2.data.processing.il_graphs import build_il_graphs
from optimil2.data.processing.il_molecules import add_molecule_arguments, config_from_arguments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_molecule_arguments(parser)
    args = parser.parse_args()

    folder = build_il_graphs(config_from_arguments(args))
    print(f"Saved to {folder}")


if __name__ == "__main__":
    main()
