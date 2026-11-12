"""Turn the real ILs into character-token tensors with a train/val split.

Usage:
    python scripts/data/build_strings.py
    python scripts/data/build_strings.py --l-max 160

Needs data/raw/ilgen/ (run scripts/data/ilgen.py first). Writes
data/processed/il_strings/: train.pt, val.pt, vocab.json, stats.json,
molecules.csv and dropped.csv. Takes about 15 seconds. Safe to re-run: the
output is overwritten and is identical for the same options.

The ILs and the split are the same as those of scripts/data/build_graphs.py
when both are run with the same options, which is what a comparison of the
string and graph models needs. Those shared options are explained in
src/optimil2/data/processing/il_molecules.py, and the string format in
src/optimil2/data/processing/il_strings.py.
"""

import argparse

from optimil2.data.processing.il_graphs import IL_GRAPHS_DIR
from optimil2.data.processing.il_molecules import add_molecule_arguments, config_from_arguments, same_molecules
from optimil2.data.processing.il_strings import L_MAX, build_il_strings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_molecule_arguments(parser)
    parser.add_argument("--l-max", type=int, default=L_MAX, help="positions per string; the build fails if an IL is longer")
    args = parser.parse_args()

    folder = build_il_strings(config_from_arguments(args), args.l_max)
    print(f"Saved to {folder}")
    if (IL_GRAPHS_DIR / "molecules.csv").exists():
        if same_molecules(folder, IL_GRAPHS_DIR):
            print(f"Same ILs, split and row order as {IL_GRAPHS_DIR.name}/")
        else:
            print(f"WARNING: {IL_GRAPHS_DIR.name}/ holds different ILs or a different split. "
                  "Rebuild both with the same options before comparing models.")


if __name__ == "__main__":
    main()
