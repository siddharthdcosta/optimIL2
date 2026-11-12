"""Turn the real ILs into graph tensors with a train/val split.

Usage:
    python scripts/data/build_graphs.py
    python scripts/data/build_graphs.py --n-max 64 --min-class-molecules 20

Needs data/raw/ilgen/ (run scripts/data/ilgen.py first). Writes
data/processed/il_graphs/: train.pt, val.pt, vocab.json, stats.json,
molecules.csv and dropped.csv. Takes about 15 seconds. Safe to re-run: the
output is overwritten and is identical for the same options.

What each option trades off is explained in
src/optimil2/data/processing/il_graphs.py.
"""

import argparse

from optimil2.data.processing.il_graphs import GraphConfig, build_il_graphs


def main() -> None:
    defaults = GraphConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-max", type=int, default=defaults.n_max, help="atom slots per graph; larger ILs are dropped")
    parser.add_argument("--min-class-molecules", type=int, default=defaults.min_class_molecules,
                        help="ILs an atom class must occur in to enter the vocabulary")
    parser.add_argument("--val-fraction", type=float, default=defaults.val_fraction, help="share of ILs held out")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="decides which cations are held out")
    args = parser.parse_args()

    folder = build_il_graphs(GraphConfig(args.n_max, args.min_class_molecules, args.val_fraction, args.seed))
    print(f"Saved to {folder}")


if __name__ == "__main__":
    main()
