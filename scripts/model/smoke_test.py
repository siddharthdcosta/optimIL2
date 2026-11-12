"""Build the crude diffusion model with random weights and check it.

Usage:
    python scripts/model/smoke_test.py                 # CPU, and MPS if available
    python scripts/model/smoke_test.py --device cpu

Needs data/processed/il_graphs/ (run scripts/data/build_graphs.py first).
Nothing is trained or saved. Runs forward passes on real ILs, one backward
pass, and full sampling; takes about a minute. What each check means is
explained in src/optimil2/model_primitive/checks.py.
"""

import argparse
import time

import torch

from optimil2.data.graph_dataset import ILGraphs
from optimil2.data.processing.il_graphs import IL_GRAPHS_DIR
from optimil2.data.processing.il_molecules import split_smiles
from optimil2.model_primitive import checks
from optimil2.model_primitive.diffusion import build_model


def run(device: str, batch_size: int) -> None:
    torch.manual_seed(0)
    data = ILGraphs("train")
    train_smiles = split_smiles(IL_GRAPHS_DIR, "train")
    model = build_model().to(device)
    x0 = torch.stack([data[i][0] for i in range(batch_size)]).to(device)
    e0 = torch.stack([data[i][1] for i in range(batch_size)]).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"\n[{device}] {parameters:,} parameters, {model.num_steps} steps, batch of {batch_size} ILs with {model.num_slots} slots")

    steps = [
        ("noise math", lambda: checks.check_noise_math(model.atom_marginals.cpu())),
        ("add noise", lambda: checks.check_add_noise(model, x0, e0)),
        ("denoiser", lambda: checks.check_denoiser(model, x0, e0)),
        ("loss", lambda: checks.check_loss(model, x0, e0)),
        ("reverse process", lambda: checks.check_reverse_process(model, x0, e0)),
        ("sampling", lambda: checks.check_sampling(model, data.vocab, train_smiles)),
    ]
    for name, check in steps:
        start = time.perf_counter()
        summary = check()
        print(f"  ok  {name:16s} {summary}  ({time.perf_counter() - start:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=["cpu", "mps"], help="default: cpu, then mps if available")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    devices = [args.device] if args.device else ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for device in devices:
        run(device, args.batch_size)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
