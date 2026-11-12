"""Build the string diffusion model with random weights and check it.

Usage:
    python scripts/model/smoke_test_string.py                 # CPU, and MPS if available
    python scripts/model/smoke_test_string.py --device cpu

The string twin of scripts/model/smoke_test.py. Needs both
data/processed/il_strings/ and data/processed/il_graphs/ (run
scripts/data/build_strings.py and scripts/data/build_graphs.py first): it
also checks that the two hold the same ILs. Nothing is trained or saved.
Runs forward passes on real ILs, one backward pass, and full sampling; takes
about a minute. What each check means is explained in
src/optimil2/model_primitive/string_checks.py.
"""

import argparse
import time

import torch
from rdkit import RDLogger

from optimil2.data.processing.il_molecules import split_smiles
from optimil2.data.processing.il_strings import IL_STRINGS_DIR
from optimil2.data.string_dataset import ILStrings
from optimil2.model_primitive import checks, string_checks
from optimil2.model_primitive.string_diffusion import build_string_model


def report(steps: list) -> None:
    for name, check in steps:
        start = time.perf_counter()
        summary = check()
        print(f"  ok  {name:16s} {summary}  ({time.perf_counter() - start:.1f}s)")


def run(device: str, batch_size: int, train_smiles: list[str]) -> None:
    torch.manual_seed(0)
    data = ILStrings("train")
    model = build_string_model().to(device)
    x0 = torch.stack([data[i] for i in range(batch_size)]).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"\n[{device}] {parameters:,} parameters, {model.num_steps} steps, batch of {batch_size} ILs with {model.length} positions")

    report([
        ("noise math", lambda: checks.check_noise_math(model.marginals.cpu())),
        ("add noise", lambda: string_checks.check_add_noise(model, x0)),
        ("denoiser", lambda: string_checks.check_denoiser(model, x0)),
        ("loss", lambda: string_checks.check_loss(model, x0)),
        ("reverse process", lambda: string_checks.check_reverse_process(model, x0)),
        ("sampling", lambda: string_checks.check_sampling(model, data.vocab, train_smiles)),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=["cpu", "mps"], help="default: cpu, then mps if available")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")  # untrained samples are invalid molecules; RDKit would warn about each
    train_smiles, val_smiles = split_smiles(IL_STRINGS_DIR, "train"), split_smiles(IL_STRINGS_DIR, "val")
    print("\n[comparison with the graph model]")
    report([
        ("same molecules", string_checks.check_same_molecules),
        ("metrics", lambda: string_checks.check_metrics(train_smiles, val_smiles)),
    ])
    devices = [args.device] if args.device else ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for device in devices:
        run(device, args.batch_size, train_smiles)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
