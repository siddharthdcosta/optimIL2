"""Self-checks for the string diffusion model, run by scripts/model/smoke_test_string.py.

The twin of checks.py, check for check. The noise process is shared, so its
two tests are imported from there (`check_noise_math`, `known_x0_posterior`)
and run on the character marginals.

Nothing here trains. Each check raises AssertionError with a message when it
fails and otherwise returns a one-line summary. They cover what cannot be
seen by reading shapes alone:

- the transition matrices and the posterior, against brute-force arithmetic
  (checks.check_noise_math)
- the forward process, against the distribution it should produce
- the denoiser: shapes, that every position sees positions on both sides,
  that the order of the tokens matters, and that the strings of a batch do
  not affect each other
- the loss: value at initialisation, and that every weight gets a gradient
- the reverse process, with an "oracle" in place of the network (it always
  predicts the true clean string): every step must then be the exact
  posterior, and sampling must end exactly on that string
- sampling with the real, untrained network, end to end, scored with the
  metrics of optimil2.evaluation

Two more check the comparison with the graph model rather than the model:

- the string data and the graph data hold the same ILs, row for row
- the metrics give the known answer on the training and validation sets and
  on hand-written cases
"""

import math
from pathlib import Path

import torch
from torch import nn

from optimil2.data.graph_dataset import ILGraphs
from optimil2.data.processing.il_graphs import IL_GRAPHS_DIR
from optimil2.data.processing.il_molecules import SPLITS, same_molecules
from optimil2.data.processing.il_strings import IL_STRINGS_DIR
from optimil2.data.processing.molgraph import decode_il
from optimil2.data.processing.molstring import PAD, CharVocab, decode
from optimil2.data.string_dataset import ILStrings
from optimil2.evaluation import evaluate
from optimil2.model_primitive.checks import known_x0_posterior
from optimil2.model_primitive.noise import cosine_alpha_bar
from optimil2.model_primitive.string_diffusion import StringDiffusion


def check_add_noise(model: StringDiffusion, x0: torch.Tensor, repeats: int = 16) -> str:
    """Noised strings agree with x_0 as often as the schedule says.

    The batch is repeated to get enough variables for a tight statistical
    test: the tolerance is 5 standard errors, so a correct model never fails.
    """
    x0 = x0.repeat(repeats, 1)
    steps = torch.full((len(x0),), 0, device=x0.device)
    assert torch.equal(model.add_noise(x0, steps), x0), "step 0 must return the clean string"

    worst = 0.0
    for step in (1, model.num_steps // 2, model.num_steps):
        x_t = model.add_noise(x0, steps + step)
        assert x_t.shape == x0.shape and 0 <= x_t.min() and x_t.max() < model.marginals.numel(), "noisy tokens out of range"
        keep = model.alpha_bar[step].item()
        # A variable equals its clean value with probability keep + (1 - keep) * m[clean value].
        expected = (keep + (1 - keep) * model.marginals[x0]).mean().item()
        observed = (x_t == x0).float().mean().item()
        tolerance = 5 * math.sqrt(0.25 / x0.numel()) + 1e-4
        assert abs(observed - expected) < tolerance, f"step {step}: {observed:.4f} unchanged, expected {expected:.4f}"
        worst = max(worst, abs(observed - expected))
    return f"forward process matches the schedule (largest deviation {worst:.4f})"


def check_denoiser(model: StringDiffusion, x: torch.Tensor) -> str:
    """Output shape and finiteness, attention in both directions, dependence on order, independent strings.

    The graph check asserts that slot order does NOT matter. A string is its
    order, so here the opposite must hold.
    """
    batch, length = x.shape
    half = length // 2
    num_classes = model.marginals.numel()
    t = torch.randint(1, model.num_steps + 1, (batch,), device=x.device)
    other = (x + 1) % num_classes  # a different token at every position
    with torch.no_grad():
        logits = model(x, t)
        assert logits.shape == (batch, length, num_classes), f"logits have shape {tuple(logits.shape)}"
        assert logits.isfinite().all(), "logits are not finite"

        # Change only the second half: the first position must notice. Under a causal mask it could not.
        ahead = (model(torch.cat([x[:, :half], other[:, half:]], dim=1), t)[:, 0] - logits[:, 0]).abs().max().item()
        # Change only the first half: the last position must notice.
        behind = (model(torch.cat([other[:, :half], x[:, half:]], dim=1), t)[:, -1] - logits[:, -1]).abs().max().item()
        assert ahead > 1e-3, f"the first position does not see later ones (change {ahead})"
        assert behind > 1e-3, f"the last position does not see earlier ones (change {behind})"

        # Reverse the tokens: without position information the logits would simply be reversed too.
        order = (model(x.flip(1), t) - logits.flip(1)).abs().max().item()
        assert order > 1e-3, f"the output ignores the order of the tokens (change {order})"

        alone = (model(x[:1], t[:1]) - logits[:1]).abs().max().item()
    assert alone < 1e-3, f"a string's logits depend on the rest of the batch by {alone}"
    return (
        f"logits {tuple(logits.shape)}; sees both sides (changes {ahead:.2f}, {behind:.2f}), "
        f"order matters ({order:.2f}), batch independence error {alone:.0e}"
    )


def check_loss(model: StringDiffusion, x0: torch.Tensor) -> str:
    """The loss starts near log(number of classes), and every weight receives a gradient."""
    model.zero_grad()
    loss = model.loss(x0)["loss"]
    loss.backward()
    num_classes = model.marginals.numel()
    value, uniform = loss.item(), math.log(num_classes)
    assert 0.5 * uniform < value < 1.5 * uniform, f"loss {value:.3f} is far from log({num_classes}) = {uniform:.3f}"
    dead = [name for name, p in model.named_parameters() if p.grad is None or not p.grad.isfinite().all() or not p.grad.any()]
    assert not dead, f"parameters with no (or non-finite) gradient: {dead}"
    model.zero_grad()
    return f"loss {value:.3f} (log {num_classes} = {uniform:.3f}); all weights get gradients"


class Oracle(nn.Module):
    """Stands in for the denoiser and always predicts a known clean string with certainty."""

    def __init__(self, x0: torch.Tensor, num_classes: int):
        super().__init__()
        self.logits = 50.0 * nn.functional.one_hot(x0, num_classes).float()

    def forward(self, tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.logits


def check_reverse_process(model: StringDiffusion, x0: torch.Tensor, num_steps: int = 50) -> str:
    """With an oracle denoiser, each reverse step is the exact posterior and sampling ends on the clean string.

    If the network knows x_0, the distribution `reverse_probs` returns for a
    position must be q(x_{t-1} | x_t, x_0). That is recomputed by
    checks.known_x0_posterior for strings noised to step t. This tests how
    `reverse_probs` is wired (which matrices, which step), not only the
    posterior formula.
    """
    chain = StringDiffusion(Oracle(x0, model.marginals.numel()), model.marginals, model.length, num_steps).to(x0.device)
    alpha_bar = cosine_alpha_bar(num_steps)

    worst = 0.0
    for step in (1, 2, num_steps // 2, num_steps - 1, num_steps):
        x_t = chain.add_noise(x0, torch.full((len(x0),), step, device=x0.device))
        probs = chain.reverse_probs(x_t, step)
        error = (probs.cpu().double() - known_x0_posterior(x_t, x0, chain.marginals, step, alpha_bar)).abs().max().item()
        assert error < 1e-4, f"step {step}: reverse distribution differs from Bayes' rule by {error}"
        worst = max(worst, error)

    assert torch.equal(chain.sample(len(x0)), x0), "with an oracle denoiser the sample must equal the clean string"
    return f"reverse steps match Bayes' rule (error {worst:.0e}); oracle sampling recovers all {len(x0)} strings exactly"


def check_sampling(model: StringDiffusion, vocab: CharVocab, train_smiles: list[str], num_strings: int = 4) -> str:
    """Full sampling with the untrained network gives well-formed token rows.

    The samples are scored against train_smiles with optimil2.evaluation,
    the same call that scores the graph model. Random weights should not
    produce valid ionic liquids; the metrics in the summary are a baseline
    for a trained model, not a pass criterion.
    """
    tokens = model.sample(num_strings)
    assert tokens.shape == (num_strings, model.length) and tokens.dtype == torch.long
    assert 0 <= tokens.min() and tokens.max() < vocab.num_classes, "character class out of range"
    strings = [decode(row, vocab) for row in tokens]
    lengths = (tokens != PAD).sum(dim=1)
    assert [len(text) for text in strings] == lengths.tolist(), "a decoded string must have one character per non-PAD token"
    metrics = evaluate(strings, train_smiles)
    return (
        f"{num_strings} strings sampled in {model.num_steps} steps, {lengths.float().mean().item():.0f} characters on average; "
        f"{metrics} (untrained)"
    )


def check_same_molecules(string_folder: Path = IL_STRINGS_DIR, graph_folder: Path = IL_GRAPHS_DIR) -> str:
    """The string tensors and the graph tensors hold the same ILs, in the same split and row order.

    First the two molecules.csv tables are compared. Then every row of both
    tensors is decoded, and the SMILES the graph gives must be the string.
    """
    assert same_molecules(string_folder, graph_folder), (
        f"{string_folder.name}/ and {graph_folder.name}/ list different ILs or a different split. "
        "Rebuild both with the same options."
    )
    counts = []
    for split in SPLITS:
        strings, graphs = ILStrings(split, string_folder), ILGraphs(split, graph_folder)
        assert len(strings) == len(graphs), f"{split}: {len(strings)} strings but {len(graphs)} graphs"
        for row in range(len(strings)):
            from_graph = decode_il(*graphs[row], graphs.vocab)
            from_string = decode(strings[row], strings.vocab)
            assert from_graph is not None and from_graph.smiles == from_string, (
                f"{split} row {row}: the graph is {from_graph and from_graph.smiles}, the string is {from_string}"
            )
        counts.append(f"{split} {len(strings):,}")
    return f"string and graph tensors hold the same ILs row for row ({', '.join(counts)})"


def check_metrics(train_smiles: list[str], val_smiles: list[str]) -> str:
    """The metrics give the known answer on real ILs and on hand-written cases.

    Training ILs are valid, distinct ion pairs and not novel. Validation ILs
    are all novel, because their cations are held out.
    """
    some_train = train_smiles[:500]
    on_train = evaluate(some_train, train_smiles)
    assert (on_train.validity, on_train.uniqueness, on_train.novelty, on_train.ion_pair) == (1, 1, 0, 1), f"training ILs score {on_train}"
    on_val = evaluate(val_smiles, train_smiles)
    assert (on_val.validity, on_val.uniqueness, on_val.novelty, on_val.ion_pair) == (1, 1, 1, 1), f"validation ILs score {on_val}"

    # Nothing below depends on what the training set is, so a few ILs stand in for it to keep this quick.
    train_smiles = train_smiles[:10]
    assert evaluate(some_train + some_train, train_smiles).uniqueness == 0.5, "each IL given twice must be 50% unique"

    cases = {  # sample: (valid, ion pair)
        None: (False, False),
        "": (False, False),
        "C((": (False, False),  # not a SMILES
        "C(C)(C)(C)(C)C": (False, False),  # carbon with five bonds
        "C[N+](C)(C)C": (False, False),  # net charge +1
        "CCO": (True, False),  # neutral, but no ions
        "[N]": (True, False),  # radical
        "C[N+](C)(C)CC(=O)[O-]": (True, False),  # both charges on one molecule
        "C[N+](C)(C)CC[N+](C)(C)C.[Cl-].[Cl-]": (True, False),  # three ions
        "C[N+](C)(C)C.[Cl-]": (True, True),
        "[Cl-].C[N+](C)(C)C": (True, True),  # anion written first
    }
    for sample, (valid, ion_pair) in cases.items():
        got = evaluate([sample], train_smiles)
        assert (got.validity, got.ion_pair) == (valid, ion_pair), f"{sample!r} scores valid {got.validity}, ion pair {got.ion_pair}"
    nothing_valid = evaluate([None, "C(("], train_smiles)
    assert nothing_valid.uniqueness is None and nothing_valid.novelty is None, "with no valid sample, uniqueness and novelty are undefined"
    rewritten = evaluate(["C[N+](C)(C)C.[Cl-]", "[Cl-].C[N+](C)(C)C", "[Cl-].[N+](C)(C)(C)C"], train_smiles)
    assert rewritten.uniqueness == 1 / 3, "one IL written three ways must count as one molecule"
    return f"train scores as known, val as novel ({on_val}); {len(cases)} hand-written cases as expected"
