"""Replay Spectral_Entropy's current synthetic objective, not the missing audit.

Run with a Python environment containing torch:
    python3 tools/review_ledger/Spectral_Entropy_recompute.py
No model training, network access, or benchmark accuracy measurement is performed.
"""

import hashlib
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "zvf-program/flagship"
sys.path.insert(0, str(PILOT))
# This source tree contains a hyphen and is resolved through the explicit path above.
generate_benchmark_fixture = importlib.import_module(
    "pilot.benchmark_spectral_harness"
).generate_benchmark_fixture
objective_module = importlib.import_module("pilot.objective")
condition_loss = objective_module.condition_loss
entropic_givens_grpo = objective_module.entropic_givens_grpo
spectral_legendre_grpo = objective_module.spectral_legendre_grpo

OUT = ROOT / "platform_hybrid/experiments/results/review_2026-10-18/Spectral_Entropy/replay.json"


def evaluate(data, objective, **kwargs):
    logps = data["logps"].detach().clone().requires_grad_(True)
    loss, advantages = objective(**{**data, "logps": logps}, **kwargs)
    gradient = torch.autograd.grad(loss, logps)[0]
    if not torch.isfinite(gradient).all() or advantages.requires_grad:
        raise RuntimeError("Expected finite gradient and detached advantage weights")
    return gradient, advantages


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = []
    for g in (4, 8, 16):
        for length in (512, 1024, 2048):
            normal, homogeneous = generate_benchmark_fixture(g, length, seed=42)
            probabilities = normal["logps"].softmax(-1)
            entropy = -(probabilities * (probabilities + 1e-12).log()).sum(-1)
            if not bool((entropy > 0.5).all()):
                raise RuntimeError("Default gate unexpectedly active")
            record: dict[str, object] = {
                "G": g,
                "L": length,
                "entropy_min": float(entropy.min()),
                "entropy_max": float(entropy.max()),
                "default_gate_count": int((entropy < 0.5).sum()),
            }
            for name, data in (("alternating", normal), ("homogeneous", homogeneous)):
                base, _ = evaluate(data, condition_loss, condition="intended_full")
                spectral, _ = evaluate(data, spectral_legendre_grpo)
                gated, _ = evaluate(data, entropic_givens_grpo)
                forced, _ = evaluate(data, entropic_givens_grpo, tau_entropy=10.0)
                difference = float((spectral - gated).abs().max())
                forced_difference = float((spectral - forced).abs().max())
                if difference != 0 or forced_difference <= 1e-12:
                    raise RuntimeError(
                        "Inactive-gate equality or forced-active positive control failed"
                    )
                if name == "homogeneous" and (
                    float(base.norm()) != 0 or float(spectral.norm()) <= 0
                ):
                    raise RuntimeError("Homogeneous reward/auxiliary gradient boundary failed")
                record[name] = {
                    "baseline_gradient_norm": float(base.norm()),
                    "spectral_gradient_norm": float(spectral.norm()),
                    "gated_gradient_norm": float(gated.norm()),
                    "default_gradient_max_difference": difference,
                    "forced_gate_count": int((entropy < 10.0).sum()),
                    "forced_gradient_max_difference": forced_difference,
                }
            rows.append(record)

    normal, homogeneous = generate_benchmark_fixture(4, 512, seed=42)
    reference, _ = evaluate(normal, condition_loss, condition="intended_full")
    unit, _ = evaluate(homogeneous, spectral_legendre_grpo, lambda_entropy=1.0)
    sweep = []
    for weight in (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0):
        flat, _ = evaluate(homogeneous, spectral_legendre_grpo, lambda_entropy=weight)
        mixed, _ = evaluate(normal, spectral_legendre_grpo, lambda_entropy=weight)
        if not torch.allclose(flat, weight * unit, rtol=1e-12, atol=1e-15):
            raise RuntimeError("Homogeneous gradient is not linear in coupling weight")
        cosine = float(
            torch.nn.functional.cosine_similarity(mixed.flatten(), reference.flatten(), dim=0)
        )
        sweep.append(
            {
                "lambda": weight,
                "homogeneous_gradient_norm": float(flat.norm()),
                "heterogeneous_cosine_to_baseline": cosine,
            }
        )

    # Independent active-row/reduction reference at lambda=0, with unequal lengths.
    data, _ = generate_benchmark_fixture(8, 32, seed=42)
    data["active_rows"][-2:] = False
    data["completion_mask"][-2:] = 0
    data["completion_mask"][0, 7:] = 0
    actual, _ = evaluate(data, spectral_legendre_grpo, lambda_entropy=0.0)
    rewards = data["rewards"][:6]
    mean = sum(float(x) for x in rewards) / 6
    sd = math.sqrt(sum((float(x) - mean) ** 2 for x in rewards) / 5)
    adv = torch.tensor([(float(x) - mean) / sd for x in rewards], dtype=torch.float64)
    ratios = data["logps"][:6].exp()
    gate = ((adv[:, None] >= 0) & (ratios <= 1.28)) | ((adv[:, None] < 0) & (ratios >= 0.8))
    expected = torch.zeros_like(actual)
    mask = data["completion_mask"][:6]
    expected[:6] = -ratios * adv[:, None] * gate * mask / mask.sum(1)[:, None] / 6
    if not torch.allclose(actual, expected, rtol=1e-12, atol=1e-15):
        raise RuntimeError("Active-row sample-SD or per-completion loss reference disagrees")

    sources = [
        "pilot/objective.py",
        "pilot/spectral_attention.py",
        "pilot/entropic_gating.py",
        "pilot/benchmark_spectral_harness.py",
    ]
    hashes = {
        str((PILOT / p).relative_to(ROOT)): hashlib.sha256((PILOT / p).read_bytes()).hexdigest()
        for p in sources
    }
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    source_paths = [str((PILOT / p).relative_to(ROOT)) for p in sources]
    source_git = {
        # Commit checked out when the replay ran; the hashes above are the binding.
        "head_commit": git("rev-parse", "HEAD"),
        # Empty list means every hashed source matches head_commit byte-for-byte.
        "sources_modified_vs_head": git("status", "--porcelain", "--", *source_paths).splitlines(),
    }
    result = {
        "evidence_type": "new deterministic synthetic CPU replay; not the historical 77-case audit",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "dtype": "float64",
        "seed": 42,
        "n_modes": 8,
        "default_lambda": 0.1,
        "default_tau": 0.5,
        "forced_control_tau": 10.0,
        "n_noise_dims": 1,
        "source_sha256": hashes,
        "source_git": source_git,
        "grid": rows,
        "lambda_sweep_G4_L512": sweep,
        "active_row_reference_max_error": float((actual - expected).abs().max()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"PASS: 9 fixture cells, 18 gate/control comparisons, 8 coupling weights, active-row reference; {OUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
