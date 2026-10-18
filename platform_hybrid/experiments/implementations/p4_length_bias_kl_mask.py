"""Illustrative scaffold only; no empirical training/evaluation implementation."""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(
    0, str(next(p for p in _Path(__file__).resolve().parents if (p / "platform_hybrid").is_dir()))
)
from platform_hybrid.experiments.research_contracts import simulation_result

import argparse
import random


def run_p4_kl_surprise_mask(*, simulation=False, output_dir=None):
    rng = random.Random(42)
    return simulation_result(
        "p4_length_bias_kl_mask",
        {
            "lengths": [rng.randint(50, 200) for _ in range(10)],
            "kl": [rng.uniform(0.1, 1.5) for _ in range(10)],
        },
        simulation=simulation,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    print(run_p4_kl_surprise_mask(simulation=args.simulation, output_dir=args.output_dir))
