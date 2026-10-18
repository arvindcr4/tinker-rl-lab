import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(
    0, str(next(p for p in _Path(__file__).resolve().parents if (p / "platform_hybrid").is_dir()))
)
from platform_hybrid.experiments.research_contracts import simulation_result

import numpy as np


def plot_sparsity(*, simulation=False, output_dir=None):
    """Quarantined illustration; never writes a paper figure with model labels."""
    rng = np.random.default_rng(42)
    return simulation_result(
        "lora_sparsity",
        {"illustrative_profiles": rng.uniform(0, 100, (2, 32)).tolist()},
        simulation=simulation,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    plot_sparsity()
