"""Automated Spectral Benchmark Harness for ZVF Program Pilot.

Runs scaling trials comparing standard GRPO, spectral_legendre_grpo, and entropic_givens_grpo
across group sizes G in {4, 8, 16} and sequence lengths L in {512, 1024, 2048}.
Calculates reward variance recovery, gradient norm retention, and FLOP overhead,
exporting structured JSON results to spectral_benchmark_results.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from pilot.objective import condition_loss, spectral_legendre_grpo, entropic_givens_grpo


@dataclass(frozen=True, slots=True)
class AlgorithmMetrics:
    algorithm: str
    std_advantages_normal: float
    std_advantages_zvf: float
    reward_variance_recovery_ratio: float
    gradient_norm_normal: float
    gradient_norm_zvf: float
    gradient_norm_retention: float
    theoretical_flops: int
    flop_overhead_ratio: float
    flop_overhead_percent: float
    execution_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrialResult:
    group_size: int
    sequence_length: int
    metrics: dict[str, AlgorithmMetrics]

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_size": self.group_size,
            "sequence_length": self.sequence_length,
            "results": {alg: metrics.to_dict() for alg, metrics in self.metrics.items()},
        }


def generate_benchmark_fixture(
    G: int,
    L: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: int = 42,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Generate deterministic normal and zero-reward-variance (ZVF) benchmark fixtures."""
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # 1. Normal reward variance fixture: binary alternating rewards
    rewards_normal = torch.tensor([1.0 if i % 2 == 0 else 0.0 for i in range(G)], dtype=dtype, device=device)

    # 2. Zero reward variance fixture (ZVF starvation): all equal rewards
    rewards_zvf = torch.ones(G, dtype=dtype, device=device)

    # Structured log-probabilities with position and row variation
    old_logps = torch.zeros((G, L), dtype=dtype, device=device)
    offsets = torch.linspace(-0.1, 0.1, steps=G, dtype=dtype, device=device)[:, None]
    positions = torch.linspace(0.1, 1.0, steps=L, dtype=dtype, device=device)[None, :]
    noise = torch.randn((G, L), generator=generator, dtype=dtype, device=device) * 0.01
    logps = old_logps + offsets * positions + noise

    completion_mask = torch.ones((G, L), dtype=dtype, device=device)
    active_rows = torch.ones(G, dtype=torch.bool, device=device)

    normal_data = {
        "rewards": rewards_normal,
        "logps": logps,
        "old_logps": old_logps,
        "completion_mask": completion_mask,
        "active_rows": active_rows,
    }

    zvf_data = {
        "rewards": rewards_zvf,
        "logps": logps,
        "old_logps": old_logps,
        "completion_mask": completion_mask,
        "active_rows": active_rows,
    }

    return normal_data, zvf_data


def compute_theoretical_flops(
    algorithm: str,
    G: int,
    L: int,
    n_modes: int = 8,
    d_feat: int = 2,
    n_noise_dims: int = 1,
) -> int:
    """Compute exact theoretical FLOP count for objective evaluation."""
    # Standard GRPO base FLOPs per token:
    # ratio & clamp (4), advantages multiplication (2), min (1), negate (1), mask & mean reduction (3) = 11 FLOPs/token
    base_flops = 11 * G * L

    if algorithm == "standard_grpo":
        return base_flops

    # Spectral Legendre augmentation:
    # 1. Legendre basis evaluation: ~5 * n_modes * L FLOPs
    legendre_flops = 5 * n_modes * L
    # 2. Spectral projection einsum(bld, ln -> bnd): 2 * G * L * d_feat * n_modes
    projection_flops = 2 * G * L * d_feat * n_modes
    # 3. Pairwise spectral distance: G * (G - 1) * 3 * n_modes * d_feat
    pairwise_flops = G * (G - 1) * 3 * n_modes * d_feat
    # 4. Standardizing s_scores and advantage combination: ~6 * G
    score_flops = 6 * G

    spectral_flops = base_flops + legendre_flops + projection_flops + pairwise_flops + score_flops

    if algorithm == "spectral_legendre_grpo":
        return int(spectral_flops)

    if algorithm == "entropic_givens_grpo":
        # Entropic Givens additional FLOPs:
        # 1. Softmax & Shannon entropy density: 6 * G * L
        entropy_flops = 6 * G * L
        # 2. Givens rotation per active row below threshold: max G * n_noise_dims * 5 * L
        givens_flops = G * n_noise_dims * 5 * L
        return int(spectral_flops + entropy_flops + givens_flops)

    raise ValueError(f"Unknown algorithm: {algorithm}")


def evaluate_algorithm_trial(
    algorithm: str,
    normal_data: dict[str, torch.Tensor],
    zvf_data: dict[str, torch.Tensor],
    G: int,
    L: int,
    num_warmup: int = 5,
    num_runs: int = 20,
) -> AlgorithmMetrics:
    """Run loss, advantage, gradient, FLOP, and timing analysis for a single algorithm trial."""
    condition_map = {
        "standard_grpo": "intended_full",
        "spectral_legendre_grpo": "spectral_legendre",
        "entropic_givens_grpo": "entropic_givens",
    }

    condition = condition_map[algorithm]

    # --- 1. Normal reward variance run ---
    logps_norm = normal_data["logps"].detach().clone().requires_grad_(True)
    loss_norm, advs_norm = condition_loss(
        condition=condition,
        rewards=normal_data["rewards"],
        logps=logps_norm,
        old_logps=normal_data["old_logps"],
        completion_mask=normal_data["completion_mask"],
        active_rows=normal_data["active_rows"],
    )
    grad_norm_tensor = torch.autograd.grad(loss_norm, logps_norm)[0]
    std_advs_normal = float(advs_norm.std(correction=1)) if len(advs_norm) > 1 else 0.0
    grad_norm_normal = float(torch.linalg.vector_norm(grad_norm_tensor))

    # --- 2. Zero reward variance (ZVF) run ---
    logps_zvf = zvf_data["logps"].detach().clone().requires_grad_(True)
    loss_zvf, advs_zvf = condition_loss(
        condition=condition,
        rewards=zvf_data["rewards"],
        logps=logps_zvf,
        old_logps=zvf_data["old_logps"],
        completion_mask=zvf_data["completion_mask"],
        active_rows=zvf_data["active_rows"],
    )
    grad_zvf_tensor = torch.autograd.grad(loss_zvf, logps_zvf)[0]
    std_advs_zvf = float(advs_zvf.std(correction=1)) if len(advs_zvf) > 1 else 0.0
    grad_norm_zvf = float(torch.linalg.vector_norm(grad_zvf_tensor))

    # Reward variance recovery ratio (relative to normal reward std or 1.0 baseline)
    if std_advs_normal > 1e-12:
        recovery_ratio = std_advs_zvf / std_advs_normal
    else:
        recovery_ratio = 1.0 if std_advs_zvf > 1e-12 else 0.0

    # Gradient norm retention under ZVF relative to normal variance
    if grad_norm_normal > 1e-12:
        grad_retention = grad_norm_zvf / grad_norm_normal
    else:
        grad_retention = 1.0 if grad_norm_zvf > 1e-12 else 0.0

    # --- 3. Timing / execution latency ---
    for _ in range(num_warmup):
        _ = condition_loss(
            condition=condition,
            rewards=zvf_data["rewards"],
            logps=zvf_data["logps"],
            old_logps=zvf_data["old_logps"],
            completion_mask=zvf_data["completion_mask"],
            active_rows=zvf_data["active_rows"],
        )

    start_time = time.perf_counter()
    for _ in range(num_runs):
        _ = condition_loss(
            condition=condition,
            rewards=zvf_data["rewards"],
            logps=zvf_data["logps"],
            old_logps=zvf_data["old_logps"],
            completion_mask=zvf_data["completion_mask"],
            active_rows=zvf_data["active_rows"],
        )
    end_time = time.perf_counter()
    avg_latency_ms = ((end_time - start_time) / num_runs) * 1000.0

    # --- 4. FLOP metrics ---
    flops = compute_theoretical_flops(algorithm, G, L)
    std_flops = compute_theoretical_flops("standard_grpo", G, L)
    flop_ratio = flops / max(std_flops, 1)
    flop_percent = (flop_ratio - 1.0) * 100.0

    return AlgorithmMetrics(
        algorithm=algorithm,
        std_advantages_normal=std_advs_normal,
        std_advantages_zvf=std_advs_zvf,
        reward_variance_recovery_ratio=recovery_ratio,
        gradient_norm_normal=grad_norm_normal,
        gradient_norm_zvf=grad_norm_zvf,
        gradient_norm_retention=grad_retention,
        theoretical_flops=flops,
        flop_overhead_ratio=flop_ratio,
        flop_overhead_percent=flop_percent,
        execution_time_ms=avg_latency_ms,
    )


def run_spectral_benchmark_harness(
    group_sizes: Sequence[int] = (4, 8, 16),
    sequence_lengths: Sequence[int] = (512, 1024, 2048),
    algorithms: Sequence[str] = (
        "standard_grpo",
        "spectral_legendre_grpo",
        "entropic_givens_grpo",
    ),
    output_path: str | Path = "spectral_benchmark_results.json",
    num_warmup: int = 5,
    num_runs: int = 20,
) -> dict[str, Any]:
    """Execute all scaling trials across group sizes G and sequence lengths L."""
    trials: list[TrialResult] = []

    print("=" * 70)
    print("ZVF PROGRAM: AUTOMATED SPECTRAL BENCHMARK HARNESS")
    print(f"Group sizes G: {list(group_sizes)}")
    print(f"Sequence lengths L: {list(sequence_lengths)}")
    print(f"Algorithms: {list(algorithms)}")
    print("=" * 70)

    for G in group_sizes:
        for L in sequence_lengths:
            print(f"\n---> Running Trial: Group Size G={G}, Sequence Length L={L}")
            normal_data, zvf_data = generate_benchmark_fixture(G, L)

            alg_metrics: dict[str, AlgorithmMetrics] = {}
            for alg in algorithms:
                metrics = evaluate_algorithm_trial(
                    algorithm=alg,
                    normal_data=normal_data,
                    zvf_data=zvf_data,
                    G=G,
                    L=L,
                    num_warmup=num_warmup,
                    num_runs=num_runs,
                )
                alg_metrics[alg] = metrics
                print(
                    f"  [{alg:22s}] ZVF Adv Std: {metrics.std_advantages_zvf:.4f} | "
                    f"Grad Norm Retention: {metrics.gradient_norm_retention * 100:.2f}% | "
                    f"FLOP Overhead: +{metrics.flop_overhead_percent:.1f}% ({metrics.execution_time_ms:.3f} ms)"
                )

            trials.append(TrialResult(group_size=G, sequence_length=L, metrics=alg_metrics))

    # Calculate summary aggregates across all trials
    avg_recovery: dict[str, float] = {}
    avg_retention: dict[str, float] = {}
    avg_flop_ratio: dict[str, float] = {}

    for alg in algorithms:
        recovery_vals = [t.metrics[alg].reward_variance_recovery_ratio for t in trials]
        retention_vals = [t.metrics[alg].gradient_norm_retention for t in trials]
        flop_ratios = [t.metrics[alg].flop_overhead_ratio for t in trials]

        avg_recovery[alg] = sum(recovery_vals) / len(recovery_vals)
        avg_retention[alg] = sum(retention_vals) / len(retention_vals)
        avg_flop_ratio[alg] = sum(flop_ratios) / len(flop_ratios)

    benchmark_output = {
        "metadata": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "group_sizes": list(group_sizes),
            "sequence_lengths": list(sequence_lengths),
            "algorithms": list(algorithms),
            "total_trials": len(trials),
            "num_warmup": num_warmup,
            "num_runs": num_runs,
        },
        "summary": {
            "average_reward_variance_recovery_ratio": avg_recovery,
            "average_gradient_norm_retention": avg_retention,
            "average_flop_overhead_ratio": avg_flop_ratio,
        },
        "scaling_trials": [trial.to_dict() for trial in trials],
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(benchmark_output, f, indent=2)

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY AGGREGATES:")
    for alg in algorithms:
        print(
            f"  {alg:24s} -> Avg Recovery: {avg_recovery[alg]:.4f} | "
            f"Avg Grad Retention: {avg_retention[alg]*100:.2f}% | "
            f"Avg FLOP Ratio: {avg_flop_ratio[alg]:.3f}x"
        )
    print("=" * 70)
    print(f"Results written to: {out_file.resolve()}")
    return benchmark_output


def main() -> None:
    parser = argparse.ArgumentParser(description="ZVF Spectral GRPO Benchmark Harness")
    parser.add_argument(
        "--output",
        type=str,
        default="spectral_benchmark_results.json",
        help="Output path for JSON results",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=5,
        help="Number of warmup runs per trial",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=20,
        help="Number of timing runs per trial",
    )
    args = parser.parse_args()

    run_spectral_benchmark_harness(
        output_path=args.output,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
    )


if __name__ == "__main__":
    main()
