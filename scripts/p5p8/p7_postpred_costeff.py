"""Iter 19 follow-up — P7 vein (e) cost-efficiency:
expected contrast-restored per extra rollout dollar.

Reads `p7_postpred_summary.json` and computes the per-method
restore-per-extra-rollout metric for each controller. A controller
is cost-efficient if its expected_restored / extra_rollouts ratio
exceeds the baseline (fixed-G=8 produces 0 extra rollouts and 0
extra restored).

Cost model (matches iter 11/14/15):
  fixed-G=8 baseline: n_steps * 16 prompts * G_BASE = 5120 rollouts.
  Each fired escalation: G_BASE -> G_NEW on that prompt
    (zvf-triage step-level: all 16 prompts in step; per-prompt:
    only the fired prompt).
"""

import json
import pathlib

WORKTREE = pathlib.Path("/home/claude/tinker-rl-lab-minimax")
SUMMARY = WORKTREE / "experiments/results/p5p8/p7_postpred_summary.json"
OUT = WORKTREE / "experiments/results/p5p8/p7_postpred_costeff.tsv"

G_BASE = 8
G_NEW = 16
N_STEPS = 40
N_PROMPTS_PER_STEP = 16
BASELINE_ROLLOUTS = N_STEPS * N_PROMPTS_PER_STEP * G_BASE  # 5120

# fires are *per-method* in the summary (already aggregated across 1 seed x 40 steps x 16 prompts)
# zvf-triage fires are per step (escalates all 16 prompts); bayes/dualformer fires are per prompt
PER_STEP_VS_PER_PROMPT = {
    "zvf_triage": "step",  # fires=steps; each fire escalates all 16 prompts
    "bayes": "prompt",  # fires=prompts; each fire escalates 1 prompt
    "dualformer": "prompt",
}


def extra_rollouts(ctrl: str, fires: int) -> int:
    extra_per_fire = (G_NEW - G_BASE) * N_PROMPTS_PER_STEP if PER_STEP_VS_PER_PROMPT[ctrl] == "step" else (G_NEW - G_BASE)
    return int(fires * extra_per_fire)


def main():
    with SUMMARY.open() as fh:
        data = json.load(fh)
    rows = []
    rows.append("method\tcontroller\ttau\tfires\textra_rollouts\texpected_restored\trestore_per_1k_extra\tbaseline_ratio")
    for method, pm in data["per_method"].items():
        for cname, cev in pm["controller_evaluations"].items():
            # parse controller name
            if cname == "dualformer_auto":
                ctrl = "dualformer"
            elif cname.startswith("zvf_triage"):
                ctrl = "zvf_triage"
            elif cname.startswith("bayes"):
                ctrl = "bayes"
            else:
                ctrl = cname.split("_")[0]
            tau = cname.split("_")[-1]
            try:
                tau = float(tau)
            except ValueError:
                tau = "-"
            fires = cev["fires"]
            restored = cev["expected_restore_sum"]
            extra = extra_rollouts(ctrl, fires)
            ratio = (restored / extra * 1000.0) if extra > 0 else 0.0
            baseline_ratio = restored / BASELINE_ROLLOUTS * 1000.0
            rows.append(f"{method}\t{ctrl}\t{tau}\t{fires}\t{extra}\t{restored:.2f}\t{ratio:.4f}\t{baseline_ratio:.4f}")
    OUT.write_text("\n".join(rows) + "\n")
    print(f"wrote {OUT}")
    print()
    print("=== restore per 1000 extra rollouts (higher = more cost-efficient) ===")
    print(f"{'method':<8}{'controller':<14}{'tau':>6}{'fires':>8}{'extra':>10}{'restored':>12}{'restored/1k_extra':>20}")
    for line in rows[1:]:
        parts = line.split("\t")
        print(f"{parts[0]:<8}{parts[1]:<14}{parts[2]:>6}{parts[3]:>8}{parts[4]:>10}{parts[5]:>12}{parts[6]:>20}")


if __name__ == "__main__":
    main()