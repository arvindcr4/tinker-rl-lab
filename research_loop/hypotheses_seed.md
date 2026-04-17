# Seed Hypotheses — 20 ideas for Wave 1

Drawn from `autoresearch.ideas.md`, the paper's Section 7 roadmap, and the
GRPO / DAPO / RLOO literature. Each entry is a self-contained hypothesis
that one agent can test in one Phase 1 run.

Format: `H<id> | category | one-line hypothesis | knobs changed | rationale`

## Group size sweep (targets all-incorrect saturation)

- **H01** | group | `group_size=32` improves last10_avg vs 8 by diversifying rollouts | group_size: 32 | 4× more samples per prompt → higher chance of a non-zero reward in the group → non-zero advantages flow
- **H02** | group | `group_size=64` further improves over 32 | group_size: 64 | Same logic, stronger dose
- **H03** | group | `group_size=128` is the ceiling before diminishing returns | group_size: 128 | Ryan-Li-style limit test
- **H04** | group | `group_size=16, batch=4` (same total samples) beats `group_size=32, batch=2` | group_size: 16, batch: 4 | Isolate group count from batch diversity

## Temperature / exploration

- **H05** | temp | `temperature=1.2` reduces zero-reward saturation by widening the sample distribution | temperature: 1.2 | Current 0.8 may be too peaked for stalled 3B
- **H06** | temp | `temperature=1.5` + `group_size=32` compound into escape | temperature: 1.5, group_size: 32 | Joint dose
- **H07** | temp | `temperature=0.6` with graded rewards gives sharper learning signal | temperature: 0.6, reward_shape: graded | Opposite direction — tighter distribution + richer reward

## Reward shaping (breaks zero-advantage)

- **H08** | reward | `reward_shape=graded` — give partial credit for \\boxed{ presence, answer format, correct-digit count | reward_shape: graded | Zero-advantage collapse is driven by all-zero groups; graded reward eliminates all-same groups
- **H09** | reward | `reward_shape=partial` — 0.5 credit for right answer without \\boxed{}, 1.0 for full form | reward_shape: partial | Lighter graded variant

## Advantage normalization

- **H10** | adv | `adv_norm=rank` — rank-based advantages tolerate all-same rewards | adv_norm: rank | Breaks the std/eps trick when std≈0
- **H11** | adv | `adv_norm=none` — raw `(r - mr)` without variance rescaling | adv_norm: none | Test whether std normalization is the issue

## Curriculum

- **H12** | curriculum | `curriculum=easy_first` on prompt-length-sorted GSM8K bootstraps 3B | curriculum: easy_first | Paper idea #4: easy-to-hard curriculum
- **H13** | curriculum | `curriculum=hard_first` counter-test | curriculum: hard_first | Control for H12 — some literature says hard-first is better

## Learning rate

- **H14** | lr | `lr=6e-5` (2×) accelerates initial learning at 0.6B | lr: 6.0e-5 | grpo_exp_b_high_lr.py existed, revisit
- **H15** | lr | `lr=1.5e-5` (0.5×) stabilizes | lr: 1.5e-5 | Opposite direction

## LoRA rank

- **H16** | rank | `rank=32` gives more capacity for representation on small model | rank: 32 | Separate from zero-loss effect
- **H17** | rank | `rank=8` tests whether the task is expressible in lower dim | rank: 8 | Minimal rank

## Model / scale proxies

- **H18** | model | `model=Qwen/Qwen3-1.7B` is the closer proxy for 3B | model: Qwen/Qwen3-1.7B | Bigger proxy = closer transfer of recipe
- **H19** | model | `model=Qwen/Qwen3-0.6B, rank=32, group_size=32` as a joint baseline | rank: 32, group_size: 32 | Combination check

## From-scratch seed (one slot reserved for a non-incremental idea)

- **H20** | scratch | Unconstrained: agent is told the failure mode and rescue goal with zero hyperparam hints | (agent's choice) | Ryan-Li-style from-scratch agent reserved for Phase 3
