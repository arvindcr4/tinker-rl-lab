# Wave 001 — Phase 1

Launched: 2026-04-11T19:14:17
Wave size: 8 agents
Current best: baseline_v0 (None)

## Hypotheses tested in this wave

- **v001** (H01): `group_size=32` improves last10_avg vs 8 by diversifying rollouts → `group_size: 32`
- **v002** (H02): `group_size=64` further improves over 32 → `group_size: 64`
- **v003** (H03): `group_size=128` is the ceiling before diminishing returns → `group_size: 128`
- **v004** (H04): `group_size=16, batch=4` (same total samples) beats `group_size=32, batch=2` → `group_size: 16, batch: 4`
- **v005** (H05): `temperature=1.2` reduces zero-reward saturation by widening the sample distribution → `temperature: 1.2`
- **v006** (H06): `temperature=1.5` + `group_size=32` compound into escape → `temperature: 1.5, group_size: 32`
- **v007** (H07): `temperature=0.6` with graded rewards gives sharper learning signal → `temperature: 0.6, reward_shape: graded`
- **v008** (H08): `reward_shape=graded` — give partial credit for \\boxed{ presence, answer format, correct-digit count → `reward_shape: graded`
