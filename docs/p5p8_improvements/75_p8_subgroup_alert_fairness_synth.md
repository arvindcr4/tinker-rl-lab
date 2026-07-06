# 75 — Iter 64 JOB B / SYNTH re-ranking

After iter-64 JOB A drove the P8 subgroup alert-distribution fairness
vein (#75) to validated, the ledger is at **76 validated rows, 0 open
proposed rows, 2 rejected-with-reason rows**.

No top-deferred item from the prior synth-surface lists remained
driveable in iter-64:

| Deferred candidate | Origin | Iter-64 status |
|---|---|---|
| P8 c_sense scaling | iter-28 surface | re-statement of iter-32 reject (#43) — needs real synchronous-LLM cost data |
| P8 noisy-sensor surrogate in cost-optimal | iter-32 surface | parked — noise-budget simulator not yet built |
| P7 controller replay on mega-corpus | iter-32 surface | parked — iter-59 already covered the N2 four-method run on the same data class |
| P6 ci_method extension to variant_delta_record | iter-32 surface | parked — iter-62 outcomes.coverage block now records `declared_deltas_coverage` per entry; the actual `ci_method` extension awaits measured-delta harvest |

JOB B therefore delivers the re-ranking + parking-record rather than a
new closed-loop vein, consistent with the iter-32 stricture that
un-drivable items be recorded as rejects, not cycled indefinitely.

**Top of re-ranked stack (all validated, P8-heavy):**

1. #75 P8 subgroup alert-distribution fairness (iter 64)
2. #70 P8 operational calibration gap (iter 60)
3. #66 P8 alert-volume Pareto (iter 56)
4. #55 P8 threshold transfer (iter 49)
5. #52 P8 asymmetric cost frontier (iter 40)

**Recommended next-iter mint veins:** seed-stability pass on #75,
multi-feature sensor ablations (V_std-only, V_max-only), P5/P8
synthesis paragraph linking iter-64 and iter-32.
