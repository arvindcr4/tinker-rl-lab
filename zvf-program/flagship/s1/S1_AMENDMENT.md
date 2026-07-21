# S1 pre-compute conformance amendment

Date: 2026-07-20

This amendment does not change the frozen E1 records or reinterpret a completed
experimental result. It records a specification distinction discovered during
the CPU-only S1 gate, before any flagship pilot or confirmatory compute.

The original S1 acceptance criterion requires canonical, TRL, and verl loss,
mask, importance-ratio, selection, and gradient agreement. Exact source-level
differentials show that the native framework defaults are not identical:

- TRL 1.2.0 standardizes with `sample_std + 1e-4`, uses distinct loss
  denominators for GRPO, DAPO, and DrGRPO, computes advantages before applying
  completion-selection masks, and has no native AERO posterior objective.
- verl 0.3.0.post1 standardizes GRPO with `sample_std + 1e-6`, reduces its PPO
  loss as one masked-token mean, and exposes no native DAPO, GSPO, DrGRPO, or
  AERO objective kernel. Its annotated tensor group index fails dictionary
  lookup; the working native path requires stable Python group identifiers.

Accordingly S1 now has two separately reported surfaces:

1. **Native-framework audit.** Execute the unmodified pinned kernels and retain
   every `PASS`, `NUMERICAL_VARIATION`, `MATERIAL_DIFFERENCE`, or `NOT_TESTED`
   verdict. Native mismatches are evidence and may never be coerced into a pass.
2. **Intended-integration audit.** Exercise the exact custom trainer/kernel
   integration that would be used in a flagship run. This surface must agree
   with the canonical reference for every prespecified field at `rtol <= 1e-6`
   and `atol <= 1e-8`, including all controller action paths and frozen
   non-treatment fields.

S1 passes only when the intended-integration audit passes on both stacks and
the native-framework audit remains attached to the implementation-freeze
receipt. Until then, S2 GPU screening and all provider expenditure remain
forbidden. If an exact integration cannot be implemented without changing the
scientific treatment, the flagship premise fails this gate and the campaign
stops rather than weakening the acceptance threshold.
