# Local TRL script: explicit reward and fresh-run contract

This correction affects newly generated local TRL scripts. It does not alter
historical experiments, frozen publications, signed records, or strict gates.
No native model/backend training was run to validate it.

## Reproduced defects

The old generated script optimized a keyword/boxing heuristic while describing
it as a process/outcome reward. With reference `2`, a wrong response containing
five `step` words and `\\boxed{999}` received 1.0; the correct response `2`
received 0.0. References were ignored. Its batch-level variance also reported
collapse rate 0.0 for two entirely collapsed groups `[0,0]` and `[0.5,0.5]`.
Two initial regression tests failed before changes: absent explicit reward was
accepted, and configured KL/optimizer/generation settings were dropped.

## Current supported boundary

- Generated scripts require `reward_callable="module:attribute"` and a fresh
  `output_dir`; CLI equivalents are `--reward-callable` and `--output-dir`.
  The user supplies a task-specific reward callable compatible with TRL.
  Importability/callability is checked; scientific reward validity is not inferred.
  There is no implicit heuristic, GSM8K verifier, PRM, or group-collapse metric.
- Local installed source and project dependency files both identify TRL 1.2.0.
  Source inspection confirms supported GRPO fields and absence of the old
  top-level Online DPO API. Scripts require that exact runtime version.
- GRPO optimizer betas/epsilon/weight decay, KL coefficient, generation count,
  temperature, completion cap, and seed are forwarded. Seed is set before model
  and adapter creation. CLI canonical seed/group/KL/completion/accumulation
  settings reach the actual generated script.
- Existing output directories are rejected before model loading. Resume and
  `model_path` are unsupported. Multiple processes/GPUs are explicitly rejected
  because exclusive output reservation has no distributed coordination contract.
- iDPO/Online DPO, explicit prompt-length caps, and evaluation datasets fail
  explicitly. TRL 1.2.0 does not accept the former prompt-cap field; this patch
  does not silently replace it with a different sequence-length constraint.
- Successful subprocess completion does not supply observed optimizer-step
  evidence: `TrainingResult.final_step` is `None`. The CLI's token-budget option
  remains an estimate based on configured batch size, canonical accumulation,
  and assumed tokens/sample; it is not token-exact enforcement or measured usage.

## Validation

63 focused CPU tests passed across `test_trl_native_contract.py`,
`test_peft_support.py`, `test_launcher_forwarding.py`, and
`test_pydantic_support.py`. The new contract file contributes 18 tests.
Generated code ran only with fully inert model/dataset/trainer/logging modules,
including the actual CLI-to-script path. Tests cover wrong-version rejection,
invalid/missing reward, existing outputs, unsupported modes, config forwarding,
and unknown observed steps. No provider call, GPU job, backend installation,
checkpoint download, or native training was performed.

Broader backend equivalence and reward correctness require separate validation;
these tests establish wrapper behavior only. Legacy trainer factory/scaffold
APIs are not certified by this generated-script correction.
