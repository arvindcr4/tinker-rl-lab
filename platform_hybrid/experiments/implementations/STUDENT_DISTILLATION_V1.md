# Student-trajectory distillation v1

Protocol ID: `student-trajectory-kl-v1`. This is a new local code contract, not a
reinterpretation of any existing experiment or evidence of model performance.
Historical results and generated papers remain unchanged.

## Objective and data flow

`OnPolicyDistillationTrainer` now generates fresh continuations from the current
student during **every** loss call, before scoring the same token histories with
student and teacher. It never substitutes teacher-generated responses. Its input
has only prompt `input_ids` and `attention_mask`; labeled response data is rejected.

The objective is forward KL(teacher || student), averaged over valid response
prediction positions, with temperature-squared scaling. Both distributions use
the declared distillation temperature (default 2), distinct from generation's
unit temperature. The first generated token and first EOS are included; prompts,
post-EOS tokens and padding are excluded. The denominator is valid tokens in the
current microbatch, not total tokens across gradient accumulation microbatches.

This is a **semi-gradient** objective: gradients flow through student scoring,
not through discrete generation or the sampled-state distribution. It is not a
policy-gradient estimator of the derivative of expected trajectory KL. Teacher
scores are detached. Both models score in eval mode to avoid mismatched dropout;
previous modes are restored even after an exception. Generation uses no gradients.

Generation uses a newly constructed Transformers `GenerationConfig`: sampling
is enabled, temperature 1, top-k 0, top-p 1, one beam and one returned sequence.
Checkpoint generation defaults (including penalties/forced tokens) are not
inherited (`use_model_defaults=False` is explicit; the installed Transformers
version must support this generation argument). Each immutable trajectory records exact input IDs, response mask,
optimizer global-step behavior version, protocol version, and the complete
resolved generation configuration including continuation budget. These snapshots
are retained as `trainer.last_trajectories` for inspection; they are not durable
run manifests or reusable training data. Sampling RNG follows the caller's/Trainer's
seed; reproducible restart/RNG checkpoint validation is outside this local v1.

## Explicit entry point

`train_on_policy_distillation` requires caller-supplied local student, teacher,
fast tokenizers, tokenized prompt dataset, and `TrainingArguments`. Set
`DistillationConfig(alpha=0.0)` explicitly: sampled student tokens are not genuine
supervised task targets. Set `remove_unused_columns=False`, `push_to_hub=False`,
and `report_to=[]`; choose a new empty output directory. The API does not load
models or publish artifacts. Invoking it runs training and therefore requires
separate authorization/resources; the regression suite does not invoke it.

Teacher and student must have identical token-to-ID maps, fast-tokenizer
pipelines, special-token mappings, and model vocabulary sizes. Slow/custom
unverifiable tokenizers fail closed. Local single-device execution is supported;
distributed, FSDP/DeepSpeed and multi-device models are rejected. The caller must
supply compatible prompt batches (the default collator needs equal stored lengths)
and ensure prompt plus continuation budgets fit both model context windows.

The former fixed-history implementation is explicitly named
`FixedHistoryDistillationTrainer` / `train_fixed_history_distillation`. It retains
its explicit causal-label/mask requirement and mixed task/KL objective. The old
two-argument `train_on_policy_distillation` call is intentionally not accepted by
the new protocol. The script's default main remains fail-closed instead of
launching its legacy demonstration automatically.

## Verification boundary

`tests/test_student_distillation.py` exercises synthetic CPU trajectories,
masking/causal shifts, token reduction, fresh generation, tokenizer rejection,
mode restoration, teacher gradient isolation, and one generation/backward pass
with random tiny Transformers models. It performs no optimizer steps, downloads,
provider jobs, or empirical training. An optional real Trainer initialization/loss test requires `accelerate>=1.1.0`;
it is dependency-skipped in the current verification environment because Accelerate
is absent. Real checkpoint/TRL-version compatibility, accelerator execution and
empirical performance remain unverified. Protocol identity is also attached to the
student model configuration so future Trainer checkpoints retain the v1 label.
