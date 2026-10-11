"""Arithmetic contract for synchronous OpenRLHF 0.8.4 dataset-budget launches.

This plans complete batches; it neither counts executed updates nor adds an
optimizer-step stop hook to the Ray actors. Resume and dynamic filtering are
not exposed by this wrapper.
"""

from .config import OpenRLHFConfig


def plan_native_budget(config: OpenRLHFConfig) -> dict:
    if config.max_steps is not None:
        raise ValueError(
            "OpenRLHF v0.8.4 has no native max_steps CLI contract; "
            "use an explicit max_samples dataset budget with max_steps=None. "
            "A dataset budget does not guarantee executed optimizer steps."
        )
    sizes = (
        config.epochs,
        config.train_batch_size,
        config.micro_batch_size,
        config.num_gpus,
        config.rollout_batch_size,
        config.max_samples,
        config.algorithm.sample_num,
    )
    if any(value < 1 for value in sizes):
        raise ValueError("Native training batch sizes, epochs and GPU counts must be positive")
    if config.max_samples % config.rollout_batch_size:
        raise ValueError(
            "max_samples must contain whole rollout batches (native loader drops tails)"
        )
    global_micro_batch = config.micro_batch_size * config.num_gpus
    if config.train_batch_size % global_micro_batch:
        raise ValueError("train_batch_size must divide into whole distributed micro-batches")
    completions = config.rollout_batch_size * config.algorithm.sample_num
    if completions % config.train_batch_size:
        raise ValueError(
            "Each rollout must contain whole global training batches: "
            "rollout_batch_size * sample_num must be divisible by train_batch_size. "
            "Accumulation spanning inner epochs or actor fits is outside the supported native mode."
        )
    rollouts = config.max_samples // config.rollout_batch_size
    updates_per_rollout = completions // config.train_batch_size * config.epochs
    return {
        "schema": "openrlhf-native-dataset-budget-v1",
        "mode": "synchronous_whole_batches",
        "scope": "arithmetic plan only; not measured optimizer updates or a max_steps stop hook",
        "prompt_rows": config.max_samples,
        "completions_per_rollout": completions,
        "planned_rollouts": rollouts,
        "micro_batches_per_actor_per_epoch": completions // global_micro_batch,
        "gradient_accumulation_micro_batches": config.train_batch_size // global_micro_batch,
        "planned_update_boundaries_per_rollout": updates_per_rollout,
        "planned_update_boundaries": rollouts * updates_per_rollout,
    }
