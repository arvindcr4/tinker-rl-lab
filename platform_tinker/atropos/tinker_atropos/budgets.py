"""Nominal generated-token budget sizing (protocol v2), including accumulation."""


def budget_steps(token_budget, tokens_per_sample, per_device_batch, accumulation, world_size=1):
    values = [token_budget, tokens_per_sample, per_device_batch, accumulation, world_size]
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values):
        raise ValueError("Token budget and batch dimensions must be positive integers")
    tokens_per_update = tokens_per_sample * per_device_batch * accumulation * world_size
    steps = token_budget // tokens_per_update
    if steps < 1:
        raise ValueError("Token budget is smaller than one full optimizer update")
    return steps
