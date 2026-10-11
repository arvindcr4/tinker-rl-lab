"""Loss factories bind rollout inputs across the asynchronous SDK boundary."""

import asyncio

import numpy as np
import pytest
import torch

from platform_tinker.tinkerrl.grpo import make_grpo_loss_fn, make_gspo_loss_fn


def after_yield(loss_fn, logprobs):
    async def invoke():
        await asyncio.sleep(0)
        loss, metrics = loss_fn(None, logprobs)
        loss.backward()
        return loss.item(), metrics

    return asyncio.run(invoke())


@pytest.mark.parametrize("factory", [make_grpo_loss_fn, make_gspo_loss_fn])
def test_response_mask_is_bound_before_callback(factory):
    lengths = [1]
    loss_fn = factory([1.0], response_lens=lengths)
    lengths[0] = 2
    logprobs = torch.tensor([-10.0, -1.0], requires_grad=True)
    after_yield(loss_fn, [logprobs])
    assert logprobs.grad.tolist() == [0.0, -1.0]


def test_nll_mask_is_bound_before_callback():
    mask = np.array([True, False])
    loss_fn = make_grpo_loss_fn([0.0, 0.0], nll_mask=mask, nll_coef=1.0)
    mask[:] = [False, True]
    logprobs = [torch.tensor([-1.0], requires_grad=True), torch.tensor([-3.0], requires_grad=True)]
    loss, metrics = after_yield(loss_fn, logprobs)
    assert loss == metrics["nll_loss"] == 1.0
    assert [value.grad.item() for value in logprobs] == [-1.0, 0.0]


@pytest.mark.parametrize("kind", ["list", "numpy", "tensor", "nonleaf_tensor"])
def test_behavior_tokens_are_copied_without_losing_precision(kind):
    value = -1.00000000001
    original = torch.tensor([value], dtype=torch.float64, requires_grad=True)
    if kind == "list":
        row = [value]
    elif kind == "numpy":
        row = np.array([value], dtype=np.float64)
    elif kind == "tensor":
        row = original
    else:
        row = original * 1.0
    old = [row]
    loss_fn = make_gspo_loss_fn([1.0], old_logprobs=old)
    with torch.no_grad():
        row[0] = -10.0
    new = torch.tensor([value], dtype=torch.float64, requires_grad=True)
    loss, metrics = after_yield(loss_fn, [new])
    assert loss == -1.0
    assert metrics["gspo_clip_frac"] == 0.0
    assert new.grad.item() == -1.0
    assert original.grad is None


def test_outer_behavior_sequence_is_bound_and_stash_remains_output():
    old = [[-1.0]]
    stash = []
    loss_fn = make_gspo_loss_fn([1.0], old_logprobs=old, stash=stash, response_lens=[1])
    old.clear()
    first = torch.tensor([-9.0, -1.0], requires_grad=True)
    after_yield(loss_fn, [first])
    assert len(stash) == 1
    assert stash[0].tolist() == [-1.0]
    assert not stash[0].requires_grad
    with torch.no_grad():
        first.fill_(-3.0)
    after_yield(loss_fn, [torch.tensor([-8.0, -2.0], requires_grad=True)])
    assert len(stash) == 1
    assert stash[0].tolist() == [-1.0]
