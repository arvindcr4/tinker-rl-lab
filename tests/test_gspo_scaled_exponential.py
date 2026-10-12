"""Independent Decimal oracles for representable scaled GSPO exponentials."""

from decimal import Decimal, localcontext

import pytest
import torch

from platform_tinker.tinkerrl.grpo import make_gspo_loss_fn


@pytest.mark.parametrize(
    ("dtype", "advantage", "drift"),
    [
        (torch.float32, -1e-5, 90.0),
        (torch.float32, 1e30, -110.0),
        (torch.float64, -1e-300, 710.0),
        (torch.float64, 1e300, -750.0),
        (torch.float16, -1e-38, 90.0),
        (torch.bfloat16, -1e-5, 90.0),
    ],
)
def test_representable_product_retains_value_and_gradient(dtype, advantage, drift):
    # Both policies have nonpositive token log-probabilities. Only their drift
    # has large magnitude; advantages scale the final objective back into range.
    new_value, old_value = (-1.0, -1.0 - drift) if drift > 0 else (-1.0 + drift, -1.0)
    values = torch.full((2,), new_value, dtype=dtype, requires_grad=True)
    with localcontext() as context:
        context.prec = 90
        expected = float(-Decimal.from_float(advantage) * Decimal.from_float(drift).exp())
    loss, metrics = make_gspo_loss_fn([advantage], old_logprobs=[[old_value] * 2])(None, [values])
    loss.backward()
    assert loss.item() == pytest.approx(expected, rel=2e-5, abs=0.0)
    expected_gradient = torch.full_like(values, expected / 2)
    torch.testing.assert_close(values.grad, expected_gradient, rtol=2e-5, atol=0.0)
    assert torch.isfinite(values.grad).all()
    assert metrics["gspo_clip_frac"] == 1.0
