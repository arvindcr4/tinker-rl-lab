"""Malformed training results must stop before an optimizer update."""

from types import SimpleNamespace

import pytest

from tests._shared_fakes import _config, _Future
from tests.test_grpo_caliber_fixes import _run, _runtime


def override_result_metrics(monkeypatch, rows, after_optimizer=None):
    import tinker

    original = tinker.ServiceClient
    responses = iter(rows)

    class AlteredResultService(original):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            forward = self.client.forward_backward_custom

            def altered(*args, **kwargs):
                forward(*args, **kwargs).result()
                return _Future(SimpleNamespace(metrics=next(responses, {})))

            self.client.forward_backward_custom = altered
            if after_optimizer is not None:
                optim = self.client.optim_step

                def update(*args, **kwargs):
                    result = optim(*args, **kwargs)
                    after_optimizer()
                    return result

                self.client.optim_step = update

    monkeypatch.setattr(tinker, "ServiceClient", AlteredResultService)


@pytest.mark.parametrize("gspo", [False, True])
@pytest.mark.parametrize(
    "metrics",
    [
        None,
        [],
        {},
        {"loss": float("nan")},
        {"loss": float("inf")},
        {"loss": 10**400},
        {"loss": True},
        {"loss": "0.1"},
        {"loss": []},
    ],
)
def test_invalid_loss_receipt_blocks_optimizer_and_completion(tmp_path, monkeypatch, gspo, metrics):
    with _runtime(monkeypatch) as holder:
        override_result_metrics(monkeypatch, [metrics])
        with pytest.raises(ValueError, match="training result must contain a finite numeric loss"):
            _run(_config(tmp_path, name="bad-result", gspo_enabled=gspo))
    assert holder["optim_calls"] == 0
    assert not (tmp_path / "bad-result_seed42.json").exists()
    assert not any("train/loss" in row for row in holder["logs"])


@pytest.mark.parametrize("alias", ["grpo_loss", "gspo_loss", "loss"])
def test_valid_zero_loss_alias_still_updates(tmp_path, monkeypatch, alias):
    with _runtime(monkeypatch) as holder:
        override_result_metrics(monkeypatch, [{alias: -0.0}])
        result = _run(_config(tmp_path, name="valid-zero"))
    assert holder["optim_calls"] == 1
    assert result.zero_loss_steps == 1
    assert (tmp_path / "valid-zero_seed42.json").exists()


def test_later_gspo_epoch_invalid_loss_stops_remaining_updates(tmp_path, monkeypatch):
    with _runtime(monkeypatch) as holder:
        override_result_metrics(monkeypatch, [{"gspo_loss": 0.1}, {}])
        with pytest.raises(ValueError, match="training result must contain a finite numeric loss"):
            _run(_config(tmp_path, name="bad-epoch", gspo_enabled=True, gspo_update_epochs=3))
    assert holder["optim_calls"] == 1
    assert len(holder["grads"]) == 2
    assert not (tmp_path / "bad-epoch_seed42.json").exists()


def test_validated_loss_value_is_not_rebound_by_optimizer_callback(tmp_path, monkeypatch):
    metrics = {"loss": 0.25}
    with _runtime(monkeypatch) as holder:
        override_result_metrics(
            monkeypatch, [metrics], after_optimizer=lambda: metrics.update(loss=float("nan"))
        )
        _run(_config(tmp_path, name="loss-snapshot"))
    assert holder["optim_calls"] == 1
    assert [row["train/loss"] for row in holder["logs"] if "train/loss" in row] == [0.25]
