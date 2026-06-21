"""Lazy W&B / TensorBoard logging for ZVF triage metrics.

No hard dependency on wandb or tensorboard: backends are imported only when a
panel is constructed (or when ``log`` first runs). If neither is available the
panel degrades to a no-op (or an in-memory buffer), so the package always
imports and a training loop never crashes for lack of a logger.

Logged series:
    zvf, gu, rolling_zvf, rolling_variance, peak_to_tail_drift, mean_reward,
    group_size, dropped_prompt_count, regime (as a categorical code).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Stable integer codes for the categorical "regime" series so it can be plotted.
_REGIME_CODE = {
    "exploitable_contrast": 1,
    "saturation": 2,
    "cold_start_collapse": 3,
}


class ZVFPanel:
    """Backend-agnostic metric sink.

    Args:
        backend: "auto" | "wandb" | "tensorboard" | "memory" | "none".
            "auto" tries wandb, then tensorboard, then falls back to "memory".
        prefix: metric name prefix (default "zvf/").
        wandb_run: an already-initialised wandb run to log into (optional).
        tb_logdir: TensorBoard log directory (used only for the tensorboard
            backend when no writer is supplied).
        tb_writer: an already-constructed ``SummaryWriter`` (optional).
    """

    def __init__(
        self,
        backend: str = "auto",
        prefix: str = "zvf/",
        wandb_run: Any = None,
        tb_logdir: Optional[str] = None,
        tb_writer: Any = None,
    ) -> None:
        self.prefix = prefix
        self.backend = "none"
        self._wandb = None
        self._wandb_run = None
        self._tb_writer = None
        self.history: List[Dict[str, Any]] = []  # always kept (cheap, in-memory)

        if backend in ("auto", "wandb"):
            ok = self._try_wandb(wandb_run)
            if ok:
                self.backend = "wandb"
            elif backend == "wandb":
                # explicit request but unavailable -> stay in-memory, warn-free
                self.backend = "memory"
        if self.backend == "none" and backend in ("auto", "tensorboard"):
            ok = self._try_tensorboard(tb_writer, tb_logdir)
            if ok:
                self.backend = "tensorboard"
            elif backend == "tensorboard":
                self.backend = "memory"
        if self.backend == "none":
            if backend == "none":
                self.backend = "none"
            else:
                self.backend = "memory"

    # ------------------------------------------------------------------ setup

    def _try_wandb(self, wandb_run: Any) -> bool:
        try:
            import wandb  # noqa: WPS433 (lazy import by design)
        except Exception:
            return False
        self._wandb = wandb
        if wandb_run is not None:
            self._wandb_run = wandb_run
            return True
        # Only attach to an already-running wandb session; never auto-init one.
        run = getattr(wandb, "run", None)
        if run is not None:
            self._wandb_run = run
            return True
        return False

    def _try_tensorboard(self, tb_writer: Any, tb_logdir: Optional[str]) -> bool:
        if tb_writer is not None:
            self._tb_writer = tb_writer
            return True
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: WPS433
        except Exception:
            try:
                from tensorboardX import SummaryWriter  # type: ignore  # noqa: WPS433
            except Exception:
                return False
        if tb_logdir is None:
            return False
        self._tb_writer = SummaryWriter(log_dir=tb_logdir)
        return True

    # ------------------------------------------------------------------- log

    def log(self, decision: Any, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Log one step. ``decision`` is a controller.StepDecision (or dict).

        Returns the flat metrics dict that was logged (also appended to
        ``self.history``).
        """
        d = decision.as_dict() if hasattr(decision, "as_dict") else dict(decision)
        # Include all numeric/scalar fields from the StepDecision, not
        # just a curated subset. The previous version silently
        # dropped ``signal``, ``n_groups``, ``filtered_rewards``,
        # and ``notes`` from the logged payload (only ``regime_code``
        # and a handful of scalars were emitted). Now every scalar
        # field except lists (dropped_prompts is logged as
        # dropped_prompt_count) is forwarded.
        metrics: Dict[str, Any] = {
            "zvf": d.get("zvf"),
            "gu": d.get("gu"),
            "rolling_zvf": d.get("rolling_zvf"),
            "mean_reward": d.get("mean_reward"),
            "group_size": d.get("group_size"),
            "dropped_prompt_count": len(d.get("dropped_prompts", [])),
            "filtered_rewards": d.get("filtered_rewards"),
            "auto_stop": int(bool(d.get("auto_stop"))),
            "regime_code": _REGIME_CODE.get(d.get("regime", ""), -1),
            "regime": d.get("regime"),
            "signal": d.get("signal"),
            "n_groups": d.get("n_groups"),
        }
        if extra:
            metrics.update(extra)

        step = d.get("step")
        self.history.append({"step": step, **metrics})
        self._emit(metrics, step)
        return metrics

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        self._emit({name: value}, step)

    def _emit(self, metrics: Dict[str, Any], step: Optional[int]) -> None:
        payload = {f"{self.prefix}{k}": v for k, v in metrics.items() if v is not None}
        if not payload:
            return
        if self.backend == "wandb" and self._wandb_run is not None:
            try:
                self._wandb_run.log(payload, step=step)
            except Exception:
                pass
        elif self.backend == "tensorboard" and self._tb_writer is not None:
            for name, value in payload.items():
                try:
                    self._tb_writer.add_scalar(name, float(value), global_step=step)
                except Exception:
                    pass
        # "memory" / "none": history already captured (memory) or dropped (none).

    def close(self) -> None:
        if self._tb_writer is not None:
            try:
                self._tb_writer.close()
            except Exception:
                pass
