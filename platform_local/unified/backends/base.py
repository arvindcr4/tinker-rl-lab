"""Backend dispatch interface.

A ``Backend`` knows **where** to run one experiment cell (local GPU, Modal, Colab,
vast.ai, GCP, HF Spaces); the framework dimension (trl/tinker/verl/openrlhf/skyrl)
is the **what**. ``plan()`` resolves a cell into a concrete ``LaunchPlan`` without
spending compute, so ``--dry-run`` and the matrix test work uniformly.
"""
from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid runtime import cycle
    from platform_local.unified.canonical import CanonicalSpec


@dataclass
class LaunchPlan:
    """Resolved description of how a backend runs one (framework) cell."""

    backend: str
    framework: str
    command: str  # exact command (modal/gcloud/python/...)
    driver_file: str  # repo file that actually executes the training
    output: str = ""  # where results land
    env: list[str] = field(default_factory=list)  # required env vars
    notes: str = ""

    def format(self) -> str:
        lines = [
            f"[{self.backend}/{self.framework}] {self.command}",
            f"  driver: {self.driver_file}",
        ]
        if self.output:
            lines.append(f"  output: {self.output}")
        if self.env:
            lines.append(f"  env:    {', '.join(self.env)}")
        if self.notes:
            lines.append(f"  notes:  {self.notes}")
        return "\n".join(lines)


class Backend(ABC):
    """Abstract compute backend."""

    name: str = ""

    @abstractmethod
    def plan(self, framework: str, spec: "CanonicalSpec") -> LaunchPlan:
        """Resolve the cell into a LaunchPlan (no compute)."""

    def run(
        self,
        framework: str,
        spec: "CanonicalSpec",
        *,
        dry_run: bool = False,
        launcher=None,
    ):
        """Execute the cell. Default impl prints the plan and shells out."""
        plan = self.plan(framework, spec)
        print(plan.format())
        if dry_run:
            return None
        return self._execute(plan, spec, launcher=launcher)

    def _execute(self, plan: LaunchPlan, spec: "CanonicalSpec", *, launcher=None):
        subprocess.run(plan.command, shell=True, check=True)
        return None
