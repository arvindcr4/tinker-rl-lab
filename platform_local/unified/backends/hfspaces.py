"""HF SPACES backend — results demo + fetch (no GPU training, by design).

A Space cannot host GPU RL training, so this backend does not launch training.
``plan()`` resolves to the results-fetch entry, which loads experiment outputs
from HF Hub / W&B / GCS into the Space's dashboard.
"""
from __future__ import annotations

from platform_local.unified.backends.base import Backend, LaunchPlan

_FETCH = "platform_hf_spaces/defense_live_demo/fetch_results.py"
_APP = "platform_hf_spaces/defense_live_demo/app.py"


class HFSpacesBackend(Backend):
    name = "hfspaces"

    def plan(self, framework: str, spec) -> LaunchPlan:
        return LaunchPlan(
            backend="hfspaces",
            framework=framework,
            command=f"python {_FETCH} --framework {framework} --task {spec.task}",
            driver_file=_FETCH,
            output="Gradio dashboard tab (Live Results) in the Space",
            env=["HF_TOKEN", "WANDB_API_KEY"],
            notes=(
                "no training (HF Spaces has no GPU); fetches results produced on "
                f"other backends. Dashboard: {_APP}"
            ),
        )

    def run(self, framework: str, spec, *, dry_run: bool = False, launcher=None):
        plan = self.plan(framework, spec)
        print(plan.format())
        if dry_run:
            return None
        # Execute the fetch (it is safe / read-only); no training result to return.
        import subprocess

        subprocess.run(plan.command, shell=True, check=True)
        return None
