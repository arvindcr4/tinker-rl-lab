"""
SkyRL Backends Package

Supports multiple compute backends for SkyRL tx (Tinker API server):
- local: Run on local GPUs using SkyRL tx
- vastai: Provision and run on vast.ai GPU instances
- colab: Run in Google Colab environment

TODO: Address methodological limitations from the adversarial review:
- "Early-Training Snapshot" & API Cost Constraints: Need cheaper, scalable backends to support full training runs (>30-50 steps).
- "Single-Seed Extrapolations": Automate multi-seed provisioning across backends to ensure statistical significance.
- "Closed-Source Confound": Expand transparent, open-source local/remote backend alternatives to the managed Tinker API.
- "Failure to Prove Generalization": Integrate held-out test set evaluation pipelines natively into backend execution.
"""

__all__ = ["vastai_runner"]
