"""
SkyRL Backends Package

Supports multiple compute backends for SkyRL tx (Tinker API server):
- local: Run on local GPUs using SkyRL tx
- vastai: Provision and run on vast.ai GPU instances
- colab: Run in Google Colab environment
"""

__all__ = ["vastai_runner"]
