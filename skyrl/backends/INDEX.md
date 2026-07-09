# skyrl/backends/ — INDEX

**Purpose:** Remote compute backends for running SkyRL tx (the local Tinker API server) on rented GPUs so any Tinker cookbook script can connect over the network.

**Key files:**
- `vastai_runner.py` — Python launcher (`python -m skyrl.backends.vastai_runner`); provisions vast.ai GPU instances over SSH (asyncssh), starts SkyRL tx, runs an algorithm/epochs job. `VastInstance`/`VastAILauncher`.
- `vastai_launch.sh` — shell launch script executed on the vast.ai instance.
- `__init__.py` — package marker.

**Find it fast:**
- to rent a GPU and serve the Tinker API → `vastai_runner.py`
- to customize the on-instance startup → `vastai_launch.sh`
