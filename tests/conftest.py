"""PyTest configuration and global fixtures.

Pre-import torch dynamo and inductor test operators.
This prevents unittest.mock.patch.dict(sys.modules, ...) in test suites
from removing dynamically loaded C++ operator libraries on unpatch.
"""

import contextlib

import torch
import torch._dynamo

# Optional: absent or failing to load on some torch builds; best-effort pre-import only.
with contextlib.suppress(Exception):
    import torch._inductor.test_operators  # noqa: F401  (side-effect import)
