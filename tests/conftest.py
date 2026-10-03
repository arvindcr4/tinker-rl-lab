"""PyTest configuration and global fixtures.

Pre-import torch dynamo and inductor test operators.
This prevents unittest.mock.patch.dict(sys.modules, ...) in test suites
from removing dynamically loaded C++ operator libraries on unpatch.
"""

import torch  # noqa: F401
import torch._dynamo  # noqa: F401

try:
    import torch._inductor.test_operators  # noqa: F401
except Exception:
    pass
