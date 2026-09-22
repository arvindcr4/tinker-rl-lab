from __future__ import annotations

import unittest

from . import modal_tinker_openai_bridge as bridge


class BridgeModelIdentityTests(unittest.TestCase):
    def test_accepts_pinned_alias_with_litellm_provider_prefix(self) -> None:
        self.assertTrue(bridge.is_supported_request_model(bridge.MODEL_ALIAS))
        self.assertTrue(bridge.is_supported_request_model(f"openai/{bridge.MODEL_ALIAS}"))

    def test_rejects_every_unpinned_alias(self) -> None:
        self.assertFalse(bridge.is_supported_request_model("openai/other-model"))
        self.assertFalse(bridge.is_supported_request_model("other-model"))


if __name__ == "__main__":
    unittest.main()
