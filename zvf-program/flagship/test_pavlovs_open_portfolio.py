from __future__ import annotations

import copy
import unittest

from flagship.pavlovs_domain_contract import load_contract
from flagship.pavlovs_open_portfolio import (
    build_open_contract,
    load_overlay,
    validate_open_portfolio,
)


class PavlovsOpenPortfolioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = load_contract()
        self.overlay = load_overlay()

    def test_open_overlay_is_complete_and_domain_valid(self) -> None:
        assert validate_open_portfolio(self.base, self.overlay) == []
        effective = build_open_contract(self.base, self.overlay)
        primary = {
            suite_id
            for suite_id, suite in effective["suite_registry"].items()
            if suite["role"] == "primary_eval"
        }
        assert len(primary) == 14
        assert "omni_math_eval" in primary
        assert "frontiermath_eval" not in primary

    def test_private_replacement_fails_closed(self) -> None:
        candidate = copy.deepcopy(self.overlay)
        candidate["replacements"]["frontiermath_eval"]["requires_private_assets"] = True
        errors = validate_open_portfolio(self.base, candidate)
        assert "frontiermath_eval: private assets are not allowed" in errors

    def test_unlicensed_replacement_fails_closed(self) -> None:
        candidate = copy.deepcopy(self.overlay)
        candidate["replacements"]["frontier_swe_eval"]["dataset_license"] = "NOASSERTION"
        errors = validate_open_portfolio(self.base, candidate)
        assert "frontier_swe_eval: dataset_license is not in the explicit allowlist" in errors

    def test_unpinned_replacement_fails_closed(self) -> None:
        candidate = copy.deepcopy(self.overlay)
        candidate["replacements"]["webbench_eval"]["source_revision"] = "main"
        errors = validate_open_portfolio(self.base, candidate)
        assert "webbench_eval: source_revision must be a 40-hex pin" in errors

    def test_historical_contract_is_not_mutated(self) -> None:
        before = copy.deepcopy(self.base)
        build_open_contract(self.base, self.overlay)
        assert self.base == before


if __name__ == "__main__":
    unittest.main()
