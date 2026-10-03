"""Tests for the run_all_audits runner mechanics (stubs only).

The real audits read the working tree; these tests pin the runner itself:
registry shape, custom-audit selection, shared context, and exit codes.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import platform_local.run_all_audits as raa


def _passing_audit(context):
    return iter(())


def _failing_audit(context):
    return iter(["STUB-001"])


def test_registry_lists_nine_callable_audits():
    assert len(raa.AUDITS) == 9
    names = [name for name, _ in raa.AUDITS]
    assert len(set(names)) == 9
    for _, fn in raa.AUDITS:
        assert callable(fn)


def test_run_suite_passes_when_all_audits_pass():
    result = raa.run_suite(audits=(("ok_a", _passing_audit), ("ok_b", _passing_audit)))
    assert result.passed
    assert result.failures == ()


def test_run_suite_collects_failures():
    result = raa.run_suite(audits=(("ok", _passing_audit), ("bad", _failing_audit)))
    assert not result.passed
    assert [r.name for r in result.failures] == ["bad"]


def test_run_suite_shares_one_context():
    seen = []

    def spy(context):
        seen.append(context)
        return iter(())

    context = Mock()
    raa.run_suite(audits=(("a", spy), ("b", spy)), context=context)
    assert seen == [context, context]


def test_main_exit_codes_follow_suite_result():
    with (
        patch.object(raa, "run_suite") as run_suite,
        patch.object(raa, "render_suite") as render_suite,
    ):
        run_suite.return_value = Mock(passed=True)
        assert raa.main() == 0
        run_suite.return_value = Mock(passed=False)
        assert raa.main() == 1
        assert render_suite.call_count == 2
