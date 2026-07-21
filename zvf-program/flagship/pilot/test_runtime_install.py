from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pilot.runtime_install import (
    ABI_COUPLED_PACKAGES,
    RuntimeInstallError,
    install_runtime,
    load_install_request,
)


class RuntimeInstallTests(unittest.TestCase):
    def test_request_requires_unique_exact_pins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "request.json"
            path.write_text(json.dumps({"package_pins": ["torch==2.7.1"]}))
            self.assertEqual(load_install_request(path), ("torch==2.7.1",))
            path.write_text(json.dumps({"package_pins": ["torch>=2.7.1"]}))
            with self.assertRaisesRegex(RuntimeInstallError, "exact"):
                load_install_request(path)
            path.write_text(
                json.dumps({"package_pins": ["torch==2.7.1", "Torch==2.7.1"]})
            )
            with self.assertRaisesRegex(RuntimeInstallError, "duplicate"):
                load_install_request(path)

    def test_install_verifies_resulting_versions(self) -> None:
        calls = []

        def runner(command, **kwargs):
            calls.append((command, kwargs))
            return SimpleNamespace(returncode=0)

        observed = install_runtime(
            ("torch==2.7.1",),
            runner=runner,
            version_getter=lambda _: "2.7.1",
            spec_finder=lambda _: None,
        )
        self.assertEqual(observed, {"torch": "2.7.1"})
        self.assertEqual(
            calls[0][0][:6],
            ["uv", "pip", "install", "--system", "--reinstall-package", "numpy"],
        )
        self.assertEqual(
            calls[1][0],
            ["uv", "pip", "uninstall", "--system", *ABI_COUPLED_PACKAGES],
        )
        with self.assertRaisesRegex(RuntimeInstallError, "mismatch"):
            install_runtime(
                ("torch==2.7.1",),
                runner=runner,
                version_getter=lambda _: "2.8.0",
                spec_finder=lambda _: None,
            )

    def test_install_falls_back_to_pip_when_uv_fails(self) -> None:
        calls = []

        def runner(command, **kwargs):
            calls.append((command, kwargs))
            return SimpleNamespace(returncode=1 if command[0] == "uv" else 0)

        install_runtime(
            ("torch==2.7.1",),
            runner=runner,
            version_getter=lambda _: "2.7.1",
            spec_finder=lambda _: None,
        )
        self.assertEqual(calls[1][0][1:4], ["-m", "pip", "install"])
        self.assertTrue(calls[1][1]["check"])

    def test_install_fails_closed_when_abi_coupled_package_remains(self) -> None:
        def runner(command, **kwargs):
            return SimpleNamespace(returncode=0)

        with self.assertRaisesRegex(RuntimeInstallError, "torchao"):
            install_runtime(
                ("torch==2.7.1",),
                runner=runner,
                version_getter=lambda _: "2.7.1",
                spec_finder=lambda name: object() if name == "torchao" else None,
            )


if __name__ == "__main__":
    unittest.main()
