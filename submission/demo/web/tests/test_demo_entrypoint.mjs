import assert from 'node:assert/strict';
import test from 'node:test';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const entrypoint = fileURLToPath(new URL('./test_demo.py', import.meta.url));

function contract(script) {
  const setup = `
import builtins
import importlib.util
import os
import pathlib
import sys
import tempfile
from unittest.mock import patch

# Never import Playwright or launch a browser, even if preflight regresses.
browser_imports = []
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "playwright" or name.startswith("playwright."):
        browser_imports.append(name)
        raise AssertionError("Browser imports are prohibited in this contract test")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
spec = importlib.util.spec_from_file_location("demo_browser_checks", sys.argv[1])
demo = importlib.util.module_from_spec(spec)
with patch.dict(os.environ, {}, clear=True):
    spec.loader.exec_module(demo)
assert not browser_imports
`;
  const result = spawnSync(process.env.PYTHON || 'python3', ['-c', setup + script, entrypoint], {
    encoding: 'utf8',
  });
  assert.ifError(result.error);
  assert.equal(result.status, 0, result.stdout + result.stderr);
}

test('direct browser checks default to dist/index.html even when the frozen page exists', () => {
  contract(`
with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
    demo.HERE = pathlib.Path(directory)
    (demo.HERE / "tinkerrl_demo.html").write_text("frozen historical page")
    built = demo.HERE / "dist/index.html"
    built.parent.mkdir()
    built.write_text("new source build")
    assert demo.demo_page() == built.as_uri()
assert not browser_imports
`);
});

test('a missing build fails with build guidance before any Playwright import', () => {
  contract(`
with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
    demo.HERE = pathlib.Path(directory)
    (demo.HERE / "tinkerrl_demo.html").write_text("must not fall back to history")
    try:
        demo.main()
    except SystemExit as error:
        message = str(error)
        assert "Demo page not found:" in message
        assert str(demo.HERE / "dist/index.html") in message
        assert "npx vite build" in message
        assert "python3 tests/test_demo.py" in message
        assert "DEMO_PAGE" in message
    else:
        raise AssertionError("Missing build unexpectedly passed preflight")
assert not browser_imports
`);
});

test('an explicit DEMO_PAGE preserves caller-relative and absolute paths', () => {
  contract(`
with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
    demo.HERE = pathlib.Path(directory) / "unbuilt-web"
    target = pathlib.Path(directory) / "custom page #1.html"
    target.write_text("explicit source build")
    os.environ["DEMO_PAGE"] = str(target)
    assert demo.demo_page() == target.as_uri()
    os.chdir(directory)
    os.environ["DEMO_PAGE"] = target.name
    assert demo.demo_page() == target.as_uri()
assert not browser_imports
`);
});

test('missing or nonfile DEMO_PAGE overrides fail without falling back to the default build', () => {
  contract(`
with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
    demo.HERE = pathlib.Path(directory)
    built = demo.HERE / "dist/index.html"
    built.parent.mkdir()
    built.write_text("valid default build")
    for target in (demo.HERE / "missing.html", built.parent):
        os.environ["DEMO_PAGE"] = str(target)
        try:
            demo.main()
        except SystemExit as error:
            assert "Demo page not found:" in str(error)
            assert str(target) in str(error)
        else:
            raise AssertionError("Invalid override unexpectedly passed preflight")
assert not browser_imports
`);
});
