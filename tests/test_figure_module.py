from __future__ import annotations

import json
import os
from unittest.mock import patch

from platform_hybrid.paper.figure_module import (
    FallbackResultsAdapter,
    FigureModule,
    FigureRequest,
    JsonResultsAdapter,
)


def test_json_results_adapter_loads_and_names_source(tmp_path):
    source = tmp_path / "results.json"
    source.write_text(json.dumps({"measured": [1, 2, 3]}))
    adapter = JsonResultsAdapter(source)

    assert adapter.load() == {"measured": [1, 2, 3]}
    assert adapter.environment()["TINKERRL_FIGURE_RESULTS_PATH"] == str(source.resolve())


def test_figure_module_records_explicit_fallback_and_restores_environment(tmp_path):
    request = FigureRequest(
        profile="profiles",
        output_dir=tmp_path,
        results=FallbackResultsAdapter("test-snapshot"),
    )
    os.environ["TINKERRL_FIGURE_OUT_DIR"] = "before"

    with patch("platform_hybrid.paper.figure_module.runpy.run_path", return_value={}):
        manifest = FigureModule().render(request)

    assert os.environ["TINKERRL_FIGURE_OUT_DIR"] == "before"
    payload = json.loads(manifest.read_text())
    assert payload["profile"] == "profiles"
    assert payload["results_source"] == "fallback:test-snapshot"


def test_wave6_renderer_receives_records_through_results_adapter(tmp_path):
    source = tmp_path / "wave6.json"
    source.write_text(json.dumps({"metadata": {}, "temperature_sweep": []}))
    received = []

    def make_figure(data, out_png, out_pdf):
        received.append((data, out_png, out_pdf))

    request = FigureRequest(
        profile="wave6",
        output_dir=tmp_path / "out",
        results=JsonResultsAdapter(source),
    )
    with patch(
        "platform_hybrid.paper.figure_module.runpy.run_path",
        return_value={"make_figure": make_figure},
    ):
        FigureModule().render(request)

    assert received[0][0] == {"metadata": {}, "temperature_sweep": []}
    assert received[0][1].name == "wave6_sensitivity.png"
    assert received[0][2].name == "wave6_sensitivity.pdf"
