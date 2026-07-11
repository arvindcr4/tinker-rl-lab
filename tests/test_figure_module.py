from __future__ import annotations

import json
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
    assert adapter.source == str(source.resolve())


def test_fallback_adapter_supplies_the_embedded_records():
    records = FallbackResultsAdapter("test-snapshot").load()

    assert records["traces"]["GRPO-Qwen3-8B"]
    assert records["trl_accuracies"]


def test_figure_module_loads_records_for_every_profile(tmp_path):
    class RecordingAdapter:
        source = "recording:test"

        def __init__(self):
            self.calls = 0

        def load(self):
            self.calls += 1
            return {"profile-data": self.calls}

    adapter = RecordingAdapter()
    request = FigureRequest(
        profile="profiles",
        output_dir=tmp_path,
        results=adapter,
    )

    def render_outputs(*_args, **_kwargs):
        for name in FigureModule.outputs["profiles"]:
            (tmp_path / name).touch()
        return {}

    with patch(
        "platform_hybrid.paper.figure_module.runpy.run_path",
        side_effect=render_outputs,
    ) as run:
        manifest = FigureModule().render(request)

    assert adapter.calls == 1
    assert run.call_args.kwargs["init_globals"]["FIGURE_RECORDS"] == {"profile-data": 1}
    assert run.call_args.kwargs["init_globals"]["FIGURE_OUTPUT_DIR"] == tmp_path.resolve()
    payload = json.loads(manifest.read_text())
    assert payload["profile"] == "profiles"
    assert payload["results_source"] == "recording:test"


def test_wave6_uses_the_same_records_interface(tmp_path):
    source = tmp_path / "wave6.json"
    source.write_text(json.dumps({"metadata": {}, "temperature_sweep": []}))
    request = FigureRequest(
        profile="wave6",
        output_dir=tmp_path / "out",
        results=JsonResultsAdapter(source),
    )

    def render_outputs(*_args, **_kwargs):
        for name in FigureModule.outputs["wave6"]:
            (request.output_dir / name).touch()
        return {}

    with patch(
        "platform_hybrid.paper.figure_module.runpy.run_path",
        side_effect=render_outputs,
    ) as run:
        FigureModule().render(request)

    assert run.call_args.kwargs["init_globals"]["FIGURE_RECORDS"] == {
        "metadata": {},
        "temperature_sweep": [],
    }


def test_figure_module_rejects_an_incomplete_renderer(tmp_path):
    request = FigureRequest(
        profile="profiles",
        output_dir=tmp_path,
        results=FallbackResultsAdapter(),
    )

    with patch("platform_hybrid.paper.figure_module.runpy.run_path", return_value={}):
        try:
            FigureModule().render(request)
        except RuntimeError as error:
            assert "did not create expected outputs" in str(error)
        else:
            raise AssertionError("incomplete renderer must fail")
