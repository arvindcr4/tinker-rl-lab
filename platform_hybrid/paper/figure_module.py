"""Deep figure module for publication rendering and provenance."""

from __future__ import annotations

import argparse
import json
import os
import runpy
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CANONICAL_FIGURES = HERE / "figures"
DEFAULT_RESULTS = REPO_ROOT / "platform_hybrid/experiments/all_results_consolidated.json"
DEFAULT_WAVE6_RESULTS = (
    REPO_ROOT / "platform_hybrid/experiments/tinker-runs/results/wave6_ablations.json"
)


class ResultsAdapter(Protocol):
    """Supplies figure records and their provenance label."""

    @property
    def source(self) -> str: ...
    def load(self) -> Any: ...
    def environment(self) -> Mapping[str, str]: ...


@dataclass(frozen=True, slots=True)
class JsonResultsAdapter:
    path: Path

    @property
    def source(self) -> str:
        try:
            return str(self.path.resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(self.path.resolve())

    def load(self) -> Any:
        return json.loads(self.path.read_text())

    def environment(self) -> Mapping[str, str]:
        return {"TINKERRL_FIGURE_RESULTS_PATH": str(self.path.resolve())}


@dataclass(frozen=True, slots=True)
class FallbackResultsAdapter:
    """Names an explicit embedded publication snapshot."""

    name: str = "embedded-publication-snapshot"

    @property
    def source(self) -> str:
        return f"fallback:{self.name}"

    def load(self) -> Any:
        return None

    def environment(self) -> Mapping[str, str]:
        return {"TINKERRL_FIGURE_RESULTS_SOURCE": self.source}


@dataclass(frozen=True, slots=True)
class FigureRequest:
    profile: str
    output_dir: Path
    results: ResultsAdapter


@contextmanager
def _temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class FigureModule:
    """Owns renderer selection, results provenance, and output manifests."""

    renderers = {
        "profiles": CANONICAL_FIGURES / "gen_figures.py",
        "paper": CANONICAL_FIGURES / "generate_figures.py",
        "wave6": CANONICAL_FIGURES / "wave6_sensitivity.py",
    }

    def render(self, request: FigureRequest) -> Path:
        if request.profile not in self.renderers:
            raise ValueError(f"unknown figure profile: {request.profile}")
        request.output_dir.mkdir(parents=True, exist_ok=True)
        environment = {
            "TINKERRL_FIGURE_OUT_DIR": str(request.output_dir.resolve()),
            **request.results.environment(),
        }
        renderer = self.renderers[request.profile]
        with _temporary_environment(environment):
            namespace = runpy.run_path(str(renderer), run_name="_tinkerrl_figure_renderer")
            if request.profile == "wave6":
                out_png = request.output_dir / "wave6_sensitivity.png"
                namespace["make_figure"](
                    request.results.load(), out_png, out_png.with_suffix(".pdf")
                )

        manifest = request.output_dir / f"{request.profile}.figure-manifest.json"
        payload = {
            "profile": request.profile,
            "results_source": request.results.source,
            "renderer": str(renderer.relative_to(REPO_ROOT)),
            "outputs": sorted(
                path.name
                for path in request.output_dir.iterdir()
                if path.suffix.lower() in {".pdf", ".png"}
            ),
        }
        temporary = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, manifest)
        return manifest


def default_request(profile: str, output_dir: Path) -> FigureRequest:
    if profile == "profiles":
        adapter: ResultsAdapter = FallbackResultsAdapter()
    elif profile == "paper":
        adapter = JsonResultsAdapter(DEFAULT_RESULTS)
    elif profile == "wave6":
        adapter = JsonResultsAdapter(DEFAULT_WAVE6_RESULTS)
    else:
        raise ValueError(f"unknown figure profile: {profile}")
    return FigureRequest(profile=profile, output_dir=output_dir, results=adapter)


def render_legacy_figure(profile: str, output_dir: Path) -> Path:
    """Compatibility interface used by historical figure paths."""
    return FigureModule().render(default_request(profile, output_dir))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["all", *FigureModule.renderers], default="all")
    parser.add_argument("--out", type=Path, default=CANONICAL_FIGURES)
    args = parser.parse_args()

    profiles = FigureModule.renderers if args.profile == "all" else [args.profile]
    module = FigureModule()
    for profile in profiles:
        manifest = module.render(default_request(profile, args.out))
        print(f"[{profile}] manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
