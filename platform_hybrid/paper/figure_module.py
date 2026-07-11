"""Deep figure module for publication rendering and provenance."""

from __future__ import annotations

import argparse
import json
import os
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CANONICAL_FIGURES = HERE / "figures"
DEFAULT_RESULTS = REPO_ROOT / "platform_hybrid/experiments/all_results_consolidated.json"
DEFAULT_WAVE6_RESULTS = (
    REPO_ROOT / "platform_hybrid/experiments/tinker-runs/results/wave6_ablations.json"
)
DEFAULT_PROFILE_RESULTS = CANONICAL_FIGURES / "performance_profiles_snapshot.json"


class ResultsAdapter(Protocol):
    """Supplies figure records and their provenance label."""

    @property
    def source(self) -> str: ...
    def load(self) -> Any: ...


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


@dataclass(frozen=True, slots=True)
class FallbackResultsAdapter:
    """Supplies the explicit publication snapshot used when measured data is absent."""

    name: str = "embedded-publication-snapshot"
    path: Path = DEFAULT_PROFILE_RESULTS

    @property
    def source(self) -> str:
        return f"fallback:{self.name}"

    def load(self) -> Any:
        return json.loads(self.path.read_text())


@dataclass(frozen=True, slots=True)
class FigureRequest:
    profile: str
    output_dir: Path
    results: ResultsAdapter


class FigureModule:
    """Owns renderer selection, results provenance, and output manifests."""

    renderers = {
        "profiles": CANONICAL_FIGURES / "gen_figures.py",
        "paper": CANONICAL_FIGURES / "generate_figures.py",
        "wave6": CANONICAL_FIGURES / "wave6_sensitivity.py",
    }
    outputs = {
        "profiles": (
            "performance_profiles.pdf",
            "performance_profiles.png",
            "sensitivity_heatmap.pdf",
            "sensitivity_heatmap.png",
        ),
        "paper": (
            "comparison_bars.png",
            "learning_curves.png",
            "ppo_vs_grpo_comparison.png",
            "scaling_plot.png",
            "sensitivity_heatmap.png",
        ),
        "wave6": ("wave6_sensitivity.pdf", "wave6_sensitivity.png"),
    }

    def render(self, request: FigureRequest) -> Path:
        if request.profile not in self.renderers:
            raise ValueError(f"unknown figure profile: {request.profile}")
        request.output_dir.mkdir(parents=True, exist_ok=True)
        records = request.results.load()
        renderer = self.renderers[request.profile]
        runpy.run_path(
            str(renderer),
            run_name="_tinkerrl_figure_renderer",
            init_globals={
                "FIGURE_RECORDS": records,
                "FIGURE_OUTPUT_DIR": request.output_dir.resolve(),
            },
        )
        missing = [
            name
            for name in self.outputs[request.profile]
            if not (request.output_dir / name).is_file()
        ]
        if missing:
            raise RuntimeError(
                f"{request.profile} renderer did not create expected outputs: {', '.join(missing)}"
            )

        manifest = request.output_dir / f"{request.profile}.figure-manifest.json"
        payload = {
            "profile": request.profile,
            "results_source": request.results.source,
            "renderer": str(renderer.relative_to(REPO_ROOT)),
            "outputs": list(self.outputs[request.profile]),
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
        output_dir = args.out / profile if args.profile == "all" else args.out
        manifest = module.render(default_request(profile, output_dir))
        print(f"[{profile}] manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
