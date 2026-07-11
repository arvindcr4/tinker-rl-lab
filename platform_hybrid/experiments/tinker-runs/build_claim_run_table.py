#!/usr/bin/env python3
"""Build the paper-suite claim-to-run provenance table.

The table joins the current Tinker/W&B registry to the headline claims in the
eight paper variants.  It deliberately distinguishes exact Tinker identities,
W&B-only summaries, local-only artifacts, resource claims, and unsupported or
conflicted claims.

Outputs:
  experiments/results/claim_to_run/claim_to_run_table.tsv
  experiments/results/claim_to_run/claim_to_run_table.md
  experiments/results/claim_to_run/manifest.json
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ENTITY = "arvindcr4-pes-university"
REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "platform_hybrid" / "experiments" / "results"
REGISTRY = RESULTS / "tinker_wandb_registry"
OUTPUT = RESULTS / "claim_to_run"

COLUMNS = [
    "paper",
    "claim_id",
    "claim",
    "run_ids",
    "wandb_links",
    "model_consistency",
    "seed_count",
    "steps",
    "heldout_metric",
    "evidence_tier",
    "source_artifact",
    "audit_note",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_tsv(path: Path, comments: bool = False) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        if comments:
            lines = (line for line in handle if not line.startswith("#"))
            return list(csv.DictReader(lines, delimiter="\t"))
        return list(csv.DictReader(handle, delimiter="\t"))


def unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def wandb_ref(run: dict[str, Any]) -> str:
    return f"wandb:{run['project']}/{run['run_id']}"


def local_ref(*parts: Any) -> str:
    return "local:" + ":".join(str(part).replace(" ", "_") for part in parts)


def get_step(run: dict[str, Any]) -> int | None:
    config = run.get("config") or {}
    summary = run.get("summary") or {}
    for key in ("steps", "n_steps", "total_steps"):
        value = config.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    value = config.get("max_steps")
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    for key in ("train/global_step", "train/step", "step"):
        value = summary.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            return int(value)
    return None


def get_seed_values(run: dict[str, Any]) -> list[str]:
    config = run.get("config") or {}
    values: list[Any] = []
    if config.get("seed") is not None:
        values.append(config["seed"])
    seeds = config.get("seeds")
    if isinstance(seeds, list):
        values.extend(seeds)
    return [str(value) for value in values]


def run_by_id(wandb_runs: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    matches = [run for run in wandb_runs if run["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"expected one W&B run {run_id}, found {len(matches)}")
    return matches[0]


def make_row(**values: str) -> dict[str, str]:
    missing = [column for column in COLUMNS if not values.get(column)]
    if missing:
        raise ValueError(f"claim row missing {missing}: {values.get('claim_id')}")
    extra = sorted(set(values) - set(COLUMNS))
    if extra:
        raise ValueError(f"claim row has unexpected columns {extra}")
    return {column: str(values[column]) for column in COLUMNS}


def join_refs(items: Iterable[str]) -> str:
    values = unique(items)
    return ";".join(values) if values else "NONE"


def registry_validation() -> tuple[int, int, list[str]]:
    """Validate every current registry JSON file against the current schema."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - dependency failure is explicit
        raise RuntimeError("jsonschema is required to build the P6 resource row") from exc

    root = REPO / "platform_hybrid" / "registry"
    schema = json.loads((root / "schema.json").read_text())
    validator = Draft202012Validator(schema)
    entries = sorted((root / "entries").glob("*.json"))
    failures: list[str] = []
    for path in entries:
        record = json.loads(path.read_text())
        if next(validator.iter_errors(record), None) is not None:
            failures.append(path.name)
    return len(entries) - len(failures), len(entries), failures


def build_rows() -> tuple[list[dict[str, str]], dict[str, Any]]:
    wandb_runs = load_jsonl(REGISTRY / "wandb_runs.jsonl")
    tinker_runs = load_jsonl(REGISTRY / "tinker_runs.jsonl")
    tinker_by_id = {run["training_run_id"]: run for run in tinker_runs}
    wandb_urls = {run["url"] for run in wandb_runs}

    rows: list[dict[str, str]] = []

    # P1: use only finished, exact-ID, model-consistent GSM8K scaling anchors.
    p1_prefixes = ("scale_gsm8k_", "frontier_gsm8k_", "arch_gsm8k_", "moe_gsm8k_")
    p1_clean: list[dict[str, Any]] = []
    p1_tinker_ids: list[str] = []
    for run in wandb_runs:
        refs = run.get("referenced_tinker_ids") or []
        if not (
            run.get("project") == "tinker-rl-lab-world-class"
            and run.get("state") == "finished"
            and str(run.get("display_name", "")).startswith(p1_prefixes)
            and refs
        ):
            continue
        exact = [tinker_by_id.get(ref) for ref in refs]
        if all(
            item is not None
            and run.get("normalized_model")
            and run.get("normalized_model") == item.get("normalized_model")
            for item in exact
        ):
            p1_clean.append(run)
            p1_tinker_ids.extend(refs)
    p1_steps = sorted({step for run in p1_clean if (step := get_step(run)) is not None})
    rows.append(
        make_row(
            paper="P1",
            claim_id="P1-C1",
            claim="Across the fitted GSM8K anchors, reward has no reliable positive size trend and no identifiable universal saturation law.",
            run_ids=join_refs(
                [f"tinker:{run_id}" for run_id in p1_tinker_ids]
                + [wandb_ref(run) for run in p1_clean]
            ),
            wandb_links=join_refs(run["url"] for run in p1_clean),
            model_consistency=(
                f"CONSISTENT for selected exact anchors: {len(p1_clean)} W&B runs / "
                f"{len(set(p1_tinker_ids))} Tinker IDs"
            ),
            seed_count=f"1 per anchor (seed 42; {len(p1_clean)} selected runs)",
            steps=(
                f"{min(p1_steps)}-{max(p1_steps)} per run" if p1_steps else "UNKNOWN"
            ),
            heldout_metric="NONE in the exact-linked W&B summaries; claim uses training reward traces",
            evidence_tier="C",
            source_artifact="platform_hybrid/experiments/results/scaling_law_fits.tsv",
            audit_note="Descriptive only: single-seed, 20-30-step anchors; failed/crashed and model-conflicting links are excluded from this row.",
        )
    )

    nemotron = run_by_id(wandb_runs, "ax59u2zl")
    nemotron_tid = nemotron["referenced_tinker_ids"][0]
    nemotron_actual = tinker_by_id[nemotron_tid]["base_model"]
    rows.append(
        make_row(
            paper="P1",
            claim_id="P1-C2",
            claim="Nemotron-120B is a distinct collapse phase with zero-reward step fraction 0.55.",
            run_ids=join_refs([f"tinker:{nemotron_tid}", wandb_ref(nemotron)]),
            wandb_links=nemotron["url"],
            model_consistency=(
                "CONFLICT: W&B labels Nemotron-120B but exact Tinker ID reports "
                f"{nemotron_actual}"
            ),
            seed_count="1 (seed 42)",
            steps="20",
            heldout_metric="NONE",
            evidence_tier="X",
            source_artifact="platform_hybrid/experiments/results/scaling_law_fits.tsv",
            audit_note="Do not present as a model-specific Nemotron result until the run identity is repaired or independently re-run.",
        )
    )

    # P2: the paper's local cross-experiment diagnostic is not linked to W&B IDs.
    zvf_rows = load_tsv(RESULTS / "zvf_summary.tsv", comments=True)
    zvf_local_ids = [
        local_ref(
            "zvf_summary",
            row["experiment"],
            row["model"],
            f"G{row['group_size'] or 'na'}",
            f"s{row['seed'] or 'na'}",
            f"row{index}",
        )
        for index, row in enumerate(zvf_rows, start=1)
    ]
    zvf_steps = [int(row["n_steps"]) for row in zvf_rows if row.get("n_steps", "").isdigit()]
    zvf_cells = {
        (row["experiment"], row["model"], row["group_size"])
        for row in zvf_rows
    }
    zvf_max_seeds = max(
        int(row["n_seeds"])
        for row in zvf_rows
        if row.get("n_seeds", "").isdigit()
    )
    p2_shared = dict(
        paper="P2",
        run_ids=join_refs(zvf_local_ids),
        wandb_links="NONE",
        model_consistency="LOCAL HETEROGENEOUS POOL; no Tinker-to-W&B identity recorded",
        seed_count=(
            f"{len(zvf_rows)} local rows pooled to {len(zvf_cells)} heterogeneous "
            f"experiment/model/G cells; up to {zvf_max_seeds} seeds per source row"
        ),
        steps=f"{min(zvf_steps)}-{max(zvf_steps)} across local rows",
        source_artifact="platform_hybrid/experiments/results/zvf_summary.tsv",
    )
    rows.append(
        make_row(
            **p2_shared,
            claim_id="P2-C1",
            claim="Mean ZVF tracks catastrophic collapse (Spearman 0.56; point-biserial 0.62).",
            heldout_metric="Collapse label is derived from peak vs last-10 held-out accuracy; n=23 pooled cells",
            evidence_tier="C",
            audit_note="Cross-cell descriptive correlation; heterogeneous models/tasks and no live W&B run mapping.",
        )
    )
    rows.append(
        make_row(
            **p2_shared,
            claim_id="P2-C2",
            claim="ZVF correlates only weakly with final held-out outcome.",
            heldout_metric="Spearman rho=0.27, bootstrap 95% CI [-0.37, 0.88], n=23 pooled cells",
            evidence_tier="C",
            audit_note="The interval spans a large negative-to-positive range; state the point estimate as weak and inconclusive, not predictive.",
        )
    )

    # P3: powered-by-seed but extremely short W&B-only group-size sweep.
    p3_ids = [
        "l22x3tca", "pi494jq6", "9619su2v", "itm8rucn",
        "69r2fxq7", "w0kdbyme", "eiphi8fy", "m3pw5try",
        "t9ccvd9f", "ho2sh257", "zlbow9m5", "x6d3i1yq",
    ]
    p3_runs = [run_by_id(wandb_runs, run_id) for run_id in p3_ids]
    gains_by_g: defaultdict[int, list[float]] = defaultdict(list)
    for run in p3_runs:
        group = int(str(run["display_name"]).split("-", 1)[0][1:])
        gains_by_g[group].append(float(run["summary"]["heldout_gain"]))
    gain_text = ", ".join(
        f"G={group}: mean delta={sum(values) / len(values):+.4f}"
        for group, values in sorted(gains_by_g.items())
    )
    rows.append(
        make_row(
            paper="P3",
            claim_id="P3-C1",
            claim="Measured held-out accuracy is effectively flat from G=2 through G=16 in the saturated arithmetic regime.",
            run_ids=join_refs(wandb_ref(run) for run in p3_runs),
            wandb_links=join_refs(run["url"] for run in p3_runs),
            model_consistency="W&B-HOMOGENEOUS Qwen/Qwen3.5-4B; Tinker identity absent",
            seed_count="3 per G (seeds 0,1,2; 12 runs)",
            steps="8 per run",
            heldout_metric=f"{gain_text}; evaluation size is not logged in W&B config",
            evidence_tier="C",
            source_artifact="W&B project rlvr-openings",
            audit_note="Direct held-out metrics exist, but the horizon is eight steps and there are no exact Tinker run IDs.",
        )
    )

    p3_g32 = [
        run for run in wandb_runs
        if run.get("project") == "zvf-audit-v2" and (run.get("config") or {}).get("G") == 32
    ]
    rows.append(
        make_row(
            paper="P3",
            claim_id="P3-C2",
            claim="A measured G=4 versus G=32 token-budget/Pareto comparison supports the group-size conclusion.",
            run_ids=join_refs(wandb_ref(run) for run in p3_g32),
            wandb_links=join_refs(run["url"] for run in p3_g32),
            model_consistency="NOT APPLICABLE TO CLAIM: available G=32 records are sampling cells, not training runs",
            seed_count="5 G=32 sampling cells; no matched G=4/G=32 training replication",
            steps="0 optimizer steps (sampling-only)",
            heldout_metric="NONE",
            evidence_tier="X",
            source_artifact="platform_hybrid/paper/sections/p3_abstract.tex",
            audit_note="The current abstract correctly says no measured G=32 training cell; keep any older G=4-vs-G=32 result out of the paper.",
        )
    )

    # P4: the local artifact is internally coherent but does not match the model named in the abstract.
    p4 = json.loads((RESULTS / "drgrpo_gsm8k_cot.json").read_text())
    p4_ids = [
        local_ref("drgrpo_gsm8k_cot", run["algo"], f"seed{run['seed']}")
        for run in p4["runs"]
    ]
    p4_models = sorted({run["model"] for run in p4["runs"]})
    grpo = p4["summary"]["grpo"]
    drgrpo = p4["summary"]["dr_grpo"]
    rows.append(
        make_row(
            paper="P4",
            claim_id="P4-C1",
            claim="On Qwen3-8B, GRPO and Dr.GRPO have indistinguishable short-horizon held-out gains and neither inflates length under a 200-token cap.",
            run_ids=join_refs(p4_ids),
            wandb_links="NONE",
            model_consistency=(
                "CONFLICT: abstract says Qwen3-8B; mapped local runs say "
                + ", ".join(p4_models)
            ),
            seed_count="3 per arm (42,123,456; 6 local runs)",
            steps="30 per run",
            heldout_metric=(
                f"GRPO {grpo['heldout_pre_mean']:.4f}->{grpo['heldout_post_mean']:.4f} "
                f"(delta {grpo['delta_mean']:+.4f}); Dr.GRPO "
                f"{drgrpo['heldout_pre_mean']:.4f}->{drgrpo['heldout_post_mean']:.4f} "
                f"(delta {drgrpo['delta_mean']:+.4f}); n=200/seed"
            ),
            evidence_tier="X",
            source_artifact="platform_hybrid/experiments/results/drgrpo_gsm8k_cot.json",
            audit_note="The underlying local result is descriptive Tier C, but the exact paper claim is Tier X until the Qwen3-8B/Qwen2.5-1.5B mismatch is fixed.",
        )
    )

    # P5: direct W&B summaries exist, but none is an exact clean Tinker join.
    backend_runs = [run_by_id(wandb_runs, run_id) for run_id in ("xmot42ot", "w83mv3ok")]
    rows.append(
        make_row(
            paper="P5",
            claim_id="P5-C1",
            claim="Changing only the training backend causes a 17x final-reward swing (0.050 to 0.856).",
            run_ids=join_refs(wandb_ref(run) for run in backend_runs),
            wandb_links=join_refs(run["url"] for run in backend_runs),
            model_consistency="CONFLICT/CONFOUNDED: Qwen3-8B-Base versus Qwen3-8B; neither W&B run has an exact Tinker ID",
            seed_count="1 per backend (seed 42)",
            steps="30 per run",
            heldout_metric="NONE; metric is final/last-10 training reward",
            evidence_tier="X",
            source_artifact="platform_hybrid/paper/sections/p5_abstract.tex",
            audit_note="Valid as an undisclosed-stack/checkpoint-confound exhibit; invalid as a causal backend-only effect.",
        )
    )

    dapo_runs = [run_by_id(wandb_runs, run_id) for run_id in ("6c7p198f", "l5m9lqij")]
    rows.append(
        make_row(
            paper="P5",
            claim_id="P5-C2",
            claim="The DAPO label yields mean ZVF 0.00 on the open trainer versus 0.58 for a closed-stack asymmetric-clip surrogate.",
            run_ids=join_refs(wandb_ref(run) for run in dapo_runs),
            wandb_links=join_refs(run["url"] for run in dapo_runs),
            model_consistency="INTENTIONALLY NON-CONSISTENT EXHIBIT: Qwen2.5-0.5B open audit versus Qwen3.5-4B managed summary; no Tinker IDs",
            seed_count="2 listed open-audit seeds versus 3 managed-summary seeds",
            steps="Open audit not logged; managed summary says 15",
            heldout_metric="Open-audit table reports held-out deltas; no directly comparable held-out metric across the two stacks",
            evidence_tier="C",
            source_artifact="platform_hybrid/paper/sections/p5_abstract.tex",
            audit_note="Supports label ambiguity, not a controlled algorithm comparison.",
        )
    )

    h2h = run_by_id(wandb_runs, "l5m9lqij")
    rows.append(
        make_row(
            paper="P5",
            claim_id="P5-C3",
            claim="On one fixed stack, four algorithm labels finish within a 0.034 last-10 reward band (0.710-0.744).",
            run_ids=wandb_ref(h2h),
            wandb_links=h2h["url"],
            model_consistency="W&B summary says Qwen/Qwen3.5-4B; exact Tinker identity absent",
            seed_count="3 per arm (42,123,456; summary artifact)",
            steps="15 per arm",
            heldout_metric="NONE; training last-10 rewards only",
            evidence_tier="C",
            source_artifact="platform_hybrid/experiments/results/p5p8/p5_iter173_headline_cis.tsv",
            audit_note="Short-horizon W&B summary record; useful supporting evidence, not an inferential headline.",
        )
    )

    # P6: deterministic resource/audit claims.
    registry_pass, registry_total, registry_failures = registry_validation()
    registry_ids = [f"registry:{path.stem}" for path in sorted((REPO / "platform_hybrid/registry/entries").glob("*.json"))]
    rows.append(
        make_row(
            paper="P6",
            claim_id="P6-C1",
            claim="The released GRPO-Registry is machine-readable and schema-valid.",
            run_ids=join_refs(registry_ids),
            wandb_links="NONE",
            model_consistency="N/A (resource claim)",
            seed_count="N/A",
            steps="N/A",
            heldout_metric=f"Current schema validation: {registry_pass}/{registry_total} entries pass",
            evidence_tier="R",
            source_artifact="platform_hybrid/registry/schema.json",
            audit_note=(
                "Resource claim is only partially satisfied; failing files: "
                + ", ".join(registry_failures)
            ),
        )
    )

    exact_by_tinker: defaultdict[str, list[tuple[dict[str, Any], bool]]] = defaultdict(list)
    for run in wandb_runs:
        for tinker_id in run.get("referenced_tinker_ids") or []:
            tinker_run = tinker_by_id.get(tinker_id)
            if tinker_run is None:
                continue
            consistent = bool(
                run.get("normalized_model")
                and run.get("normalized_model") == tinker_run.get("normalized_model")
            )
            exact_by_tinker[tinker_id].append((run, consistent))
    consistent_tinker = sorted(
        tinker_id for tinker_id, pairs in exact_by_tinker.items() if any(ok for _, ok in pairs)
    )
    conflicting_tinker = sorted(set(exact_by_tinker) - set(consistent_tinker))
    exact_wandb = unique(
        run["url"] for pairs in exact_by_tinker.values() for run, _ in pairs
    )
    exact_wandb_refs = unique(
        wandb_ref(run) for pairs in exact_by_tinker.values() for run, _ in pairs
    )
    rows.append(
        make_row(
            paper="P6",
            claim_id="P6-C2-NEW-AUDIT",
            claim="The live Tinker-to-W&B identity audit separates exact model-consistent links from exact-ID/model-conflicting links.",
            run_ids=join_refs(
                [f"tinker:{tinker_id}" for tinker_id in sorted(exact_by_tinker)]
                + exact_wandb_refs
            ),
            wandb_links=join_refs(exact_wandb),
            model_consistency=(
                f"{len(consistent_tinker)}/{len(exact_by_tinker)} exact Tinker IDs have a model-consistent W&B link; "
                f"{len(conflicting_tinker)} have only model-conflicting links"
            ),
            seed_count="As logged per underlying run; identity audit does not pool seeds",
            steps="As logged per underlying run",
            heldout_metric="NONE in the model-consistent exact-linked W&B summaries",
            evidence_tier="R",
            source_artifact="platform_hybrid/experiments/results/tinker_wandb_registry/tinker_wandb_correlation.csv",
            audit_note="New review audit, not yet a paper claim; use it to repair provenance before promoting empirical results.",
        )
    )

    # P7: the 368-run audit and two interventional summary records.
    zvf_audit = [run for run in wandb_runs if run.get("project") == "zvf-audit"]
    audit_seeds = sorted({seed for run in zvf_audit for seed in get_seed_values(run)})
    audit_steps = sorted({step for run in zvf_audit if (step := get_step(run)) is not None})
    audit_states = Counter(run.get("state", "unknown") for run in zvf_audit)
    rows.append(
        make_row(
            paper="P7",
            claim_id="P7-C1",
            claim="A 368-run W&B audit reproduces the predicted U-shape of ZVF in accuracy and the monotone interior-G effect.",
            run_ids=join_refs(wandb_ref(run) for run in zvf_audit),
            wandb_links=join_refs(run["url"] for run in zvf_audit),
            model_consistency="W&B-ONLY multi-model audit; 0 exact Tinker IDs",
            seed_count=f"{len(audit_seeds)} logged seed labels ({','.join(audit_seeds)}), not a fixed-cell replication count",
            steps=f"{min(audit_steps)}-{max(audit_steps)} configured steps",
            heldout_metric="NONE; W&B summaries contain no held-out/eval metric",
            evidence_tier="C",
            source_artifact="W&B project zvf-audit",
            audit_note=(
                f"{audit_states['finished']} finished, {audit_states['failed']} failed, "
                f"{audit_states['crashed']} crashed; diagnostic/sampling evidence only."
            ),
        )
    )

    grad = run_by_id(wandb_runs, "ds83rymc")
    corr = grad["summary"].get("pearson_gradnorm_vs_p1mp")
    rows.append(
        make_row(
            paper="P7",
            claim_id="P7-C2",
            claim="Gradient magnitude tracks p(1-p) with correlation +0.71 in the toy open-backward-pass experiment.",
            run_ids=wandb_ref(grad),
            wandb_links=grad["url"],
            model_consistency="W&B says Qwen/Qwen2.5-0.5B-Instruct; exact Tinker identity absent",
            seed_count="3 (0,1,2) in one aggregate W&B summary",
            steps="30",
            heldout_metric=f"NONE; mechanism metric Pearson corr={corr}",
            evidence_tier="C",
            source_artifact="W&B run zvf-colab-experiments/ds83rymc",
            audit_note="Directional toy-scale mechanism evidence; not a held-out performance result.",
        )
    )

    adaptive = run_by_id(wandb_runs, "6c7p198f")
    rows.append(
        make_row(
            paper="P7",
            claim_id="P7-C3",
            claim="Adaptive-G matches the best fixed-recipe held-out gain (+0.575) at 186 rollouts.",
            run_ids=wandb_ref(adaptive),
            wandb_links=adaptive["url"],
            model_consistency="W&B says Qwen/Qwen2.5-0.5B-Instruct; exact Tinker identity absent",
            seed_count="1 independent run per arm (aggregate config lists [0,1], but arm-level CI ledger is n=1)",
            steps="NOT LOGGED in W&B summary",
            heldout_metric="Adaptive-G held-out delta +0.575; mean ZVF 0.23; 186 rollouts",
            evidence_tier="C",
            source_artifact="platform_hybrid/experiments/results/p5p8/p7_iter123_headline_cis.tsv",
            audit_note="Single-seed arm; the local ledger explicitly marks the confidence interval as INSUFFICIENT_N.",
        )
    )

    # P8: one local XGBoost record and one W&B-only SFT record.
    rows.append(
        make_row(
            paper="P8",
            claim_id="P8-C1",
            claim="XGBoost reaches AUC 0.7955 on the 10,000-row synthetic held-out split.",
            run_ids=local_ref("qp8_fraud", "xgboost", "seed42"),
            wandb_links="NONE",
            model_consistency="CONSISTENT local XGBoost configuration; no remote run identity",
            seed_count="1 (random_state 42)",
            steps="200 estimators (not optimizer steps)",
            heldout_metric="AUC 0.7955 on n=10,000",
            evidence_tier="C",
            source_artifact="platform_hybrid/experiments/results/quick_20260704/qp8_fraud.tsv",
            audit_note="Single synthetic data configuration; the repository also contains a separate xgboost_results.json with AUC 0.79424, so pin the exact artifact/version.",
        )
    )

    fraud = run_by_id(wandb_runs, "ek1b2cxn")
    fraud_auc = fraud["summary"]["eval_final/auc"]
    fraud_acc = fraud["summary"]["eval_final/accuracy"]
    rows.append(
        make_row(
            paper="P8",
            claim_id="P8-C2",
            claim="Qwen3.5-4B SFT row serialization is near chance for ranking fraud risk.",
            run_ids=wandb_ref(fraud),
            wandb_links=fraud["url"],
            model_consistency="W&B says Qwen/Qwen3.5-4B; Tinker identity absent",
            seed_count="1 (seed 0)",
            steps="63 SFT minibatches",
            heldout_metric=f"AUC {fraud_auc:.6f}; accuracy {fraud_acc:.3f}; n=500 positive-enriched held-out rows",
            evidence_tier="C",
            source_artifact="platform_hybrid/experiments/results/quick_20260704/qp8-fraud-sft.tsv",
            audit_note="The LLM and XGBoost use different held-out class mixtures; do not interpret their AUC difference as a paired margin.",
        )
    )

    rows.append(
        make_row(
            paper="P8",
            claim_id="P8-C3",
            claim="The proposed sensor/scribe/agentic-triage hybrid produces measured operational benefit over XGBoost alone.",
            run_ids="UNLINKED",
            wandb_links="NONE",
            model_consistency="N/A; architecture is proposed, not experimentally evaluated",
            seed_count="0",
            steps="0",
            heldout_metric="NONE",
            evidence_tier="X",
            source_artifact="platform_hybrid/paper/sections/p8_abstract.tex",
            audit_note="Keep as future-work/design guidance unless a controlled operational evaluation is added.",
        )
    )

    metadata = {
        "wandb_url_set": wandb_urls,
        "wandb_run_count": len(wandb_runs),
        "tinker_run_count": len(tinker_runs),
        "p1_selected_wandb_runs": len(p1_clean),
        "p1_selected_tinker_ids": len(set(p1_tinker_ids)),
        "exact_tinker_ids": len(exact_by_tinker),
        "exact_model_consistent_tinker_ids": len(consistent_tinker),
        "exact_model_conflicting_tinker_ids": len(conflicting_tinker),
        "registry_validation_pass": registry_pass,
        "registry_validation_total": registry_total,
        "registry_validation_failures": registry_failures,
    }
    return rows, metadata


def validate_rows(rows: list[dict[str, str]], metadata: dict[str, Any]) -> None:
    claim_ids = [row["claim_id"] for row in rows]
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("duplicate claim_id in output")
    if {row["paper"] for row in rows} != {f"P{i}" for i in range(1, 9)}:
        raise ValueError("table does not cover all eight papers")
    for row in rows:
        for link in row["wandb_links"].split(";"):
            if link in {"", "NONE"}:
                continue
            if link not in metadata["wandb_url_set"]:
                raise ValueError(f"unknown W&B URL in {row['claim_id']}: {link}")
        source = row["source_artifact"]
        if source.startswith("platform_hybrid/") and not (REPO / source).exists():
            raise ValueError(f"missing source artifact in {row['claim_id']}: {source}")


def compact_list(value: str, limit: int = 3) -> str:
    if value in {"NONE", "UNLINKED"}:
        return value
    items = value.split(";")
    shown = items[:limit]
    suffix = f"; +{len(items) - limit} more in TSV" if len(items) > limit else ""
    return "; ".join(shown) + suffix


def compact_links(value: str, limit: int = 3) -> str:
    if value == "NONE":
        return value
    links = value.split(";")
    rendered = []
    for link in links[:limit]:
        bits = link.rstrip("/").split("/")
        label = f"{bits[-3]}/{bits[-1]}" if len(bits) >= 3 else bits[-1]
        rendered.append(f"[{label}]({link})")
    if len(links) > limit:
        rendered.append(f"+{len(links) - limit} more in TSV")
    return "; ".join(rendered)


def md_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def write_outputs(rows: list[dict[str, str]], metadata: dict[str, Any]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tsv_path = OUTPUT / "claim_to_run_table.tsv"
    with tsv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    tier_counts = Counter(row["evidence_tier"] for row in rows)
    md = [
        "# Claim-to-run table",
        "",
        "Generated from the checked-in Tinker/W&B registry and current local evidence artifacts. "
        "The Markdown view abbreviates long run lists; the sibling TSV contains every run ID and every W&B URL.",
        "",
        "Evidence tiers use the repository's statistical-rigor thresholds: **A** = at least 5 seeds and at least 100 steps; "
        "**B** = 3-4 seeds and 50-99 steps; **C** = completed/descriptive evidence that does not meet A/B. "
        "This audit adds **R** for deterministic resource/schema claims and **X** for unlinked, contradicted, or provenance-conflicted claims.",
        "",
        "| Paper | Claim | Run IDs | W&B links | Model consistency | Seeds | Steps | Held-out metric | Tier |",
        "|---|---|---|---|---|---:|---:|---|:---:|",
    ]
    for row in rows:
        values = [
            row["paper"],
            f"{row['claim_id']}: {row['claim']}",
            compact_list(row["run_ids"]),
            compact_links(row["wandb_links"]),
            row["model_consistency"],
            row["seed_count"],
            row["steps"],
            row["heldout_metric"],
            row["evidence_tier"],
        ]
        md.append("| " + " | ".join(md_escape(value) for value in values) + " |")
    md.extend(
        [
            "",
            "## Audit summary",
            "",
            f"- {len(rows)} headline/resource claims across all eight papers.",
            "- Tier counts: " + ", ".join(f"{tier}={tier_counts[tier]}" for tier in sorted(tier_counts)) + ".",
            f"- Exact live provenance: {metadata['exact_model_consistent_tinker_ids']}/{metadata['exact_tinker_ids']} "
            "referenced Tinker IDs have at least one model-consistent W&B link; "
            f"{metadata['exact_model_conflicting_tinker_ids']} have only conflicting model labels.",
            f"- Current P6 registry validation: {metadata['registry_validation_pass']}/{metadata['registry_validation_total']} pass.",
            "- No empirical headline row reaches Tier A or B under the statistical-rigor appendix thresholds.",
            "",
            "## Review stop rules",
            "",
            "- Do not call P5-C1 a backend-only causal effect; the base checkpoint changes.",
            "- Do not call P1-C2 a Nemotron result until the exact-ID/model conflict is repaired.",
            "- Fix the P4 abstract model name or supply the missing Qwen3-8B runs.",
            "- Keep P3 G=32 and P8 hybrid-benefit statements as future work.",
        ]
    )
    (OUTPUT / "claim_to_run_table.md").write_text("\n".join(md) + "\n")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_count": len(rows),
        "paper_counts": dict(sorted(Counter(row["paper"] for row in rows).items())),
        "tier_counts": dict(sorted(tier_counts.items())),
        "validation": "PASS",
        **{key: value for key, value in metadata.items() if key != "wandb_url_set"},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    rows, metadata = build_rows()
    validate_rows(rows, metadata)
    write_outputs(rows, metadata)
    print(f"PASS: wrote {len(rows)} claims to {OUTPUT}")


if __name__ == "__main__":
    main()
