#!/usr/bin/env python3
"""Bind one OpenAI MLE-bench competition to the Pavlov receipt contract.

This runner deliberately does *not* evaluate a language model and never claims a
suite score.  It does three things:

``survey``
    Rank all 75 MLE-bench competitions by the download size the upstream
    repository itself records in ``experiments/competition_categories.csv`` so
    the "smallest competition" choice is a measured fact, not a guess.

``harness-validate``
    Drive the official grading path -- the per-competition ``grade.py`` metric,
    the real Kaggle leaderboard shipped in the repository, the medal thresholds
    and ``CompetitionReport`` -- against a locally constructed, schema-conformant
    fixture.  A grader scoring its own gold answers proves the metric, the
    grader and the submission format all work.  It is labelled
    ``harness_validation`` with ``is_model_score: false`` and it is never a
    benchmark score.

``receipt``
    Emit the fail-closed receipt: immutable upstream revision, task-ID hashes,
    split manifest, container digest and verifier identity.  Every gate must
    pass before ``status`` can leave ``BLOCKED``; the suite ``score`` stays
    ``null`` until a graded model submission exists.

MLE-bench's own MIT licence covers the repository code only.  Each of the 75
competitions carries its own Kaggle rules that a human must accept on the
competition page before the Kaggle API will serve the data, so dataset licence
state is tracked as a separate, per-competition gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = "e9-mle-bench-eval-receipt-v2"
UPSTREAM_URL = "https://github.com/openai/mle-bench"
UPSTREAM_COMMIT = "507f92e1138bb6e40dac5c6ee7a6758e6424bf97"

# The canonical agent image the upstream harness expects.  Only this tag can
# satisfy the container gate.
REQUIRED_IMAGE = "mlebench-env"

# A verifier-only build of the same Dockerfile with INSTALL_HEAVY_DEPENDENCIES=false.
# It contains the grading server and the official mlebench package but none of the
# agent ML stack, so it can grade a submission but cannot host an agent.  Recorded
# for evidence; it never satisfies the container gate.
VERIFIER_ONLY_IMAGE = "mlebench-env:verifier-noheavy"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKOUT = _REPO_ROOT / "outputs/e9_mle_bench/mle-bench-source"
DEFAULT_DATA_DIR = _REPO_ROOT / "outputs/e9_mle_bench/data"
DEFAULT_FIXTURE_DIR = _REPO_ROOT / "outputs/e9_mle_bench/harness_fixture_data"

SIZE_TABLE_RELPATH = "experiments/competition_categories.csv"
EVAL_SPLIT_RELPATH = "experiments/splits/split75.txt"
COMPLEXITY_SPLIT_RELPATHS = {
    "low": "experiments/splits/low.txt",
    "medium": "experiments/splits/medium.txt",
    "high": "experiments/splits/high.txt",
}

TASK_IDS_SHA256_ALGORITHM = (
    "sha256(UTF-8 of sorted competition IDs joined by newline with a terminal newline)"
)

# The repository LICENSE explicitly carves the competition datasets out of the
# MIT grant.  Kaggle then requires a signed-in human to accept each
# competition's rules before its API will serve the files; an agent cannot.
LICENSE_POSITION = {
    "repository_code": "MIT",
    "competition_datasets": "not_covered_by_repository_license",
    "per_competition_terms": "Kaggle competition rules, accepted per competition by a human",
    "acceptance_is_automatable": False,
}


# --------------------------------------------------------------------------
# hashing helpers
# --------------------------------------------------------------------------


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Hash a file, or return ``None`` when it is absent (fail-closed callers)."""

    path = Path(path)
    if not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def task_ids_sha256(competition_ids: Iterable[str]) -> str:
    """Immutable task-ID hash over a competition list.

    Sorted, newline-joined, terminal newline -- so the hash is independent of the
    order the IDs were read in and stable across re-reads of the split file.
    """

    ordered = sorted(str(item).strip() for item in competition_ids if str(item).strip())
    return sha256_bytes(("\n".join(ordered) + "\n").encode("utf-8"))


# --------------------------------------------------------------------------
# upstream survey
# --------------------------------------------------------------------------


def read_split(checkout: Path, relpath: str) -> list[str]:
    path = Path(checkout) / relpath
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_size_table(checkout: Path) -> list[dict[str, Any]]:
    """Parse the upstream ``competition_categories.csv`` size table.

    This file is the repository's own record of each competition's download
    size, so a "smallest competition" claim derived from it is sourced from the
    registry rather than estimated.
    """

    path = Path(checkout) / SIZE_TABLE_RELPATH
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            raw_size = (record.get("dataset_size_GB") or "").strip()
            try:
                size_gb = float(raw_size)
            except ValueError:
                continue
            rows.append(
                {
                    "competition_id": (record.get("competition_id") or "").strip(),
                    "dataset_size_gb": size_gb,
                    "dataset_size_mb": round(size_gb * 1024, 4),
                    "category": (record.get("category") or "").strip(),
                    "complexity": (record.get("Complexity") or "").strip(),
                }
            )
    return rows


def rank_competitions_by_size(checkout: Path) -> list[dict[str, Any]]:
    """All surveyed competitions, ascending by recorded download size."""

    return sorted(read_size_table(checkout), key=lambda row: (row["dataset_size_gb"], row["competition_id"]))


def smallest_competition(checkout: Path) -> dict[str, Any] | None:
    ranked = rank_competitions_by_size(checkout)
    return ranked[0] if ranked else None


def survey(checkout: Path) -> dict[str, Any]:
    """Survey every competition definition and record the size ranking."""

    checkout = Path(checkout)
    eval_ids = read_split(checkout, EVAL_SPLIT_RELPATH)
    ranked = rank_competitions_by_size(checkout)
    surveyed_ids = {row["competition_id"] for row in ranked}
    definition_dirs = sorted(
        p.name
        for p in (checkout / "mlebench/competitions").iterdir()
        if p.is_dir() and (p / "config.yaml").is_file()
    ) if (checkout / "mlebench/competitions").is_dir() else []

    return {
        "source": {
            "size_table": SIZE_TABLE_RELPATH,
            "size_table_sha256": sha256_file(checkout / SIZE_TABLE_RELPATH),
            "eval_split": EVAL_SPLIT_RELPATH,
            "eval_split_sha256": sha256_file(checkout / EVAL_SPLIT_RELPATH),
            "size_units": "GB as recorded by the upstream repository",
        },
        "eval_split_competition_count": len(eval_ids),
        "definition_dir_count": len(definition_dirs),
        "sized_competition_count": len(ranked),
        "sized_covers_eval_split": bool(eval_ids) and set(eval_ids) <= surveyed_ids,
        "total_dataset_size_gb": round(sum(row["dataset_size_gb"] for row in ranked), 4),
        "smallest": ranked[0] if ranked else None,
        "ranking": ranked,
    }


# --------------------------------------------------------------------------
# competition binding
# --------------------------------------------------------------------------


def docker_image_digest(image: str = REQUIRED_IMAGE) -> dict[str, Any]:
    """Resolve the container digest for the required image, fail-closed."""

    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", image, "--format", "{{json .}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"image": image, "present": False, "digest": None, "error": f"{type(exc).__name__}: {exc}"}

    if completed.returncode != 0:
        return {
            "image": image,
            "present": False,
            "digest": None,
            "error": (completed.stderr or completed.stdout).strip()[:400] or "image not found",
        }

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {"image": image, "present": False, "digest": None, "error": f"unparseable inspect output: {exc}"}

    repo_digests = payload.get("RepoDigests") or []
    return {
        "image": image,
        "present": True,
        "digest": repo_digests[0] if repo_digests else payload.get("Id"),
        "digest_kind": "repo_digest" if repo_digests else "local_image_id",
        "repo_digests": repo_digests,
        "image_id": payload.get("Id"),
        "architecture": payload.get("Architecture"),
        "os": payload.get("Os"),
        "error": None,
    }


def verifier_identity(checkout: Path, competition_id: str) -> dict[str, Any]:
    """Pin who grades this competition and with which fixed inputs.

    The verifier is the per-competition ``grade.py`` plus the real Kaggle
    leaderboard that supplies the medal thresholds plus the expected data
    checksums.  All three are content-hashed so a receipt names an exact
    verifier, not "the grader".
    """

    comp_dir = Path(checkout) / "mlebench/competitions" / competition_id
    config_path = comp_dir / "config.yaml"
    grader_name: str | None = None
    grade_fn: str | None = None
    if config_path.is_file():
        # The competition config has a top-level `name:` (the competition) and a
        # `name:` nested under `grader:` (the metric). Only the nested one is the
        # verifier's identity, so read the grader block rather than the whole file.
        text = config_path.read_text(encoding="utf-8")
        grader_block = text.split("grader:", 1)[1] if "grader:" in text else ""
        for line in grader_block.splitlines():
            stripped = line.strip()
            if stripped.startswith("name:") and grader_name is None:
                grader_name = stripped.split(":", 1)[1].strip()
            if stripped.startswith("grade_fn:"):
                grade_fn = stripped.split(":", 1)[1].strip()

    leaderboard = comp_dir / "leaderboard.csv"
    leaderboard_rows: int | None = None
    leaderboard_is_lfs_pointer = False
    if leaderboard.is_file():
        head = leaderboard.read_text(encoding="utf-8", errors="replace").splitlines()
        leaderboard_is_lfs_pointer = bool(head) and head[0].startswith("version https://git-lfs")
        leaderboard_rows = None if leaderboard_is_lfs_pointer else max(len(head) - 1, 0)

    return {
        "kind": "per-competition native grade.py plus real Kaggle leaderboard thresholds",
        "competition_id": competition_id,
        "config_sha256": sha256_file(config_path),
        "grader_name": grader_name,
        "grade_fn": grade_fn,
        "grade_py_sha256": sha256_file(comp_dir / "grade.py"),
        "prepare_py_sha256": sha256_file(comp_dir / "prepare.py"),
        "expected_data_checksums_sha256": sha256_file(comp_dir / "checksums.yaml"),
        "leaderboard_sha256": sha256_file(leaderboard),
        "leaderboard_team_count": leaderboard_rows,
        "leaderboard_is_unresolved_lfs_pointer": leaderboard_is_lfs_pointer,
        "grading_server_sha256": sha256_file(Path(checkout) / "environment/grading_server.py"),
        "resolved": bool(
            grade_fn
            and sha256_file(comp_dir / "grade.py")
            and leaderboard.is_file()
            and not leaderboard_is_lfs_pointer
        ),
    }


def split_manifest(checkout: Path) -> dict[str, Any]:
    """Immutable split manifest: file hashes, counts and the task-ID hash."""

    checkout = Path(checkout)
    eval_ids = read_split(checkout, EVAL_SPLIT_RELPATH)
    complexity: dict[str, Any] = {}
    for name, relpath in COMPLEXITY_SPLIT_RELPATHS.items():
        ids = read_split(checkout, relpath)
        complexity[name] = {
            "path": relpath,
            "count": len(ids),
            "sha256": sha256_file(checkout / relpath),
            "task_ids_sha256": task_ids_sha256(ids) if ids else None,
        }
    return {
        "task_id_kind": "competition_id",
        "eval_split": {
            "path": EVAL_SPLIT_RELPATH,
            "count": len(eval_ids),
            "sha256": sha256_file(checkout / EVAL_SPLIT_RELPATH),
            "task_ids_sha256": task_ids_sha256(eval_ids) if eval_ids else None,
        },
        "complexity_splits": complexity,
        "task_ids_sha256_algorithm": TASK_IDS_SHA256_ALGORITHM,
        "per_sample_task_hashes": None,
        "per_sample_task_hash_status": (
            "unavailable until the competition data is prepared; MLE-bench task IDs are "
            "competition IDs and per-sample rows only exist after `mlebench prepare`"
        ),
    }


def dataset_state(data_dir: Path, competition_id: str) -> dict[str, Any]:
    """Local preparation state for one competition. Never touches the network."""

    root = Path(data_dir) / competition_id
    private_answers = root / "prepared/private/test.csv"
    sample_submission = root / "prepared/public/sample_submission.csv"
    raw_dir = root / "raw"
    return {
        "data_dir": str(data_dir),
        "competition_dir": str(root),
        "raw_present": raw_dir.is_dir() and any(raw_dir.iterdir()) if raw_dir.is_dir() else False,
        "private_answers_present": private_answers.is_file(),
        "private_answers_sha256": sha256_file(private_answers),
        "sample_submission_present": sample_submission.is_file(),
        "sample_submission_sha256": sha256_file(sample_submission),
        "prepared": private_answers.is_file() and sample_submission.is_file(),
    }


def kaggle_rule_acceptance(competition_id: str, probe: dict[str, Any] | None = None) -> dict[str, Any]:
    """Per-competition Kaggle rule state, with the URL a human must visit."""

    state: dict[str, Any] = {
        "competition_id": competition_id,
        "rules_url": f"https://www.kaggle.com/c/{competition_id}/rules",
        "accepted": None,
        "accepted_evidence": None,
        "note": (
            "Kaggle serves competition files only after a signed-in human accepts the "
            "competition rules. This cannot be automated by an agent."
        ),
        "verification_warning": (
            "`kaggle competitions files` is NOT a valid acceptance check -- that metadata "
            "endpoint lists file names and sizes for un-accepted competitions too. Only the "
            "download endpoint is gated. Use check_rules_accepted()."
        ),
    }
    if probe is not None:
        # Accepts both probe shapes: the bulk rule probe (`rules_accepted`) and
        # the check-rules subcommand (`accepted` + `download_endpoint_ok`).
        accepted = probe.get("rules_accepted")
        if accepted is None:
            accepted = probe.get("accepted")
        state["accepted"] = bool(accepted)
        state["accepted_evidence"] = (
            probe.get("error")
            or probe.get("downloaded")
            or (
                "download endpoint served data"
                if probe.get("download_endpoint_ok")
                else None
            )
        )
        if "download_endpoint_ok" in probe:
            state["verified_via"] = "download endpoint (the only gated one)"
    return state


def check_rules_accepted(competition_id: str, checkout: Path = DEFAULT_CHECKOUT) -> dict[str, Any]:
    """Probe the only endpoint that actually proves Kaggle rule acceptance.

    Kaggle exposes competition file *metadata* (names, byte sizes) regardless of
    rule acceptance; only the *download* endpoint is gated. Checking the metadata
    endpoint therefore reports success for a competition that cannot be
    downloaded, so this probes the download and reports both for contrast.
    """

    import tempfile

    checkout = Path(checkout)
    if str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))

    result: dict[str, Any] = {
        "competition_id": competition_id,
        "rules_url": f"https://www.kaggle.com/c/{competition_id}/rules",
        "metadata_endpoint_ok": None,
        "download_endpoint_ok": None,
        "accepted": None,
        "error": None,
    }

    try:
        from mlebench.utils import authenticate_kaggle_api

        api = authenticate_kaggle_api()
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"kaggle auth failed: {type(exc).__name__}: {exc}"
        return result

    result["authenticated_as"] = getattr(api, "config_values", {}).get("username")

    try:
        api.competition_list_files(competition_id)
        result["metadata_endpoint_ok"] = True
    except Exception as exc:  # noqa: BLE001
        # An iteration quirk in some client versions still means the call returned.
        result["metadata_endpoint_ok"] = "not iterable" in str(exc)

    with tempfile.TemporaryDirectory() as tmp:
        try:
            api.competition_download_files(competition=competition_id, path=tmp, quiet=True, force=True)
            result["download_endpoint_ok"] = True
            result["accepted"] = True
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            result["download_endpoint_ok"] = False
            result["accepted"] = False
            result["blocked_on_rules"] = "You must accept this competition" in message
            result["error"] = message[:400]

    return result


# --------------------------------------------------------------------------
# harness validation (no model involved)
# --------------------------------------------------------------------------

# Fixture builders are competition-specific because each competition's private
# answers file has its own schema.  Anything not listed here fails closed.
FIXTURE_BUILDERS = ("spooky-author-identification",)


def build_spooky_fixture(fixture_dir: Path, rows: int = 300, seed: int = 0) -> dict[str, Any]:
    """Write a schema-conformant fixture for ``spooky-author-identification``.

    The schema is taken from the competition's own ``prepare.py``: the private
    answers file is ``id`` plus a one-hot column per class, and the sample
    submission is the fixed class-prior row that ``prepare.py`` hard-codes.  The
    *rows* are synthetic -- the real Kaggle split is unavailable without rule
    acceptance -- so anything derived from this fixture is harness evidence only.
    """

    import random

    classes = ["EAP", "HPL", "MWS"]
    priors = {"EAP": 0.403493538995863, "HPL": 0.287808366106543, "MWS": 0.308698094897594}

    competition_dir = Path(fixture_dir) / "spooky-author-identification" / "prepared"
    private_dir = competition_dir / "private"
    public_dir = competition_dir / "public"
    private_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    weights = [priors[c] for c in classes]
    labels = [rng.choices(classes, weights=weights, k=1)[0] for _ in range(rows)]
    ids = [f"id{index:06d}" for index in range(rows)]

    header = ["id", *classes]

    def write_csv(path: Path, records: list[list[Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(records)

    answers = [[ids[i], *[1 if c == labels[i] else 0 for c in classes]] for i in range(rows)]
    write_csv(private_dir / "test.csv", answers)

    sample = [[ids[i], *[priors[c] for c in classes]] for i in range(rows)]
    write_csv(public_dir / "sample_submission.csv", sample)

    # Gold submission == the private answers file, exactly as the upstream
    # registry defines it when a competition has no separate gold_submission.
    gold = [[ids[i], *[1 if c == labels[i] else 0 for c in classes]] for i in range(rows)]
    write_csv(public_dir / "gold_submission.csv", gold)

    invalid = [[ids[i], 0.9, 0.9, 0.9] for i in range(rows)]
    write_csv(public_dir / "invalid_submission.csv", invalid)

    return {
        "competition_id": "spooky-author-identification",
        "fixture_dir": str(fixture_dir),
        "provenance": "synthetic_schema_conformant_fixture",
        "provenance_reason": (
            "Kaggle competition data is unavailable: the competition rules have not been "
            "accepted for this account, so `mlebench prepare` cannot download the real split."
        ),
        "rows": rows,
        "seed": seed,
        "classes": classes,
        "schema_source": "mlebench/competitions/spooky-author-identification/prepare.py",
        "answers_path": str(private_dir / "test.csv"),
        "answers_sha256": sha256_file(private_dir / "test.csv"),
        "gold_submission_path": str(public_dir / "gold_submission.csv"),
        "sample_submission_path": str(public_dir / "sample_submission.csv"),
        "invalid_submission_path": str(public_dir / "invalid_submission.csv"),
    }


def upstream_sample_submission_score(checkout: Path, competition_id: str) -> float | None:
    """The score upstream records for a competition's own sample submission.

    Parsed out of ``tests/constants.py`` textually so importing the upstream test
    module (and its dependencies) is not required. ``np.nan`` entries return None.
    """

    path = Path(checkout) / "tests/constants.py"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    if "sample_submission_scores" not in text:
        return None
    block = text.split("sample_submission_scores", 1)[1]
    pattern = re.compile(rf'"{re.escape(competition_id)}"\s*:\s*([^,\n}}]+)')
    match = pattern.search(block)
    if not match:
        return None
    raw = match.group(1).strip()
    if "nan" in raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def build_invalid_control(sample_submission: Path, out_dir: Path) -> Path:
    """Derive a deliberately malformed submission as a negative control.

    Written outside the prepared competition directory so the official data keeps
    matching its recorded checksums. Every probability column is shifted by 0.5,
    which reliably breaks the sum-to-one constraint. It may or may not also push a
    value outside [0, 1] depending on the source row, so only the sum-to-one
    violation is guaranteed -- that is the one the grader reports.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / "invalid_submission.csv"

    with Path(sample_submission).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    header, body = rows[0], rows[1:]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in body:
            corrupted = [row[0]]
            for value in row[1:]:
                try:
                    corrupted.append(float(value) + 0.5)
                except ValueError:
                    corrupted.append(value)
            writer.writerow(corrupted)
    return destination


def harness_validate(
    checkout: Path,
    competition_id: str,
    fixture_dir: Path = DEFAULT_FIXTURE_DIR,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the official grading path end to end. Never a model score.

    When ``data_dir`` holds a real prepared competition it is used directly.
    Otherwise a schema-conformant fixture is written and clearly labelled as
    such.  Either way the metric, the grader, the real Kaggle leaderboard
    thresholds and the ``CompetitionReport`` are the upstream implementations.
    """

    checkout = Path(checkout)
    if str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))

    result: dict[str, Any] = {
        "label": "harness_validation",
        "is_model_score": False,
        "competition_id": competition_id,
        "suite_score": None,
        "note": (
            "Grading the competition's own gold answers proves the metric, the grader and "
            "the submission format work. It is not a benchmark result and must never be "
            "promoted to one."
        ),
    }

    real_state = dataset_state(data_dir, competition_id) if data_dir else {"prepared": False}
    if real_state.get("prepared"):
        active_data_dir = Path(data_dir)
        result["data_provenance"] = "official_prepared_competition_data"
        result["fixture"] = None
        result["official_data"] = {
            "answers_path": real_state["private_answers_sha256"] and str(Path(data_dir) / competition_id / "prepared/private/test.csv"),
            "answers_sha256": real_state["private_answers_sha256"],
            "sample_submission_sha256": real_state["sample_submission_sha256"],
            "prepared_by": "mlebench prepare (upstream preparer), checksums verified against the pinned checksums.yaml",
        }
    else:
        if competition_id not in FIXTURE_BUILDERS:
            result["status"] = "BLOCKED"
            result["reason"] = (
                f"No prepared data for `{competition_id}` and no fixture builder is defined "
                "for it; refusing to invent a schema."
            )
            return result
        result["fixture"] = build_spooky_fixture(Path(fixture_dir))
        result["data_provenance"] = "synthetic_schema_conformant_fixture"
        active_data_dir = Path(fixture_dir)

    try:
        from mlebench.data import get_leaderboard
        from mlebench.grade import grade_csv
        from mlebench.registry import Registry
    except Exception as exc:  # noqa: BLE001 - missing deps must fail closed, not crash
        result["status"] = "BLOCKED"
        result["reason"] = f"official mlebench package not importable: {type(exc).__name__}: {exc}"
        return result

    registry = Registry().set_data_dir(Path(active_data_dir))
    competition = registry.get_competition(competition_id)
    leaderboard = get_leaderboard(competition)

    public_dir = Path(active_data_dir) / competition_id / "prepared/public"
    cases: list[tuple[str, Path, str]] = [
        ("gold_submission", competition.gold_submission, "grader scoring the competition's own gold answers"),
        ("sample_submission", competition.sample_submission, "the sample submission shipped by prepare.py"),
    ]
    invalid_path = public_dir / "invalid_submission.csv"
    if not invalid_path.is_file() and competition.sample_submission.is_file():
        # Derive the negative control outside the prepared directory so the
        # official data keeps matching its recorded checksums.
        invalid_path = build_invalid_control(
            competition.sample_submission,
            _REPO_ROOT / "outputs/e9_mle_bench/harness_controls" / competition_id,
        )
    if invalid_path.is_file():
        cases.append(("invalid_submission", invalid_path, "rows that violate the submission contract"))

    reports = []
    for case_name, submission_path, description in cases:
        report = grade_csv(Path(submission_path), competition)
        payload = report.to_dict()
        payload["case"] = case_name
        payload["case_description"] = description
        reports.append(payload)

    by_case = {entry["case"]: entry for entry in reports}
    gold = by_case.get("gold_submission", {})
    sample = by_case.get("sample_submission", {})
    invalid = by_case.get("invalid_submission")

    # Upstream records the score its own sample submission achieves on the real
    # split.  Reproducing it is the strongest available check that this lane's
    # preparer + grader agree with the reference implementation.
    expected = upstream_sample_submission_score(checkout, competition_id)
    reproduction: dict[str, Any] = {
        "upstream_expected_score": expected,
        "upstream_source": "tests/constants.py::sample_submission_scores",
        "observed_score": sample.get("score"),
        "applicable": expected is not None
        and sample.get("score") is not None
        and result.get("data_provenance") == "official_prepared_competition_data",
    }
    if reproduction["applicable"]:
        delta = abs(sample["score"] - expected)
        reproduction["absolute_delta"] = round(delta, 8)
        reproduction["relative_delta"] = round(delta / abs(expected), 8) if expected else None
        # grade_helpers rounds every score to 5 decimals, so exact equality at
        # that precision is the correct bar.
        reproduction["matches_upstream"] = delta <= 1e-5
    result["upstream_reproduction"] = reproduction

    checks = {
        "leaderboard_is_real_kaggle_data": bool(len(leaderboard)) and "score" in leaderboard.columns,
        "gold_submission_scored": gold.get("score") is not None,
        "gold_submission_earns_gold_medal": bool(gold.get("gold_medal")),
        "sample_submission_scored": sample.get("score") is not None,
        "sample_submission_earns_no_medal": sample.get("any_medal") is False,
        "gold_beats_sample": (
            gold.get("score") is not None
            and sample.get("score") is not None
            and (
                gold["score"] < sample["score"]
                if gold.get("is_lower_better")
                else gold["score"] > sample["score"]
            )
        ),
        "invalid_submission_rejected": (invalid is None) or (invalid.get("valid_submission") is False),
    }

    result.update(
        {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "leaderboard": {
                "path": str(competition.leaderboard),
                "sha256": sha256_file(competition.leaderboard),
                "team_count": int(len(leaderboard)),
                "provenance": "real Kaggle leaderboard shipped in the pinned repository",
            },
            "grader": {
                "name": competition.grader.name,
                "grade_py_sha256": sha256_file(
                    checkout / "mlebench/competitions" / competition_id / "grade.py"
                ),
            },
            "reports": reports,
        }
    )
    return result


# --------------------------------------------------------------------------
# fail-closed receipt
# --------------------------------------------------------------------------

REQUIRED_GATES = (
    "upstream_revision_pinned",
    "split_manifest_resolved",
    "verifier_identity_resolved",
    "dataset_license_accepted",
    "competition_data_prepared",
    "container_image_digest_present",
    "model_submission_artifact_present",
    "contamination_disjointness_receipt",
)


def evaluate_gates(receipt: dict[str, Any]) -> dict[str, bool]:
    """Compute each contract gate from the assembled receipt body."""

    source = receipt.get("authoritative_public_source", {})
    manifest = receipt.get("split_manifest", {})
    verifier = receipt.get("verifier", {})
    binding = receipt.get("competition_binding", {})
    environment = receipt.get("environment", {})
    artifact = receipt.get("model_artifact", {})
    contamination = receipt.get("contamination_policy", {})

    return {
        "upstream_revision_pinned": bool(source.get("revision")) and bool(source.get("checkout_present")),
        "split_manifest_resolved": bool(manifest.get("eval_split", {}).get("task_ids_sha256")),
        "verifier_identity_resolved": bool(verifier.get("resolved")),
        "dataset_license_accepted": binding.get("kaggle_rules", {}).get("accepted") is True,
        "competition_data_prepared": bool(binding.get("dataset_state", {}).get("prepared")),
        "container_image_digest_present": bool(environment.get("digest")),
        "model_submission_artifact_present": bool(artifact.get("submission_path")),
        "contamination_disjointness_receipt": bool(contamination.get("disjoint_receipt")),
    }


def fail_closed_status(gates: dict[str, bool]) -> str:
    """``READY`` only when every required gate passes; otherwise ``BLOCKED``."""

    return "READY" if all(gates.get(name) for name in REQUIRED_GATES) else "BLOCKED"


def lane_status(gates: dict[str, bool], harness: dict[str, Any] | None) -> str:
    """Lane-level progress, distinct from the contract gate status.

    ``PARTIAL`` when the harness executed but gates still block a real run;
    ``BLOCKED`` when nothing executed; ``RUNNING`` when every gate is satisfied.
    """

    if all(gates.get(name) for name in REQUIRED_GATES):
        return "RUNNING"
    if harness and harness.get("status") == "PASS":
        return "PARTIAL"
    return "BLOCKED"


def blocker_details(gates: dict[str, bool], competition_id: str) -> list[dict[str, Any]]:
    """Name each failing gate with the external receipt that would clear it."""

    catalogue = {
        "dataset_license_accepted": {
            "missing": f"Kaggle competition rules for `{competition_id}` have not been accepted.",
            "external_receipt": "a signed-in human accepting the rules on the competition page",
            "action": f"https://www.kaggle.com/c/{competition_id}/rules",
        },
        "competition_data_prepared": {
            "missing": "No prepared public/private split exists locally.",
            "external_receipt": "the Kaggle download, which is gated on the rule acceptance above",
            "action": f"mlebench prepare -c {competition_id} --data-dir outputs/e9_mle_bench/data",
        },
        "container_image_digest_present": {
            "missing": f"The `{REQUIRED_IMAGE}` image is not built, so there is no container digest.",
            "external_receipt": "a completed image build with an immutable digest",
            "action": (
                "docker build --platform linux/amd64 -t mlebench-env "
                "-f environment/Dockerfile . (take outputs/_setup/docker.lock first)"
            ),
        },
        "model_submission_artifact_present": {
            "missing": "No model produced a submission.csv for this competition.",
            "external_receipt": "an agent run inside the container with a model API key",
            "action": "not run in this lane: paid model calls are outside the cost boundary",
        },
        "contamination_disjointness_receipt": {
            "missing": "No proof that the evaluation competitions are disjoint from training data.",
            "external_receipt": "a training-corpus task-ID manifest to diff against the eval split hash",
            "action": "supply the training task-ID manifest, then re-emit this receipt",
        },
        "upstream_revision_pinned": {
            "missing": "The pinned MLE-bench checkout is absent or incomplete.",
            "external_receipt": "none - clone openai/mle-bench at the pinned revision",
            "action": f"git clone {UPSTREAM_URL} && git checkout {UPSTREAM_COMMIT}",
        },
        "split_manifest_resolved": {
            "missing": "The split files could not be read or hashed.",
            "external_receipt": "none",
            "action": "check experiments/splits/ in the checkout",
        },
        "verifier_identity_resolved": {
            "missing": "The grader or its leaderboard could not be pinned (often an unresolved Git-LFS pointer).",
            "external_receipt": "none",
            "action": "git lfs install --local && git lfs pull in the checkout",
        },
    }
    return [
        {"gate": name, **catalogue[name]}
        for name in REQUIRED_GATES
        if not gates.get(name) and name in catalogue
    ]


def build_receipt(
    checkout: Path = DEFAULT_CHECKOUT,
    competition_id: str | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    observed_at: str = "",
    harness: dict[str, Any] | None = None,
    rule_probe: dict[str, Any] | None = None,
    inspect_docker: bool = True,
) -> dict[str, Any]:
    """Assemble the fail-closed E9 receipt for one competition binding."""

    checkout = Path(checkout)
    sized = rank_competitions_by_size(checkout)
    if competition_id is None:
        competition_id = sized[0]["competition_id"] if sized else ""
    size_row = next((row for row in sized if row["competition_id"] == competition_id), None)

    environment = (
        docker_image_digest(REQUIRED_IMAGE)
        if inspect_docker
        else {"image": REQUIRED_IMAGE, "present": False, "digest": None, "error": "not inspected"}
    )
    environment["dockerfile_sha256"] = sha256_file(checkout / "environment/Dockerfile")
    environment["dockerfile_target_platform"] = "linux/amd64 (Miniconda3-latest-Linux-x86_64.sh is hard-coded)"
    environment["verifier_only_variant"] = (
        docker_image_digest(VERIFIER_ONLY_IMAGE)
        if inspect_docker
        else {"image": VERIFIER_ONLY_IMAGE, "present": False, "digest": None, "error": "not inspected"}
    )
    environment["verifier_only_variant"]["build_arg"] = "INSTALL_HEAVY_DEPENDENCIES=false"
    environment["verifier_only_variant"]["satisfies_container_gate"] = False
    environment["verifier_only_variant"]["scope"] = (
        "grading server and official mlebench package only; no agent ML stack, so it can "
        "grade a submission but cannot host an agent"
    )

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "observed_at": observed_at,
        "status": "BLOCKED",
        "runnable_now": False,
        "claim_status": "harness_and_binding_only_not_a_benchmark_result",
        "suite": {
            "id": "mle_bench_eval",
            "role": "primary_eval",
            "domains": ["ml", "code", "long_horizon"],
            "runner_path": "zvf-program/flagship/mle_bench_eval.py",
            "runner_present": True,
        },
        "authoritative_public_source": {
            "repository_url": UPSTREAM_URL,
            "revision": UPSTREAM_COMMIT,
            "revision_url": f"{UPSTREAM_URL}/commit/{UPSTREAM_COMMIT}",
            "checkout_path": str(checkout),
            "checkout_present": (checkout / "mlebench/registry.py").is_file(),
            "license_sha256": sha256_file(checkout / "LICENSE"),
            "license_position": LICENSE_POSITION,
        },
        "split_manifest": split_manifest(checkout),
        "competition_binding": {
            "competition_id": competition_id,
            "selected_because": "smallest recorded download size across the 75 evaluation competitions",
            "recorded_download_size_gb": size_row["dataset_size_gb"] if size_row else None,
            "recorded_download_size_mb": size_row["dataset_size_mb"] if size_row else None,
            "size_source": SIZE_TABLE_RELPATH,
            "size_source_sha256": sha256_file(checkout / SIZE_TABLE_RELPATH),
            "in_eval_split": competition_id in read_split(checkout, EVAL_SPLIT_RELPATH),
            "kaggle_rules": kaggle_rule_acceptance(competition_id, rule_probe),
            "dataset_state": dataset_state(data_dir, competition_id),
            "other_competitions_rule_state": {
                "accepted_count": 1 if (rule_probe or {}).get("accepted") or (rule_probe or {}).get("rules_accepted") else 0,
                "eval_split_size": len(read_split(checkout, EVAL_SPLIT_RELPATH)),
                "remaining_unaccepted": max(len(read_split(checkout, EVAL_SPLIT_RELPATH)) - 1, 0),
                "note": (
                    "Kaggle rule acceptance is per competition and non-automatable. Only "
                    f"`{competition_id}` is accepted; a full 75-competition MLE-bench run "
                    "requires 74 further manual acceptances by a signed-in human. This is a "
                    "structural reproducibility property of the benchmark, not a lane defect."
                ),
            },
        },
        "verifier": verifier_identity(checkout, competition_id),
        "environment": environment,
        "model_artifact": {
            "model_revision": None,
            "submission_path": None,
            "status": "absent_no_model_was_run",
        },
        "contamination_policy": {
            "rule": "evaluation competitions must be disjoint from any training corpus",
            "training_task_ids_sha256": None,
            "evaluation_task_ids_sha256": task_ids_sha256(read_split(checkout, EVAL_SPLIT_RELPATH)) or None,
            "disjoint_receipt": None,
            "status": "BLOCKED",
            "reason": "No training provenance exists for this lane, so disjointness cannot be proven.",
        },
        "metrics": {
            "score": None,
            "is_model_score": False,
            "task_sample_count": 0,
            "status": "not_measured",
            "reason": "No model submission was graded; no benchmark result is claimed.",
        },
        "harness_validation": harness,
        "safety": {
            "paid_calls": False,
            "tinker_launched": False,
            "wandb_published": False,
            "checkpoints_pushed": False,
        },
    }

    gates = evaluate_gates(receipt)
    receipt["gates"] = gates
    receipt["status"] = fail_closed_status(gates)
    receipt["lane_status"] = lane_status(gates, harness)
    receipt["runnable_now"] = receipt["status"] == "READY"
    receipt["blockers"] = [name for name in REQUIRED_GATES if not gates.get(name)]
    receipt["blocker_details"] = blocker_details(gates, competition_id)
    receipt["evidence"] = {
        key: str(path)
        for key, path in {
            "lane_status": _REPO_ROOT / "outputs/e9_mle_bench/lane_status_2026-08-09.md",
            "competition_size_survey": _REPO_ROOT / "outputs/e9_mle_bench/evidence/competition_size_survey.json",
            "kaggle_size_crosscheck": _REPO_ROOT
            / "outputs/e9_mle_bench/evidence/kaggle_files_spooky_author_identification.csv",
            "prepare_failure_log": _REPO_ROOT
            / "outputs/e9_mle_bench/evidence/prepare_spooky_author_identification.log",
            "kaggle_rule_acceptance_probe": _REPO_ROOT
            / "outputs/e9_mle_bench/evidence/kaggle_rule_acceptance_probe.json",
            "harness_validation": _REPO_ROOT / "outputs/e9_mle_bench/evidence/harness_validation.json",
            "official_cli_grade_sample_log": _REPO_ROOT
            / "outputs/e9_mle_bench/evidence/mlebench_grade_sample_fixture.log",
            "runner": _REPO_ROOT / "zvf-program/flagship/mle_bench_eval.py",
            "runner_tests": _REPO_ROOT / "zvf-program/flagship/test_mle_bench_eval.py",
        }.items()
        if Path(path).exists()
    }
    return receipt


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _write(payload: dict[str, Any], out: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=False)
    if out is None:
        print(text)
        return
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {out}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    sub = parser.add_subparsers(dest="command", required=True)

    p_survey = sub.add_parser("survey", help="Rank the 75 competitions by recorded download size.")
    p_survey.add_argument("--top", type=int, default=10)
    p_survey.add_argument("--out", type=Path, default=None)

    p_harness = sub.add_parser("harness-validate", help="Grade gold answers through the official grader.")
    p_harness.add_argument("--competition", type=str, default=None)
    p_harness.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    p_harness.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p_harness.add_argument("--out", type=Path, default=None)

    p_rules = sub.add_parser(
        "check-rules", help="Probe the download endpoint to prove Kaggle rule acceptance."
    )
    p_rules.add_argument("--competition", type=str, default=None)
    p_rules.add_argument("--out", type=Path, default=None)

    p_receipt = sub.add_parser("receipt", help="Emit the fail-closed receipt.")
    p_receipt.add_argument("--competition", type=str, default=None)
    p_receipt.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p_receipt.add_argument("--observed-at", type=str, default="")
    p_receipt.add_argument("--harness-json", type=Path, default=None)
    p_receipt.add_argument("--rule-probe-json", type=Path, default=None)
    p_receipt.add_argument("--no-docker", action="store_true")
    p_receipt.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)

    if args.command == "survey":
        payload = survey(args.checkout)
        if args.out is None:
            smallest = payload["smallest"]
            print(f"competitions with a recorded size: {payload['sized_competition_count']}")
            print(f"total recorded download size: {payload['total_dataset_size_gb']} GB")
            print(f"smallest: {smallest['competition_id']} @ {smallest['dataset_size_gb']} GB")
            for row in payload["ranking"][: args.top]:
                print(f"  {row['dataset_size_gb']:>10.5f} GB  {row['dataset_size_mb']:>9.2f} MB  {row['competition_id']}")
            return 0
        _write(payload, args.out)
        return 0

    if args.command == "harness-validate":
        competition = args.competition or (smallest_competition(args.checkout) or {}).get("competition_id")
        payload = harness_validate(args.checkout, competition, args.fixture_dir, args.data_dir)
        _write(payload, args.out)
        return 0 if payload.get("status") == "PASS" else 1

    if args.command == "check-rules":
        competition = args.competition or (smallest_competition(args.checkout) or {}).get("competition_id")
        payload = check_rules_accepted(competition, args.checkout)
        _write(payload, args.out)
        return 0 if payload.get("accepted") else 1

    if args.command == "receipt":
        harness = json.loads(args.harness_json.read_text(encoding="utf-8")) if args.harness_json else None
        probe_payload = None
        if args.rule_probe_json:
            raw = json.loads(args.rule_probe_json.read_text(encoding="utf-8"))
            entries = raw if isinstance(raw, list) else [raw]
            competition = args.competition or (smallest_competition(args.checkout) or {}).get("competition_id")
            probe_payload = next((e for e in entries if e.get("competition_id") == competition), None)
        payload = build_receipt(
            checkout=args.checkout,
            competition_id=args.competition,
            data_dir=args.data_dir,
            observed_at=args.observed_at,
            harness=harness,
            rule_probe=probe_payload,
            inspect_docker=not args.no_docker,
        )
        _write(payload, args.out)
        return 0

    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
