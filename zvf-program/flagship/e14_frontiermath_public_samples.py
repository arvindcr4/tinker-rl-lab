#!/usr/bin/env python3
"""E14 FrontierMath — public sample transcript parser, hasher, and fail-closed receipt.

WHAT THIS MODULE OPERATES ON
----------------------------
Epoch AI publishes ``sample_question_transcripts.zip``: 150 JSONL chat
transcripts covering **five representative sample problems**.  Those five
problems are *published illustrations*.  They are **not** the FrontierMath
benchmark, which is a private held-out problem set that Epoch AI does not
distribute.

Consequently this module is structurally incapable of producing a FrontierMath
score:

* the public samples carry **no ground-truth answers**,
* they carry **no grader verdicts**,
* and five published illustrations are not a benchmark split.

Every public entry point therefore either (a) returns descriptive statistics
that are explicitly labelled as sample-corpus statistics, or (b) raises
:class:`ScoreProhibited`.  :func:`build_blocked_receipt` hard-codes
``status="BLOCKED"`` and ``score=None`` and re-validates that invariant before
returning.

This module performs no network I/O and never executes transcript code blocks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "e14-frontiermath-public-samples-v1"
MODULE_ID = "e14-frontiermath-public-sample-parser"

# --------------------------------------------------------------------------
# Provenance / labelling constants.  These strings are load-bearing: they are
# stamped onto every artifact this module emits so that no downstream consumer
# can mistake sample-corpus statistics for a benchmark result.
# --------------------------------------------------------------------------

PUBLIC_SAMPLE_SOURCE_URL = "https://epoch.ai/files/sample_question_transcripts.zip"

ARTIFACT_LABEL = "REPRESENTATIVE_PUBLIC_SAMPLES_NOT_THE_BENCHMARK"

ARTIFACT_DISCLAIMER = (
    "These artifacts are derived from Epoch AI's PUBLIC REPRESENTATIVE SAMPLE "
    "TRANSCRIPTS (5 published sample problems). They are NOT the FrontierMath "
    "benchmark. FrontierMath is a private held-out problem set that is not "
    "distributed. No number computed from these samples is a FrontierMath "
    "score, an approximation of one, or a lower/upper bound on one. The sample "
    "corpus contains no ground-truth answers and no grader verdicts."
)

NOT_A_SUBSTITUTE = (
    "GSM8K, MATH / MATH-500, MathArena, AIME sets, and any other public math "
    "benchmark are NOT substitutes for FrontierMath. A score on any of them is "
    "not a FrontierMath score."
)

# --------------------------------------------------------------------------
# Observed structure of the public corpus (verified 2026-08-09 against the
# extracted archive; see MODULE DOCSTRING for the provenance).
# --------------------------------------------------------------------------

EXPECTED_TRANSCRIPT_COUNT = 150
EXPECTED_MODELS = (
    "claude-3-5-sonnet-20241022",
    "gemini-1.5-pro-002",
    "gpt-4o-2024-08-06",
    "grok-beta",
    "o1-mini",
    "o1-preview",
)
EXPECTED_PROBLEM_TOKENS = ("ALL3", "CWA2", "CWD31", "RAP1", "TIK2")
EXPECTED_RUNS_PER_CELL = 5

FILENAME_RE = re.compile(r"^(?P<model>.+)_(?P<problem>[A-Za-z]+\d+)_run-(?P<run>\d+)\.jsonl$")

#: The literal comment the Epoch harness greps for to locate the final answer.
FINAL_ANSWER_MARKER = "# This is the final answer"

#: The pickle file the final-answer script must write.  The grader unpickles it.
FINAL_ANSWER_PICKLE = "final_answer.p"

CODE_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)

PROBLEM_STATEMENT_MARKER = "Here is the mathematical problem you need to solve:"
RETURN_TYPE_MARKER = "The expected return type of final_answer is the following:"

# Harness (non-first ``user``) turn taxonomy.  Every follow-up user turn in the
# public corpus falls into exactly one of these three shapes.
TURN_TASK_PROMPT = "task_prompt"
TURN_CODE_RESULT = "code_execution_result"
TURN_FINAL_ANSWER_FAILED_STDERR = "final_answer_failed_stderr"
TURN_FINAL_ANSWER_FAILED_TIMEOUT = "final_answer_failed_timeout"
TURN_UNKNOWN = "unknown_harness_turn"

ALLOWED_MESSAGE_KEYS = frozenset({"role", "content"})
ALLOWED_ROLES = frozenset({"user", "assistant"})


class ScoreProhibited(RuntimeError):
    """Raised whenever any caller attempts to derive a score from public samples.

    This is the fail-closed guard.  It is not recoverable by argument: there is
    no ground truth in the corpus, so a correctness rate is not computable, and
    even if it were it would not be a FrontierMath score.
    """


class TranscriptSchemaError(ValueError):
    """Raised when a transcript does not match the validated public-sample schema."""


@dataclass(frozen=True)
class TranscriptIdentity:
    """Model / problem / run triple encoded in a public sample filename."""

    model: str
    problem_token: str
    run_index: int
    filename: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "problem_token": self.problem_token,
            "run_index": self.run_index,
            "filename": self.filename,
        }


@dataclass
class TranscriptFacts:
    """Everything this module can factually state about one public transcript.

    Deliberately contains no ``correct``, ``score``, ``reward``, or ``verdict``
    field — those are not derivable from the corpus.
    """

    identity: TranscriptIdentity
    sha256: str
    message_count: int
    assistant_turns: int
    user_turns: int
    harness_turn_kinds: dict[str, int]
    code_block_count: int
    final_answer_block_count: int
    has_final_answer: bool
    declared_return_type: str | None
    problem_statement_sha256: str | None
    preamble_sha256: str | None
    ground_truth_present: bool = False
    grader_verdict_present: bool = False
    schema_errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "identity": self.identity.as_dict(),
            "sha256": self.sha256,
            "message_count": self.message_count,
            "assistant_turns": self.assistant_turns,
            "user_turns": self.user_turns,
            "harness_turn_kinds": dict(sorted(self.harness_turn_kinds.items())),
            "code_block_count": self.code_block_count,
            "final_answer_block_count": self.final_answer_block_count,
            "has_final_answer": self.has_final_answer,
            "declared_return_type": self.declared_return_type,
            "problem_statement_sha256": self.problem_statement_sha256,
            "preamble_sha256": self.preamble_sha256,
            "ground_truth_present": self.ground_truth_present,
            "grader_verdict_present": self.grader_verdict_present,
            "schema_errors": list(self.schema_errors),
        }
        return payload


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def parse_transcript_filename(filename: str) -> TranscriptIdentity:
    """Decode ``<model>_<PROBLEM>_run-<n>.jsonl`` into its parts."""
    name = Path(filename).name
    match = FILENAME_RE.match(name)
    if match is None:
        raise TranscriptSchemaError(f"filename does not match public sample convention: {name!r}")
    return TranscriptIdentity(
        model=match.group("model"),
        problem_token=match.group("problem"),
        run_index=int(match.group("run")),
        filename=name,
    )


def load_transcript(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL transcript into a list of message dicts (no validation)."""
    messages: list[dict[str, Any]] = []
    text = Path(path).read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TranscriptSchemaError(f"{Path(path).name}:{lineno}: invalid JSON: {exc}") from exc
        messages.append(obj)
    return messages


def validate_transcript(messages: Iterable[Any]) -> list[str]:
    """Validate against the schema observed across all 150 public samples.

    Schema (verified, not assumed):

    * a non-empty list of JSON objects, one per line;
    * each object has exactly the keys ``role`` and ``content``;
    * ``role`` is ``user`` or ``assistant``; ``content`` is a non-empty ``str``;
    * roles strictly alternate;
    * the first message is ``user`` (the task prompt) and the last is
      ``assistant``.

    Returns a list of human-readable errors; empty means valid.
    """
    errors: list[str] = []
    messages = list(messages)
    if not messages:
        return ["transcript is empty"]

    previous_role: str | None = None
    for index, message in enumerate(messages):
        prefix = f"message[{index}]"
        if not isinstance(message, dict):
            errors.append(f"{prefix}: expected object, got {type(message).__name__}")
            previous_role = None
            continue
        keys = set(message.keys())
        if keys != ALLOWED_MESSAGE_KEYS:
            missing = sorted(ALLOWED_MESSAGE_KEYS - keys)
            extra = sorted(keys - ALLOWED_MESSAGE_KEYS)
            if missing:
                errors.append(f"{prefix}: missing key(s) {missing}")
            if extra:
                errors.append(f"{prefix}: unexpected key(s) {extra}")
        role = message.get("role")
        content = message.get("content")
        if role not in ALLOWED_ROLES:
            errors.append(f"{prefix}: role must be one of {sorted(ALLOWED_ROLES)}, got {role!r}")
        if not isinstance(content, str):
            errors.append(f"{prefix}: content must be str, got {type(content).__name__}")
        elif not content.strip():
            errors.append(f"{prefix}: content is empty")
        if previous_role is not None and role == previous_role:
            errors.append(f"{prefix}: role {role!r} repeats; roles must alternate")
        previous_role = role if role in ALLOWED_ROLES else None

    first = messages[0]
    if isinstance(first, dict) and first.get("role") != "user":
        errors.append("first message must be the user task prompt")
    last = messages[-1]
    if isinstance(last, dict) and last.get("role") != "assistant":
        errors.append("last message must be an assistant turn")
    return errors


def classify_harness_turn(content: str, *, is_first: bool = False) -> str:
    """Classify a ``user`` turn.

    ``user`` turns are written by Epoch's harness, not by a human.  The first is
    the task prompt; the rest are code-execution feedback.
    """
    if is_first:
        return TURN_TASK_PROMPT
    head = content.lstrip()
    if head.startswith("Results from executing code block"):
        return TURN_CODE_RESULT
    if head.startswith("Final answer failed"):
        if "timed out" in head[:80]:
            return TURN_FINAL_ANSWER_FAILED_TIMEOUT
        return TURN_FINAL_ANSWER_FAILED_STDERR
    return TURN_UNKNOWN


def extract_code_blocks(content: str) -> list[str]:
    """Return every fenced code block in an assistant turn."""
    return CODE_FENCE_RE.findall(content)


def extract_final_answer_blocks(messages: Iterable[dict[str, Any]]) -> list[str]:
    """Return every assistant code block carrying :data:`FINAL_ANSWER_MARKER`.

    This mirrors how Epoch's grader locates a submission: it greps the transcript
    for the code block containing the literal marker comment, executes it, and
    unpickles ``final_answer.p``.  Locating the block is all this module does —
    it never executes it, and it has nothing to compare the result against.
    """
    blocks: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        for block in extract_code_blocks(content):
            if FINAL_ANSWER_MARKER in block:
                blocks.append(block)
    return blocks


def split_task_prompt(content: str) -> tuple[str, str | None, str | None]:
    """Split the first user turn into (preamble, problem statement, return type)."""
    preamble, marker, rest = content.partition(PROBLEM_STATEMENT_MARKER)
    if not marker:
        return content, None, None
    statement, ret_marker, return_type = rest.partition(RETURN_TYPE_MARKER)
    return (
        preamble,
        statement.strip() or None,
        return_type.strip() if ret_marker else None,
    )


def detect_ground_truth(messages: Iterable[dict[str, Any]]) -> bool:
    """Report whether a transcript carries a ground-truth answer.

    Always ``False`` for the public corpus.  This is a *positive check*, not an
    assumption: a ground truth would have to appear either as a structured field
    (rejected by :func:`validate_transcript`, which permits only ``role`` and
    ``content``) or as harness text.  Harness text is exhaustively classified by
    :func:`classify_harness_turn` into the task prompt and code-execution
    feedback, neither of which reveals the answer.
    """
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if set(message.keys()) - ALLOWED_MESSAGE_KEYS:
            return True  # an out-of-schema field could carry it; fail loud.
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        if classify_harness_turn(content, is_first=index == 0) == TURN_UNKNOWN:
            return True  # unrecognised harness turn: refuse to claim absence.
    return False


def detect_grader_verdict(messages: Iterable[dict[str, Any]]) -> bool:
    """Report whether a transcript carries a grader verdict.  Same logic as above."""
    return detect_ground_truth(messages)


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(payload: str) -> str:
    return sha256_bytes(payload.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_digest(file_hashes: dict[str, str]) -> str:
    """Order-independent digest over ``{filename: sha256}``.

    Canonical form is ``"<filename>  <sha256>\\n"`` per entry, sorted by
    filename, so the digest is stable across filesystems and traversal order.
    """
    lines = "".join(f"{name}  {file_hashes[name]}\n" for name in sorted(file_hashes))
    return sha256_text(lines)


# --------------------------------------------------------------------------
# Corpus-level facts
# --------------------------------------------------------------------------


def analyze_transcript(path: str | Path) -> TranscriptFacts:
    """Produce the factual record for one public sample transcript."""
    path = Path(path)
    identity = parse_transcript_filename(path.name)
    messages = load_transcript(path)
    errors = validate_transcript(messages)

    harness_kinds: Counter[str] = Counter()
    code_blocks = 0
    declared_return_type: str | None = None
    statement_hash: str | None = None
    preamble_hash: str | None = None

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if message.get("role") == "user":
            harness_kinds[classify_harness_turn(content, is_first=index == 0)] += 1
            if index == 0:
                preamble, statement, return_type = split_task_prompt(content)
                preamble_hash = sha256_text(preamble)
                statement_hash = sha256_text(statement) if statement else None
                declared_return_type = return_type
        else:
            code_blocks += len(extract_code_blocks(content))

    final_blocks = extract_final_answer_blocks(messages)
    return TranscriptFacts(
        identity=identity,
        sha256=sha256_file(path),
        message_count=len(messages),
        assistant_turns=sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant"),
        user_turns=sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "user"),
        harness_turn_kinds=dict(harness_kinds),
        code_block_count=code_blocks,
        final_answer_block_count=len(final_blocks),
        has_final_answer=bool(final_blocks),
        declared_return_type=declared_return_type,
        problem_statement_sha256=statement_hash,
        preamble_sha256=preamble_hash,
        ground_truth_present=detect_ground_truth(messages),
        grader_verdict_present=detect_grader_verdict(messages),
        schema_errors=errors,
    )


def build_public_sample_manifest(samples_dir: str | Path) -> dict[str, Any]:
    """Build the immutable, fully labelled manifest over the public sample corpus.

    The manifest is descriptive only.  It contains counts and hashes.  It
    contains no score, no accuracy, and no correctness field, because the corpus
    supplies no ground truth against which any of those could be computed.
    """
    samples_dir = Path(samples_dir)
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"public sample directory not found: {samples_dir}")

    paths = sorted(p for p in samples_dir.iterdir() if p.suffix == ".jsonl")
    if not paths:
        raise FileNotFoundError(f"no .jsonl transcripts under {samples_dir}")

    facts = [analyze_transcript(p) for p in paths]
    file_hashes = {f.identity.filename: f.sha256 for f in facts}

    models = Counter(f.identity.model for f in facts)
    problems = Counter(f.identity.problem_token for f in facts)
    runs = Counter(f.identity.run_index for f in facts)
    grid: dict[str, dict[str, int]] = defaultdict(dict)
    for fact in facts:
        cell = grid[fact.identity.model]
        cell[fact.identity.problem_token] = cell.get(fact.identity.problem_token, 0) + 1

    statements: dict[str, set[str]] = defaultdict(set)
    return_types: dict[str, set[str]] = defaultdict(set)
    preambles: set[str] = set()
    for fact in facts:
        if fact.problem_statement_sha256:
            statements[fact.identity.problem_token].add(fact.problem_statement_sha256)
        if fact.declared_return_type:
            return_types[fact.identity.problem_token].add(fact.declared_return_type)
        if fact.preamble_sha256:
            preambles.add(fact.preamble_sha256)

    harness_totals: Counter[str] = Counter()
    schema_errors: dict[str, list[str]] = {}
    for fact in facts:
        harness_totals.update(fact.harness_turn_kinds)
        if fact.schema_errors:
            schema_errors[fact.identity.filename] = fact.schema_errors

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "module_id": MODULE_ID,
        "artifact_label": ARTIFACT_LABEL,
        "artifact_disclaimer": ARTIFACT_DISCLAIMER,
        "not_a_substitute": NOT_A_SUBSTITUTE,
        "is_benchmark_split": False,
        "is_model_score": False,
        "source_url": PUBLIC_SAMPLE_SOURCE_URL,
        "samples_dir": str(samples_dir),
        "corpus": {
            "transcript_count": len(facts),
            "expected_transcript_count": EXPECTED_TRANSCRIPT_COUNT,
            "grid_is_complete": _grid_is_complete(grid, problems, runs),
            "models": dict(sorted(models.items())),
            "problem_tokens": dict(sorted(problems.items())),
            "runs_per_index": dict(sorted(runs.items())),
            "model_by_problem": {m: dict(sorted(c.items())) for m, c in sorted(grid.items())},
        },
        "prompt_contract": {
            "distinct_preamble_hashes": sorted(preambles),
            "preamble_is_uniform": len(preambles) == 1,
            "final_answer_marker": FINAL_ANSWER_MARKER,
            "final_answer_pickle": FINAL_ANSWER_PICKLE,
            "problem_statement_hashes": {
                token: sorted(hashes) for token, hashes in sorted(statements.items())
            },
            "declared_return_types": {
                token: sorted(values) for token, values in sorted(return_types.items())
            },
        },
        "harness_turn_totals": dict(sorted(harness_totals.items())),
        "gradability": {
            "ground_truth_present_anywhere": any(f.ground_truth_present for f in facts),
            "grader_verdict_present_anywhere": any(f.grader_verdict_present for f in facts),
            "transcripts_with_final_answer_block": sum(1 for f in facts if f.has_final_answer),
            "transcripts_without_final_answer_block": [
                f.identity.filename for f in facts if not f.has_final_answer
            ],
            "gradable_locally": False,
            "gradable_locally_reason": (
                "The public sample corpus carries no ground-truth answers and no grader "
                "verdicts. Correctness cannot be computed. Even with ground truth, five "
                "published sample problems are not the FrontierMath benchmark."
            ),
        },
        "hashes": {
            "corpus_sha256": corpus_digest(file_hashes),
            "file_sha256": dict(sorted(file_hashes.items())),
        },
        "schema_validation": {
            "all_transcripts_valid": not schema_errors,
            "invalid_transcripts": schema_errors,
        },
        "score": None,
        "score_refusal_reason": (
            "Fail-closed: this module never emits a score for frontiermath_eval. "
            "See gradability.gradable_locally_reason."
        ),
        "facts": [f.as_dict() for f in facts],
    }
    return manifest


def _grid_is_complete(
    grid: dict[str, dict[str, int]],
    problems: Counter[str],
    runs: Counter[int],
) -> bool:
    if sorted(grid) != sorted(EXPECTED_MODELS):
        return False
    if sorted(problems) != sorted(EXPECTED_PROBLEM_TOKENS):
        return False
    if sorted(runs) != list(range(1, EXPECTED_RUNS_PER_CELL + 1)):
        return False
    return all(
        cells.get(token) == EXPECTED_RUNS_PER_CELL
        for cells in grid.values()
        for token in EXPECTED_PROBLEM_TOKENS
    )


# --------------------------------------------------------------------------
# Fail-closed scoring guards
# --------------------------------------------------------------------------


def compute_frontiermath_score(*_args: Any, **_kwargs: Any) -> float:
    """Always raises.  There is no code path that yields a FrontierMath score here.

    Kept as a named symbol so that any caller reaching for a score fails loudly
    and traceably instead of silently inventing one from the sample corpus.
    """
    raise ScoreProhibited(
        "Refusing to emit a FrontierMath score. "
        + ARTIFACT_DISCLAIMER
        + " "
        + NOT_A_SUBSTITUTE
    )


def assert_receipt_emits_no_score(receipt: dict[str, Any]) -> None:
    """Re-validate the fail-closed invariant on an assembled receipt."""
    if receipt.get("status") != "BLOCKED":
        raise ScoreProhibited(f"receipt status must be BLOCKED, got {receipt.get('status')!r}")
    if receipt.get("score") is not None:
        raise ScoreProhibited(f"receipt score must be null, got {receipt.get('score')!r}")
    experiment = receipt.get("experiment", {})
    if experiment.get("measured_metrics") is not None:
        raise ScoreProhibited("receipt must not carry measured_metrics")
    if experiment.get("related_benchmark_substitution"):
        raise ScoreProhibited("receipt must not substitute a related benchmark")
    if receipt.get("is_model_score"):
        raise ScoreProhibited("receipt must declare is_model_score=false")
    label = receipt.get("public_sample_asset", {}).get("artifact_label")
    if label != ARTIFACT_LABEL:
        raise ScoreProhibited(f"receipt is missing the public-sample label: {label!r}")


def build_blocked_receipt(
    manifest: dict[str, Any],
    *,
    recorded_at: str,
    checkout: str,
    commit: str,
    archive_path: str | None = None,
    archive_sha256: str | None = None,
    blockers: list[str] | None = None,
    access_document: str | None = None,
    evidence_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the E14 receipt.  Status is always BLOCKED; score is always null."""
    receipt: dict[str, Any] = {
        "schema_version": "e14-frontiermath-blocked-receipt-v1",
        "status": "BLOCKED",
        "score": None,
        "is_model_score": False,
        "recorded_at": recorded_at,
        "suite": {
            "suite_id": "frontiermath_eval",
            "suite_role": "primary_eval",
            "domain": "math",
            "benchmark_owner": "Epoch AI",
            "contract_split": "private held-out evaluation set",
            "data_release_model": "not distributed to third parties",
        },
        "score_refusal": {
            "refused": True,
            "guard": f"{MODULE_ID}.compute_frontiermath_score",
            "reason": ARTIFACT_DISCLAIMER,
            "not_a_substitute": NOT_A_SUBSTITUTE,
        },
        "execution_source": {
            "checkout": checkout,
            "commit": commit,
        },
        "public_sample_asset": {
            "artifact_label": ARTIFACT_LABEL,
            "artifact_disclaimer": ARTIFACT_DISCLAIMER,
            "source_url": PUBLIC_SAMPLE_SOURCE_URL,
            "archive_path": archive_path,
            "archive_sha256": archive_sha256,
            "samples_dir": manifest.get("samples_dir"),
            "transcript_count": manifest.get("corpus", {}).get("transcript_count"),
            "corpus_sha256": manifest.get("hashes", {}).get("corpus_sha256"),
            "is_benchmark_split": False,
            "evaluation_use": "characterization_only_not_used_as_score",
        },
        "corpus_facts": {
            "corpus": manifest.get("corpus"),
            "prompt_contract": manifest.get("prompt_contract"),
            "harness_turn_totals": manifest.get("harness_turn_totals"),
            "gradability": manifest.get("gradability"),
            "schema_validation": manifest.get("schema_validation"),
        },
        "experiment": {
            "attempted": False,
            "evaluated_task_count": 0,
            "evaluated_sample_count": 0,
            "measured_metrics": None,
            "related_benchmark_substitution": False,
            "harness_validation": None,
        },
        "blockers": blockers or [],
        "access_document": access_document,
        "evidence_paths": evidence_paths or [],
    }
    assert_receipt_emits_no_score(receipt)
    return receipt


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="e14_frontiermath_public_samples",
        description=(
            "Characterize and hash Epoch AI's PUBLIC REPRESENTATIVE FrontierMath sample "
            "transcripts. Never emits a FrontierMath score."
        ),
        epilog=ARTIFACT_DISCLAIMER,
    )
    parser.add_argument(
        "--samples-dir",
        required=True,
        help="directory holding the extracted sample_question_transcripts/*.jsonl",
    )
    parser.add_argument("--manifest-out", help="write the full manifest JSON here")
    parser.add_argument("--receipt-out", help="write the BLOCKED receipt JSON here")
    parser.add_argument("--archive", help="path to frontiermath_public_samples.zip (hashed if given)")
    parser.add_argument("--recorded-at", default="", help="ISO date stamped on the receipt")
    parser.add_argument("--checkout", default=str(Path.cwd()), help="repo checkout path")
    parser.add_argument("--commit", default="", help="git commit recorded on the receipt")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="print the summary without the per-transcript facts array",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = build_public_sample_manifest(args.samples_dir)

    archive_sha = sha256_file(args.archive) if args.archive else None

    if args.manifest_out:
        Path(args.manifest_out).write_text(
            json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )

    if args.receipt_out:
        receipt = build_blocked_receipt(
            manifest,
            recorded_at=args.recorded_at,
            checkout=args.checkout,
            commit=args.commit,
            archive_path=args.archive,
            archive_sha256=archive_sha,
        )
        Path(args.receipt_out).write_text(
            json.dumps(receipt, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )

    summary = {k: v for k, v in manifest.items() if k != "facts"} if args.summary_only else manifest
    print(json.dumps(summary, indent=2, sort_keys=False))
    print(f"\n{ARTIFACT_LABEL}: {ARTIFACT_DISCLAIMER}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
