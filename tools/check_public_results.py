#!/usr/bin/env python3
"""Check public C1 scientific records and document tables, without raw execution.

This checks consistency of published review labels, embedded text, counts and
bindings. It does not independently adjudicate mathematics, regenerate withheld
completions, establish all-screen eligibility, or authenticate original sources.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys


# Direct script execution must use this checkout's unchanged public parser.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.public_analysis.parser_v2 import normalize_number


REVISION = "reports/public_revision_2026-10-03"
RESULTS = "research/public_results"
LEDGER = f"{RESULTS}/c1_case_ledger.json"
SUMMARY = f"{RESULTS}/c1_result_summary.json"
METHOD = f"{REVISION}/addendum/scientific_method_public.json"
BINDINGS = f"{REVISION}/thesis/c1_reference_bindings.json"
PROVENANCE = "research/public_analysis/PROVENANCE.json"
DOCUMENTS = (
    f"{REVISION}/thesis/ch_back_c1_evidence.md",
    f"{REVISION}/addendum/C1_Research_Addendum_2026-10-03_PUBLIC.md",
)
INPUT_FILES = (
    LEDGER,
    SUMMARY,
    METHOD,
    BINDINGS,
    PROVENANCE,
    f"{RESULTS}/c1_case_ledger.csv",
    f"{REVISION}/thesis/c1_case_ledger.json",
    f"{REVISION}/thesis/c1_case_ledger.csv",
    f"{REVISION}/addendum/case_ledger_public.json",
    f"{REVISION}/addendum/case_ledger_public.csv",
    *DOCUMENTS,
)
LIMITATION = (
    "Public-record consistency only: no raw execution replay, independent mathematical "
    "validation, full-screen eligibility reconstruction, absolute generation-seed origin "
    "reconstruction, or original-source authentication."
)
CASE_FLAGS = (
    "first_second_category_disagreement",
    "frozen_primary_event",
    "conservative_corroborated_numeric_recovery",
)
IDENTITY_FIELDS = (
    "selected_index",
    "original_prompt_rank",
    "audit_id",
    "prompt_id",
    "question_sha256",
    "recorded_gold_reasoning_sha256",
)
CATEGORIES = {
    "clear_wrong_gold": "W",
    "materially_ambiguous": "A",
    "inconsistent_question": "I",
    "correct_final_value_with_flawed_reference_reasoning": "R",
    "supported": "S",
    "supported_with_conventional_assumptions": "C",
}
CATEGORY_FIRST = {
    "clear_wrong_gold": "disputed",
    "inconsistent_question": "disputed",
    "materially_ambiguous": "ambiguous",
    "correct_final_value_with_flawed_reference_reasoning": "supported",
    "supported": "supported",
    "supported_with_conventional_assumptions": "supported",
}
EVENT_PARTITIONS = {
    "corroborated_numeric_recovery": "Corroborated",
    "ordinary_reading_support_first_second_disagreement": "Convention / disagreement",
    "conventional_model_support_first_second_disagreement": "Convention / disagreement",
    "ordinary_period_reading_support_first_second_disagreement": "Convention / disagreement",
    "material_ambiguity_or_underdetermination": "Ambiguous",
    "material_aggregation_ambiguity": "Ambiguous",
    "material_probability_event_ambiguity": "Ambiguous",
    "underdetermined_with_invalid_gold_match_reasoning": "Ambiguous",
    "wrong_gold_match_after_correct_initial_answers": "Wrong reference",
}
CSV_FIELDS = (
    "selected_index",
    "original_prompt_rank",
    "audit_id",
    "prompt_id",
    "recorded_gold_value",
    "first_gold_review",
    "second_exclusive_category",
    "first_second_category_disagreement",
    "frozen_primary_event",
    "conservative_corroborated_numeric_recovery",
    "reconciliation_note",
)
ADDENDUM_CSV_FIELDS = (
    "selected_index",
    "original_prompt_rank",
    "question_sha256",
    "recorded_gold_reasoning_sha256",
    "recorded_gold_value",
    "first_gold_review",
    "second_exclusive_category",
    "first_second_category_disagreement",
    "reconciliation_note",
    "frozen_primary_event",
    "conservative_corroborated_numeric_recovery",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(actual, expected, label):
    """JSON equality that cannot conflate true, 1, and 1.0."""
    require(type(actual) is type(expected), f"{label}: incorrect value type")
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), f"{label}: fields differ")
        for key in expected:
            exact(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f"{label}: length differs")
        for index, value in enumerate(expected):
            exact(actual[index], value, f"{label}[{index}]")
    else:
        require(actual == expected, f"{label}: differs from published-record derivation")


def count(value, label):
    require(type(value) is int and value >= 0, f"{label}: expected nonnegative integer")
    return value


def text(value, label):
    require(isinstance(value, str) and bool(value), f"{label}: expected nonempty string")
    return value


def canonical_number(value, label):
    text(value, label)
    require(normalize_number(value) == value, f"{label}: not a canonical frozen-v2 numeric value")


def digest(value, label):
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
        f"{label}: expected lowercase SHA-256",
    )


def sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"{path.name}: duplicate JSON key {key}")
            result[key] = value
        return result

    def finite(value):
        result = float(value)
        require(math.isfinite(result), f"{path.name}: nonfinite JSON number")
        return result

    def constant(value):
        raise ValueError(f"{path.name}: nonfinite JSON constant {value}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_float=finite,
        parse_constant=constant,
    )


def numeric_equal(actual, expected, label):
    require(
        type(actual) in (int, float) and math.isfinite(actual), f"{label}: expected finite number"
    )
    require(
        math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14),
        f"{label}: arithmetic mismatch",
    )


def check_interval(record, events, denominator, label, *, wilson=False):
    exact(record["events"], events, f"{label}.events")
    exact(record["denominator"], denominator, f"{label}.denominator")
    require(0 <= events <= denominator and denominator > 0, f"{label}: invalid denominator")
    fraction = events / denominator
    numeric_equal(record["fraction"], fraction, f"{label}.fraction")
    if "percent" in record:
        numeric_equal(record["percent"], 100 * fraction, f"{label}.percent")
    if wilson:
        z = 1.959963984540054
        den = 1 + z * z / denominator
        center = (fraction + z * z / (2 * denominator)) / den
        half = (
            z
            * math.sqrt(
                fraction * (1 - fraction) / denominator + z * z / (4 * denominator * denominator)
            )
            / den
        )
        bounds = [max(0.0, center - half), min(1.0, center + half)]
        require(
            isinstance(record["wilson_95"], list) and len(record["wilson_95"]) == 2,
            f"{label}: expected two Wilson bounds",
        )
        for i, expected in enumerate(bounds):
            numeric_equal(record["wilson_95"][i], expected, f"{label}.wilson_95[{i}]")
        exact(record["status"], "DEFINED", f"{label}.status")
        if events:
            exact(record["zero_events_exact_upper_one_sided_95"], None, f"{label}.zero_bound")
        else:
            numeric_equal(
                record["zero_events_exact_upper_one_sided_95"],
                -math.expm1(math.log(0.05) / denominator),
                f"{label}.zero_bound",
            )
        if "wilson95_percent_display" in record:
            exact(
                record["wilson95_percent_display"],
                [round(100 * x, 2) for x in bounds],
                f"{label}.wilson95_percent_display",
            )


def check_case_references(rows, cases, label):
    require(isinstance(rows, list), f"{label}: expected case-reference list")
    indices = []
    for row in rows:
        index = count(row["selected_index"], f"{label}.selected_index")
        require(index < len(cases), f"{label}: case reference out of range")
        exact(row, {key: cases[index][key] for key in IDENTITY_FIELDS}, label)
        indices.append(index)
    require(indices == sorted(set(indices)), f"{label}: duplicate or reordered case references")
    return indices


def check_summary_references(node, cases, label="summary"):
    """Validate every nested public summary reference, including excluded groups."""
    if isinstance(node, dict):
        if "cases" in node:
            indices = check_case_references(node["cases"], cases, label)
            if "count" in node:
                exact(node["count"], len(indices), f"{label}.count")
        for key, value in node.items():
            if key != "cases":
                check_summary_references(value, cases, f"{label}.{key}")
    elif isinstance(node, list) and node and isinstance(node[0], dict):
        if "selected_index" in node[0]:
            check_case_references(node, cases, label)


def check_csv(path, cases, fields):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        exact(reader.fieldnames, list(fields), f"{path.name}: CSV header")
        rows = list(reader)
    exact(
        rows, [{key: str(case[key]) for key in fields} for case in cases], f"{path}: CSV projection"
    )


def check_documents(root, cases, events, method):
    expected_cases = []
    for case in cases:
        category = CATEGORIES[case["second_exclusive_category"]]
        category += "*" if case["first_second_category_disagreement"] else ""
        flags = (
            "E N"
            if case["conservative_corroborated_numeric_recovery"]
            else ("E" if case["frozen_primary_event"] else "-")
        )
        expected_cases.append(
            [
                f"C{case['selected_index']:02d}",
                str(case["original_prompt_rank"]),
                case["recorded_gold_value"],
                category,
                flags,
            ]
        )
    expected_events = [
        [
            event["selected_index"],
            event["original_prompt_rank"],
            event["recorded_gold_value"],
            ", ".join(map(str, event["frozen_primary_success_indices"])),
            EVENT_PARTITIONS[event["event_evidence_classification"]],
        ]
        for event in events
    ]
    for relative in DOCUMENTS:
        case_rows, event_rows = [], []
        for line in (root / relative).read_text(encoding="utf-8").splitlines():
            if not line.startswith("| C"):
                continue
            columns = [column.strip() for column in line.strip("|").split("|")]
            if re.fullmatch(r"C\d+", columns[0]):
                case_rows.append(columns)
            else:
                match = re.fullmatch(r"C(\d+) .+, (\d+)", columns[0])
                if match:
                    event_rows.append([int(match[1]), int(match[2]), *columns[1:]])
        exact(case_rows, expected_cases, f"{relative}: case table")
        exact(event_rows, expected_events, f"{relative}: event table")
    method_prose = (root / DOCUMENTS[1]).read_text(encoding="utf-8")
    for pattern, value, label in (
        (r"ordering seed was (\d+)\.", method["seed"], "ordering seed"),
        (
            r"affine multiplier (\d+) modulo",
            method["seed_scheme"]["affine_multiplier"],
            "multiplier",
        ),
        (r"modulo (\d+) and generation-slot", method["seed_scheme"]["affine_modulus"], "modulus"),
        (
            r"generation-slot offset (\d+)\.",
            method["generation_seed_slot_offset"],
            "generation slot offset",
        ),
    ):
        matches = re.findall(pattern, method_prose)
        require(len(matches) == 1, f"method prose: missing/duplicate {label}")
        exact(value, int(matches[0]), f"method prose.{label}")


def check_method(method, cases, summary, provenance):
    exact(method["schema"], "c1-public-scientific-method-v1", "method.schema")
    ordering_seed = count(method["seed"], "ordering seed")
    ordered_hashes = [sha(f"{ordering_seed}:order:{case['prompt_id']}") for case in cases]
    require(
        ordered_hashes == sorted(ordered_hashes), "method: selected deterministic order mismatch"
    )
    for key, expected in {
        "initial_draws": 8,
        "fresh_draws": 24,
        "screen_prompts": 4096,
        "max_selected": 64,
        "parser_version": 2,
    }.items():
        exact(method[key], expected, f"method.{key}")
    exact(method["prompt_suffix_sha256"], sha(method["prompt_suffix"]), "method.prompt_suffix hash")
    exact(
        method["parser_sha256"],
        provenance["sources"]["original_parser_policy_2"]["original_code_sha256"],
        "method original parser source binding",
    )
    scheme = method["seed_scheme"]
    modulus = count(scheme["affine_modulus"], "seed modulus")
    multiplier = count(scheme["affine_multiplier"], "seed multiplier")
    require(modulus > 1 and multiplier > 0, "invalid affine seed parameters")
    exact(scheme["coprime_multiplier"], math.gcd(modulus, multiplier) == 1, "coprime multiplier")
    require(scheme["coprime_multiplier"], "seed multiplier must be coprime")
    seeds = method["selected_case_seeds"]
    require(isinstance(seeds, list) and len(seeds) == len(cases), "method seed case count")
    all_seeds = []
    screen_indices = summary["deterministic_selection"]["selected_screen_indices"]
    for case, row in zip(cases, seeds):
        for key in ("selected_index", "original_prompt_rank", "question_sha256"):
            exact(row[key], case[key], f"method seed identity.{key}")
        for key, length in (
            ("initial_seeds", method["initial_draws"]),
            ("fresh_seeds", method["fresh_draws"]),
        ):
            values = row[key]
            require(isinstance(values, list) and len(values) == length, f"method.{key}: length")
            for value in values:
                require(count(value, key) < modulus, f"method.{key}: seed out of range")
            all_seeds.extend(values)
            require(
                all((b - a) % modulus == multiplier % modulus for a, b in zip(values, values[1:])),
                f"method.{key}: affine seed spacing",
            )
    require(
        len(all_seeds) == len(set(all_seeds)), "method: duplicate selected-case generation seed"
    )
    # The public record documents original screen-index semantics and lists
    # affine sequences. Check their relative cross-case strides, anchored to the
    # first recorded seed in each phase. The absolute seed-origin formula and
    # interpretation of the phase offset are not published, so are not invented.
    for key, draws in (
        ("initial_seeds", method["initial_draws"]),
        ("fresh_seeds", method["fresh_draws"]),
    ):
        anchor = seeds[0][key][0]
        for row, screen_index in zip(seeds, screen_indices):
            expected = (anchor + multiplier * draws * (screen_index - screen_indices[0])) % modulus
            exact(row[key][0], expected, f"method.{key}: cross-case screen-index stride")
    exact(
        summary["deterministic_selection"]["eligible_screened_total"],
        method["screen_prompts"],
        "method screening count",
    )


def _check(root):
    ledger = read_json(root / LEDGER)
    summary = read_json(root / SUMMARY)
    method = read_json(root / METHOD)
    bindings = read_json(root / BINDINGS)
    provenance = read_json(root / PROVENANCE)
    exact(ledger["schema"], "c1-publication-case-ledger-v1", "ledger.schema")
    exact(summary["schema"], "public-scientific-result-summary-v1", "summary.schema")
    cases, events = ledger["cases"], ledger["events"]
    require(
        isinstance(cases, list) and isinstance(events, list), "ledger requires case/event lists"
    )
    selection = summary["deterministic_selection"]
    eligible = count(selection["eligible"], "eligible")
    selected = count(selection["selected"], "selected")
    screened = count(selection["eligible_screened_total"], "screened")
    require(0 < eligible <= screened == 4096, "screening design/count mismatch")
    require(selected == min(64, eligible) == len(cases), "selected cohort length mismatch")
    screen_indices = selection["selected_screen_indices"]
    require(isinstance(screen_indices, list), "screen indices require a list")
    for value in screen_indices:
        require(count(value, "screen index") < screened, "screen index out of range")
    require(screen_indices == sorted(set(screen_indices)), "screen indices duplicate or reordered")
    exact(
        selection["selected_original_ranks"],
        [value + 640 for value in screen_indices],
        "screen rank offset",
    )
    require(len(screen_indices) == selected, "selected screen indices count mismatch")
    for index, case in enumerate(cases):
        exact(case["selected_index"], index, "ordered case index")
        exact(
            case["original_prompt_rank"], selection["selected_original_ranks"][index], "case rank"
        )
        for flag in CASE_FLAGS:
            require(type(case[flag]) is bool, f"case {index}.{flag}: expected boolean")
        for key in ("question", "recorded_gold_reasoning"):
            exact(case[f"{key}_sha256"], sha(text(case[key], key)), f"case {index}.{key} hash")
        exact(case["prompt_id"], sha(" ".join(case["question"].split())), f"case {index}.prompt_id")
        digest(case["audit_id"], f"case {index}.audit_id")
        canonical_number(case["recorded_gold_value"], "recorded gold value")
        text(case["reconciliation_note"], "reconciliation note")
        category = case["second_exclusive_category"]
        require(category in CATEGORIES, f"case {index}: unknown second category")
        require(
            case["first_gold_review"] in {"supported", "disputed", "ambiguous"},
            f"case {index}: unknown first category",
        )
        exact(
            case["first_second_category_disagreement"],
            case["first_gold_review"] != CATEGORY_FIRST[category],
            f"case {index}.disagreement",
        )
        require(
            not case["conservative_corroborated_numeric_recovery"] or case["frozen_primary_event"],
            f"case {index}: conservative witness is not a primary event",
        )
    for key in ("audit_id", "prompt_id", "question_sha256"):
        require(len({case[key] for case in cases}) == selected, f"duplicate case {key}")
    event_indices = [case["selected_index"] for case in cases if case["frozen_primary_event"]]
    exact([event["selected_index"] for event in events], event_indices, "ordered event identities")
    responses, screen_responses, follow_responses = [], [], []
    partitions = Counter()
    for event in events:
        index = event["selected_index"]
        case = cases[index]
        for key in set(case) & set(event):
            exact(event[key], case[key], f"event {index}.{key}")
        kind = event["event_evidence_classification"]
        require(kind in EVENT_PARTITIONS, f"event {index}: unknown classification")
        partitions[EVENT_PARTITIONS[kind]] += 1
        success = event["frozen_primary_success_indices"]
        require(
            isinstance(success, list) and bool(success), f"event {index}: missing fresh successes"
        )
        for value in success:
            require(count(value, "fresh sample index") < 24, "fresh sample index out of range")
        require(
            success == sorted(set(success)), f"event {index}: duplicate/reordered fresh indices"
        )
        rows = event["response_checks"]
        require(isinstance(rows, list), f"event {index}: response list required")
        expected = [("screen", value) for value in range(8)] + [
            ("followup", value) for value in success
        ]
        exact(
            [[row["phase"], row["sample_index"]] for row in rows],
            [list(x) for x in expected],
            f"event {index}: response coverage/order",
        )
        for row in rows:
            digest(row["text_raw_sha256"], "response content hash")
            exact(row["cap_hit"], False, "reviewed response cap flag")
            exact(row["extraction_fidelity_rechecked"], "faithful", "extraction fidelity")
            reward = int(row["phase"] == "followup")
            exact(row["frozen_reward"], reward, "response frozen reward")
            canonical_number(row["parsed_value"], "parsed numeric value")
            require(
                isinstance(row["terminal_numeric_literal"], str), "numeric literal must be a string"
            )
            require(
                normalize_number(row["terminal_numeric_literal"]) == row["parsed_value"],
                "stored terminal literal and parsed value disagree",
            )
            exact(
                row["parsed_value"] == case["recorded_gold_value"],
                bool(reward),
                "frozen exact match",
            )
            require(
                row["reconciled_numeric_status"] in {"correct", "ambiguous", "wrong"},
                "unknown reconciled numeric status",
            )
            text(row["visible_reasoning_status"], "visible reasoning status")
            text(row["reconciler_reasoning_caveat"], "reasoning caveat")
        initial = rows[:8]
        fresh = rows[8:]
        corroborated = (
            case["first_gold_review"] == "supported"
            and CATEGORY_FIRST[case["second_exclusive_category"]] == "supported"
            and all(row["reconciled_numeric_status"] == "wrong" for row in initial)
            and any(row["reconciled_numeric_status"] == "correct" for row in fresh)
        )
        exact(
            event["conservative_corroborated_numeric_recovery"], corroborated, "witness review rule"
        )
        exact(kind == "corroborated_numeric_recovery", corroborated, "witness classification")
        responses.extend(rows)
        screen_responses.extend(initial)
        follow_responses.extend(fresh)
    check_interval(
        summary["frozen_primary"], len(events), len(cases), "frozen primary", wilson=True
    )
    exact(summary["frozen_primary"]["unchanged"], True, "frozen primary unchanged")
    check_interval(
        summary["frozen_eligible_screen"], eligible, screened, "eligible screen", wilson=True
    )
    witness_indices = [
        case["selected_index"]
        for case in cases
        if case["conservative_corroborated_numeric_recovery"]
    ]
    witness = summary["conservative_same_denominator_support"]
    check_interval(witness, len(witness_indices), selected, "conservative support")
    exact(
        [row["selected_index"] for row in witness["cases"]],
        witness_indices,
        "conservative case set",
    )
    exact(
        witness["not_corroborated_parser_event_count"],
        len(events) - len(witness_indices),
        "uncorroborated event count",
    )
    count(witness["pending_annotation_cases_after_addendum"], "pending annotation count")
    exact(
        witness["not_corroborated_event_partition"],
        {
            "first_second_convention_or_wording_disagreement": partitions[
                "Convention / disagreement"
            ],
            "material_ambiguity_or_underdetermination": partitions["Ambiguous"],
            "clear_wrong_gold_match": partitions["Wrong reference"],
        },
        "uncorroborated event partition",
    )
    exact(
        [
            row["selected_index"]
            for row in summary["ordinary_reading_event_support_not_reestimated"]["cases"]
        ],
        [
            event["selected_index"]
            for event in events
            if EVENT_PARTITIONS[event["event_evidence_classification"]]
            == "Convention / disagreement"
        ],
        "ordinary-reading event set",
    )
    exact(
        summary["first_review_gold_counts"],
        dict(Counter(case["first_gold_review"] for case in cases)),
        "first review counts",
    )
    refined = summary["second_review_refined_mutually_exclusive_counts"]
    exact(
        refined,
        dict(Counter(case["second_exclusive_category"] for case in cases)),
        "refined counts",
    )
    original = summary["second_review_original_mutually_exclusive_counts"]
    require(
        set(original) == set(CATEGORIES) - {"correct_final_value_with_flawed_reference_reasoning"},
        "original review count fields",
    )
    require(
        sum(count(value, "original count") for value in original.values()) == selected,
        "original review count total",
    )
    for key in ("clear_wrong_gold", "materially_ambiguous", "inconsistent_question"):
        exact(original[key], refined[key], f"refinement preserves {key}")
    require(
        all(
            original[key] >= refined[key]
            for key in ("supported", "supported_with_conventional_assumptions")
        ),
        "refinement increases a supported group",
    )
    disagreement = summary["first_second_category_disagreements"]
    exact(
        [row["selected_index"] for row in disagreement["cases"]],
        [case["selected_index"] for case in cases if case["first_second_category_disagreement"]],
        "disagreement cases",
    )
    check_summary_references(summary, cases)
    defects = summary["conservative_corroborated_defects"]
    members = []
    for key in (
        "wrong_final_value_under_stated_or_ordinary_model",
        "inconsistent_discrete_premises_with_reference_error",
        "correct_final_with_corroborated_reference_explanation_defect",
        "materially_ambiguous_with_explicit_reference_reasoning_defect",
    ):
        members.extend(row["selected_index"] for row in defects[key]["cases"])
    require(len(members) == len(set(members)), "corroborated defect groups overlap")
    exact(
        [row["selected_index"] for row in defects["explicit_reference_defect_union"]["cases"]],
        sorted(members),
        "defect union",
    )
    output = summary["event_output_review"]
    for key, expected in {
        "events": len(events),
        "initial_raw_responses": len(screen_responses),
        "fresh_primary_success_raw_responses": len(follow_responses),
        "total_raw_responses": len(responses),
        "all_extractions_faithful": True,
        "all_uncapped": True,
    }.items():
        exact(output[key], expected, f"response summary.{key}")
    exact(
        output["first_numeric_status_on_38_matches"],
        dict(Counter(row["reconciled_numeric_status"] for row in follow_responses)),
        "numeric status totals",
    )
    caveat = output["numeric_correct_does_not_mean_reasoning_valid"]
    count(caveat["case_selected_index"], "reasoning caveat case index")
    count(caveat["fresh_sample_index"], "reasoning caveat sample index")
    matches = [
        row
        for event in events
        if event["selected_index"] == caveat["case_selected_index"]
        for row in event["response_checks"]
        if row["phase"] == "followup" and row["sample_index"] == caveat["fresh_sample_index"]
    ]
    require(len(matches) == 1, "reasoning caveat references missing response")
    exact(
        caveat["text_raw_sha256"], matches[0]["text_raw_sha256"], "reasoning caveat content binding"
    )
    exact(matches[0]["reconciled_numeric_status"], "correct", "reasoning caveat correct number")
    exact(
        matches[0]["visible_reasoning_status"],
        "correct_final_number_with_invalid_visible_reasoning",
        "reasoning caveat preserved",
    )
    exact(read_json(root / f"{REVISION}/thesis/c1_case_ledger.json"), ledger, "thesis ledger copy")
    addendum = read_json(root / f"{REVISION}/addendum/case_ledger_public.json")
    exact(addendum["schema"], "c1-public-scientific-case-ledger-v1", "addendum.schema")
    for key in ("cases", "events"):
        expected = [
            {k: v for k, v in row.items() if k not in {"audit_id", "prompt_id"}}
            for row in ledger[key]
        ]
        exact(addendum[key], expected, f"addendum.{key} projection")
    check_csv(root / f"{RESULTS}/c1_case_ledger.csv", cases, CSV_FIELDS)
    check_csv(root / f"{REVISION}/thesis/c1_case_ledger.csv", cases, CSV_FIELDS)
    check_csv(root / f"{REVISION}/addendum/case_ledger_public.csv", cases, ADDENDUM_CSV_FIELDS)
    check_method(method, cases, summary, provenance)
    exact(bindings["schema"], "public-c1-reference-bindings-v1", "reference bindings schema")
    sources = {}
    for source in bindings["sources"]:
        require(source["id"] not in sources, "duplicate source binding")
        sources[source["id"]] = source
        digest(source["sha256"], "reference source digest")
        count(source["bytes"], "reference source byte count")
    exact(
        summary["source_summary_sha256"], sources["S2"]["sha256"], "review summary source binding"
    )
    require(
        all(source in sources for source in ledger["source_ids"]),
        "unresolved ledger source reference",
    )
    check_documents(root, cases, events, method)
    return {
        "status": "PASS",
        "selected_cases": selected,
        "frozen_primary_events": len(events),
        "conservative_numeric_witnesses": len(witness_indices),
        "eligible_screened": eligible,
        "screened_questions": screened,
        "reviewed_response_records": len(responses),
        "document_tables_checked": len(DOCUMENTS) * 2,
        "limitation": LIMITATION,
    }


def check_public_results(root: Path) -> dict:
    """Validate the release's public scientific files; raise ValueError on any mismatch."""
    try:
        return _check(Path(root))
    except (OSError, KeyError, IndexError, TypeError, csv.Error, ArithmeticError) as error:
        raise ValueError(f"Malformed or missing public scientific record: {error}") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        result = check_public_results(args.root)
    except ValueError as error:
        print(
            json.dumps(
                {"status": "FAIL", "error": str(error), "scope": LIMITATION},
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
