"""Public derivative: extracted scientific functions; operational code removed.

Selected definitions are unchanged from the retained source identified in
PROVENANCE.json. This module alone cannot authenticate empirical evidence.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math

def require(condition, message):
    if not condition:
        raise ValueError(message)


def wilson(successes, total):
    require(type(successes) is int and type(total) is int and 0 <= successes <= total, "Invalid binomial counts")
    if not total:
        return None
    z = 1.959963984540054
    den = total + z*z
    center = (successes + z*z/2) / den
    half = z * math.sqrt(successes * (1-successes/total) + z*z/4) / den
    return [max(0., center-half), min(1., center+half)]


def estimate(events, total):
    return {"events": events, "denominator": total, "fraction": events/total if total else None,
            "wilson_95": wilson(events, total),
            "zero_events_exact_upper_one_sided_95": -math.expm1(math.log(.05)/total) if total and not events else None}


def prediction(initial_correct, prior):
    require(initial_correct in (0, 8), "Prediction needs initially homogeneous group")
    # Symmetric prior gives the same unchanged-event probability in either stratum.
    unchanged = math.prod((8 + prior + j) / (8 + 2*prior + j) for j in range(24))
    return 1 - unchanged


def quality(rows):
    completions = [c for row in rows for c in row["completions"]]
    n = len(completions)
    invalid = sum(c["parse_status"] != "ok" for c in completions)
    caps = sum(c["cap_hit"] for c in completions)
    return {"prompts": len(rows), "completions": n, "parse_failures": invalid, "cap_hits": caps,
            "parse_failure_rate": invalid/n if n else None, "cap_hit_rate": caps/n if n else None,
            "passes_smoke_quality_gate": bool(n and invalid/n <= .05 and caps/n <= .05)}


def analyze_validated(rows):
    """Arithmetic only; call validate_run before treating output as evidence."""
    per_prompt, audit = [], []
    event_counts = Counter({"first_change_parser_failure": 0, "first_change_cap_hit": 0,
                            "first_change_parseable_wrong": 0, "first_change_parseable_correct": 0,
                            "initial8_any_parser_failure": 0, "initial8_any_cap_hit": 0,
                            "initial8_and_first_change_all_parseable_uncapped": 0})
    for row in rows:
        cs = row["completions"]
        rewards = [c["reward"] for c in cs]
        k8 = sum(rewards[:8])
        hom = k8 in (0, 8)
        mixed = {n: 0 < sum(rewards[:n]) < n for n in (8, 16, 32)}
        rescued = hom and mixed[32]
        clean = all(c["parse_status"] == "ok" and not c["cap_hit"] for c in cs)
        p = {"prompt_id": row["prompt_id"], "dataset_index": row["dataset_index"],
             "initial_correct_8": k8, "combined_correct_32": sum(rewards),
             "initial_homogeneous": int(hom), "primary_event": int(rescued),
             "conditional_event_g16": int(hom and mixed[16]),
             "mixed_g8": int(mixed[8]), "mixed_g16": int(mixed[16]), "mixed_g32": int(mixed[32]),
             "fresh16_mixed": int(0 < sum(rewards[8:24]) < 16),
             "all32_parseable_uncapped": int(clean)}
        per_prompt.append(p)
        if rescued:
            first = next(c for c in cs[8:] if c["reward"] != rewards[0])
            first_invalid = first["parse_status"] != "ok"
            initial_invalid = any(c["parse_status"] != "ok" for c in cs[:8])
            initial_cap = any(c["cap_hit"] for c in cs[:8])
            flags = {"first_change_parser_failure": first_invalid, "first_change_cap_hit": first["cap_hit"],
                     "first_change_parseable_wrong": not first_invalid and first["reward"] == 0,
                     "first_change_parseable_correct": not first_invalid and first["reward"] == 1,
                     "initial8_any_parser_failure": initial_invalid, "initial8_any_cap_hit": initial_cap,
                     "initial8_and_first_change_all_parseable_uncapped": not (first_invalid or first["cap_hit"] or initial_invalid or initial_cap)}
            event_counts.update({key: int(value) for key, value in flags.items()})
            audit.append({"prompt_id": row["prompt_id"], "dataset_index": row["dataset_index"],
                          "question": row["question"], "gold_answer": row["gold_answer"], "gold_value": row["gold_value"],
                          "initial_reward": rewards[0], "initial_eight": cs[:8], "first_changing_response": first,
                          "automated_event_flags": flags, "manual_review_status": "PENDING",
                          "machine_eligible_conservative_witness": flags["initial8_and_first_change_all_parseable_uncapped"],
                          "changing_sample_index": first["sample_index"],
                          "reviewed_text_sha256": {str(c["sample_index"]): hashlib.sha256(c["text"].encode()).hexdigest() for c in cs[:8]+[first]},
                          "gold_review": None, "initial_numeric_extraction": None, "changing_numeric_extraction": None,
                          "initial_or_changing_cap_hit": initial_cap or first["cap_hit"],
                          "correctness_flip_supported": None, "qualifies_conservative_witness": None,
                          "exclusion_reasons": [], "reviewed_at_utc": None,
                          "manual_extracted_initial_values": [None]*8, "manual_extracted_changing_value": None,
                          "manual_classification": None, "gold_disputed": None, "reviewer": None,
                          "review_note": None,
                          "instruction": "Inspect all initial eight plus first changing response; classify numeric contrast, format-only contrast, cap-involved contrast, or ambiguous/gold-disputed. Never rewrite raw rewards."})
    hom = [p for p in per_prompt if p["initial_homogeneous"]]
    subgroup = lambda ps: estimate(sum(p["primary_event"] for p in ps), len(ps))
    priors = {}
    for name, prior in (("beta_1_1", 1.), ("beta_half_half", .5)):
        q = prediction(0, prior)
        priors[name] = {"predicted_probability": q,
                        "brier": sum((q-p["primary_event"])**2 for p in hom)/len(hom) if hom else None,
                        "calibration_error_predicted_minus_observed": q-sum(p["primary_event"] for p in hom)/len(hom) if hom else None}
    priors["plugin_zero"] = {"predicted_probability": 0., "brier": sum(p["primary_event"] for p in hom)/len(hom) if hom else None}
    n = len(rows)
    phase_quality = quality(rows)
    summary = {"schema_version": 2, "primary": subgroup(hom),
               "all_wrong_descriptive": subgroup([p for p in hom if p["initial_correct_8"] == 0]),
               "all_correct_descriptive": subgroup([p for p in hom if p["initial_correct_8"] == 8]),
               "mixedness_all_prompts": {str(g): estimate(sum(p[f"mixed_g{g}"] for p in per_prompt), n) for g in (8,16,32)},
               "conditional_g16_descriptive": estimate(sum(p["conditional_event_g16"] for p in hom), len(hom)),
               "fresh16_all_prompts_descriptive": estimate(sum(p["fresh16_mixed"] for p in per_prompt), n),
               "fresh16_initially_homogeneous_descriptive": estimate(sum(p["fresh16_mixed"] for p in hom), len(hom)),
               "parseable_uncapped_sensitivity": {**subgroup([p for p in hom if p["all32_parseable_uncapped"]]),
                   "warning": "Post-outcome selected subgroup; not an unbiased alternative or replacement primary"},
               "event_type_sensitivity_nonexclusive": dict(event_counts), "fixed_prior_predictions": priors,
               "quality": phase_quality, "main_quality_breach": not phase_quality["passes_smoke_quality_gate"],
               "generated_tokens": sum(c["output_token_count"] for row in rows for c in row["completions"]),
               "input_tokens_counting_every_draw": sum(c["input_token_count"] for row in rows for c in row["completions"]),
               "precision_status": "AT_LEAST_100_HOMOGENEOUS" if len(hom) >= 100 else "PRECISION_LIMITED",
               "manual_review_required": len(audit), "manual_review_complete": not audit,
               "conservative_numeric_witnesses": {"verified_count": None if audit else 0, "original_primary_denominator":len(hom),
                   "machine_eligible_pending_manual_review":event_counts["initial8_and_first_change_all_parseable_uncapped"],
                   "status":"PENDING_MANUAL_REVIEW" if audit else "NO_PRIMARY_EVENTS"},
               "scope": "Fixed policy/task/decoder observed reward variation; no training or algorithm-superiority inference",
               "interval_caveat": "Binomial working model across prompts; no guaranteed population coverage under arbitrary dependence"}
    return summary, per_prompt, audit
