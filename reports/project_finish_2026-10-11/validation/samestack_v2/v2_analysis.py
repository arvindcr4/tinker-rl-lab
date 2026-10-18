"""Independent exact grading audit and canonical seed analysis.

Importing has no CLI/filesystem side effects. CLI (legacy directory argument is
accepted): python v2_analysis.py [scripts_dir] DATA GOLD [--plan PLAN.json].
Output is strict JSON; this does not modify stored results or historical audits.
"""
import argparse
import hashlib
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
from pathlib import Path
import re

GRADING_SPEC_ID = "boxed-exact-rational-v2"


def last_boxed(text):
    """Independently scan backwards for the last complete exact boxed command."""
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    for match in reversed(starts):
        depth = 1
        for end in range(match.end(), len(text)):
            depth += (text[end] == "{") - (text[end] == "}")
            if depth == 0:
                return text[match.end():end]
    return None


def to_num(value):
    """Exact GSM8K coefficient; optional one full-string numeric decoration.

    Currency prefix/wrapper, percent suffix (coefficient, not /100), or the
    documented second/seconds suffix are allowed. No decorations within
    fraction operands, broad unit stripping, float tolerance or scavenging.
    """
    if not isinstance(value, str) or len(value) > 4096:
        return None
    value = re.sub(r"\\[!,;: ]", "", value).strip()
    decorations = []
    if len(value) >= 2 and value[0] == value[-1] == "$":
        value = value[1:-1].strip()
        decorations.append("currency")
    else:
        currency = re.match(r"(?:\\\$|\$)", value)
        if currency:
            value = value[currency.end():].strip()
            decorations.append("currency")
    suffix = re.search(r"(?:\\%|%)\s*$", value)
    if suffix:
        value = value[:suffix.start()].strip()
        decorations.append("percent")
    suffix = re.search(r"(?:\\text\{\s*seconds?\s*\}|\s+seconds?)\s*$", value)
    if suffix:
        value = value[:suffix.start()].strip()
        decorations.append("seconds")
    return _to_rational(value) if len(decorations) <= 1 else None


def _to_rational(value):
    """Parse the entire undecorated rational recursively, without float math."""
    fraction = re.fullmatch(r"([+-]?)\\(?:frac|dfrac|tfrac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}", value)
    if fraction:
        numerator, denominator = _to_rational(fraction[2].strip()), _to_rational(fraction[3].strip())
        if numerator is None or denominator is None or denominator == 0:
            return None
        return (-1 if fraction[1] == "-" else 1) * numerator / denominator
    if value.count("/") == 1:
        numerator, denominator = (_to_rational(part.strip()) for part in value.split("/"))
        return None if numerator is None or denominator in (None, 0) else numerator / denominator
    # Grouping belongs only to an atom's integer portion, never to a substring
    # of its decimal digits or exponent. Fraction operands are checked above.
    if re.fullmatch(r"[+-]?(?:(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value) is None:
        return None
    try:
        number = Decimal(value.replace(",", ""))
        return Fraction(number) if number.is_finite() and abs(number.adjusted()) <= 10000 else None
    except (InvalidOperation, ValueError, OverflowError, ZeroDivisionError):
        return None


def grade(text, gold):
    boxed = last_boxed(text)
    if boxed is None:
        return 0, "nobox"
    predicted, expected = to_num(boxed), to_num(gold)
    if predicted is None:
        return 0, "unparsable"
    if expected is None:
        raise ValueError("gold answer must be a finite exact number")
    return int(predicted == expected), "ok"


def _reject_constant(value):
    raise ValueError(f"nonstandard JSON numeric token: {value}")


def load_json(path):
    from platform_hybrid.experiments.samestack_contract import strict_json_load
    return strict_json_load(path)


def load_inputs(data_dir, gold_path):
    """Explicit I/O only; never invoked at import time."""
    test = load_json(gold_path)["test"]
    gold = [item["answer"].split("####")[-1].strip() for item in test]
    gold_ids = [f"test:{index}:" + hashlib.sha256(json.dumps({"question": item["question"], "answer": item["answer"]},
                sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
                for index, item in enumerate(test)]
    records, completions, completion_validated_ids = [], [], []
    for path in sorted(Path(data_dir).rglob("*_s*.json")):
        if re.fullmatch(r"(?:grpo_g8|grpo_g2|ppo|ppo_8x8)_s\d+\.json", path.name) is None:
            if path.name.endswith(".complete.json"):
                continue
            raise ValueError(f"Unexpected run filename: {path}")
        record = load_json(path)
        if record.get("manifest"):
            from platform_hybrid.experiments.samestack_contract import validate_completion
            try:
                record = validate_completion(path)
            except (OSError, ValueError, TypeError, KeyError) as error:
                raise ValueError(f"Manifested input has no valid completion bundle: {path}: {error}") from error
            completion_validated_ids.append(record["manifest"]["run_id"])
        record["source_path"] = str(path)
        records.append(record)
        completions.append(load_json(path.with_name(path.stem + "_texts.json.gz")))
    return {"records": records, "completions": completions, "gold": gold, "gold_item_ids": gold_ids, "completion_validated_run_ids": completion_validated_ids}


def validate_inputs(inputs):
    gold, records, completions = inputs["gold"], inputs["records"], inputs["completions"]
    if not gold or any(to_num(answer) is None for answer in gold):
        raise ValueError("all gold answers must be finite exact numbers")
    if len(records) != len(completions):
        raise ValueError("every run must have exactly one completion artifact")
    for record, texts in zip(records, completions):
        if record.get("n_eval") != len(gold):
            raise ValueError("n_eval must equal the ordered gold length")
        ids = (record.get("manifest") or {}).get("test_item_ids")
        if ids is not None:
            if inputs.get("gold_item_ids") != ids:
                raise ValueError("ordered gold item IDs do not match run manifest")
            if texts.get("test_item_ids") != ids:
                raise ValueError("ordered completion item IDs do not match run manifest")
        for stage in ("pre", "post"):
            if (not isinstance(texts.get(stage), list) or len(texts[stage]) != len(gold)
                    or any(not isinstance(text, str) for text in texts[stage])):
                raise ValueError(f"{stage} texts must match ordered gold length")
            values = record.get(stage + "_correct")
            if not isinstance(values, list) or len(values) != len(gold):
                raise ValueError(f"{stage} verdicts must match ordered gold length")
    return inputs


def _modules():
    try:
        from . import analysis_plan, stats_lib
    except ImportError:
        import analysis_plan
        import stats_lib
    return analysis_plan, stats_lib


def item_diagnostics(records, analysis, B=4000):
    """Nonprimary item models: conditional on seeds or a two-way sensitivity."""
    import numpy as np
    from scipy.stats import wilcoxon
    _, statistics = _modules()
    if type(B) is not int or B < 1:
        raise ValueError("B must be a positive integer")
    by_id = {(record.get("manifest") or {}).get("run_id"): record for record in records}
    rng, output = np.random.default_rng(1), {}
    contrasts = {**analysis["primary_contrasts"], **analysis["exploratory_contrasts"]}
    for key, contrast in contrasts.items():
        if contrast["status"] != "available":
            continue
        diffs = np.array([np.array(by_id[pair["a_run_id"]]["post_correct"]) -
                          np.array(by_id[pair["b_run_id"]]["post_correct"]) for pair in contrast["pairs"]])
        n_seeds, n_items = diffs.shape
        b, c = int((diffs == 1).sum()), int((diffs == -1).sum())
        two_way = [diffs[np.ix_(rng.integers(n_seeds, size=n_seeds), rng.integers(n_items, size=n_items))].mean()
                   for _ in range(B)]
        items = [diffs[:, rng.integers(n_items, size=n_items)].mean() for _ in range(B)]
        item_means = diffs.mean(axis=0)
        nonzero = item_means[item_means != 0]
        output[key] = {"primary": False, "n_seeds": n_seeds, "n_items": n_items,
                       "mean_diff": float(diffs.mean()), "pooled_x_only": b, "pooled_y_only": c,
                       "pooled_mcnemar_naive_p": statistics.mcnemar_exact(b, c),
                       "pooled_mcnemar_caveat": "Repeated seed-item observations are dependent; pooled p is not primary inference",
                       "per_seed_mcnemar_p": [statistics.mcnemar_exact(int((row == 1).sum()), int((row == -1).sum())) for row in diffs],
                       "twoway_boot_ci": np.percentile(two_way, [2.5, 97.5]).tolist(),
                       "items_only_boot_ci": np.percentile(items, [2.5, 97.5]).tolist(),
                       "items_only_scope": "Conditional on observed training seeds; no training-run uncertainty",
                       "wilcoxon_items_seedavg_p": float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0}
    return output


def analyze(inputs, plan=None, bootstrap_samples=4000):
    """Pure analysis of materialized inputs; original verdicts are never replaced."""
    validate_inputs(inputs)
    plans, statistics = _modules()
    analysis = plans.analyze_runs(inputs["records"], plan,
                                  completion_validated_run_ids=inputs.get("completion_validated_run_ids", ()))
    audits = []
    for record, texts in zip(inputs["records"], inputs["completions"]):
        audit = {"arm": record["arm"], "seed": record["seed"], "run_id": (record.get("manifest") or {}).get("run_id"),
                 "grading_spec": GRADING_SPEC_ID, "n_items": len(inputs["gold"])}
        for stage in ("pre", "post"):
            scores = [grade(text, gold)[0] for text, gold in zip(texts[stage], inputs["gold"])]
            audit[stage + "_accuracy"] = sum(scores) / len(scores)
            audit[stage + "_disagreement_indices"] = [i for i, (score, stored) in enumerate(zip(scores, record[stage + "_correct"])) if score != stored]
            audit[stage + "_nobox"] = sum(last_boxed(text) is None for text in texts[stage]) / len(scores)
        audits.append(audit)
    power = {}
    for key, contrast in analysis["primary_contrasts"].items():
        if contrast["status"] != "available" or contrast["sd"] <= 0:
            power[key] = {"available": False, "reason": "requires at least two pairs with positive pilot SD"}
            continue
        sd = contrast["sd"]
        power[key] = {"available": True, "scope": "pilot unadjusted paired-t sensitivity; not family or TOST design power",
                      "sd_diff": sd, "mde_n7": statistics.mde(sd, 7), "mde_n10": statistics.mde(sd, 10),
                      "seeds_for_0.05": statistics.n_needed(sd, .05), "seeds_for_0.01": statistics.n_needed(sd, .01)}
    return {"analysis": analysis, "independent_grading_audit": audits,
            "item_level": item_diagnostics(inputs["records"], analysis, B=bootstrap_samples), "power": power,
            "actual_design_power": {"status": "unmet", "reason": "Requires justified joint seed/item variance, correlation, effect and equivalence-margin assumptions; pilot t sensitivity is not a substitute"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="DATA GOLD, or legacy SCRIPTS DATA GOLD")
    parser.add_argument("--plan", help="Explicit canonical plan JSON; omitted means unbound interim analysis")
    args = parser.parse_args(argv)
    if len(args.paths) not in (2, 3):
        parser.error("provide DATA GOLD, optionally preceded by legacy SCRIPTS")
    data_dir, gold_path = args.paths[-2:]
    plan = load_json(args.plan) if args.plan else None
    print(json.dumps(analyze(load_inputs(data_dir, gold_path), plan), indent=2, allow_nan=False))


if __name__ == "__main__":
    import sys
    # Isolated Python omits the script directory. Only CLI startup changes path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
