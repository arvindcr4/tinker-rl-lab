"""Candidate v3 strict publication audit; historical JSON is never overwritten.

Usage: python -I v2_analysis.py SCRIPTS DATA GOLD --protocol PLAN.json
The explicit plan freezes revisions, evaluation identities, seeds and input hashes.
Legacy exports without these bindings fail closed rather than gaining verification.
"""
import argparse
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import gzip, hashlib, json, math, platform, re, sys
import scipy
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_lib import paired, holm, mde, n_needed, mcnemar_exact

ANALYSIS_VERSION = "samestack-publication-v3-candidate"
GRADER_VERSION = "independent-boxed-rational-v3-candidate"
PRIMARY = ["g8-ppo", "g8-ppo_8x8", "g8-g2"]

def require(ok, message):
    if not ok: raise ValueError(message)

def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def load(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    raw = Path(path).read_bytes()
    if str(path).endswith(".gz"): raw = gzip.decompress(raw)
    def finite_float(value):
        number = float(value)
        require(math.isfinite(number), "nonfinite JSON number")
        return number
    return json.loads(raw, object_pairs_hook=pairs, parse_float=finite_float, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))

def validated_inputs(data, gold_path, protocol):
    gold_doc = load(gold_path)
    require(protocol.get("schema_version") == "samestack-publication-protocol-v3-candidate", "explicit candidate protocol required")
    require(protocol.get("primary_contrasts") == PRIMARY, "frozen primary family required")
    seeds = protocol.get("expected_seeds")
    require(isinstance(seeds, list) and len(seeds) >= 2 and all(type(s) is int for s in seeds) and sorted(set(seeds)) == seeds, "ordered unique expected seeds required")
    require(protocol.get("endpoint") == "post_accuracy" and protocol.get("stopping_rule") == "fixed_expected_seeds", "endpoint/stopping rule required")
    require(protocol.get("equivalence_margin") == .02, "candidate margin is a sensitivity parameter fixed at .02")
    dataset = protocol.get("dataset", {})
    gold_sha = hashlib.sha256(Path(gold_path).read_bytes()).hexdigest()
    require(dataset.get("sha256") == gold_sha and isinstance(dataset.get("revision"), str) and bool(dataset["revision"]), "dataset hash/revision binding required")
    items = gold_doc["test"]
    ids = [item.get("id", "test:" + str(i) + ":" + object_hash({"question": item["question"], "answer": item["answer"]})) for i, item in enumerate(items)]
    require(ids and len(ids) == len(set(ids)), "unique ordered gold item IDs required")
    gold = [item["answer"].split("####")[-1].strip() for item in items]
    require(all(to_num(g) is not None for g in gold), "invalid gold number")
    environment = protocol.get("environment", {})
    require(isinstance(environment, dict) and all(isinstance(environment.get(field), str) and bool(environment[field].strip())
        for field in ("provider", "hardware", "software", "precision", "checkpoint_revision", "tokenizer_revision", "evaluation_protocol")),
        "complete environment provenance fields required")
    require(protocol.get("environment_fingerprint") == object_hash(environment), "environment descriptor fingerprint mismatch")
    selected = protocol.get("selected_runs")
    require(isinstance(selected, list) and selected, "selected-run manifest required")
    selected_by_file = {r["file"]: r for r in selected}
    require(len(selected_by_file) == len(selected), "duplicate selected file")
    found = sorted(Path(data).glob("*_s*.json"))
    require({p.name for p in found} == set(selected_by_file), "unselected or missing run files")
    seen, run_ids, attempts, validated = set(), set(), set(), []
    for path in found:
        r = load(path); tx_path = path.with_name(path.stem + "_texts.json.gz"); tx = load(tx_path)
        key = (r["arm"], r["seed"])
        require(key not in seen, "Duplicate experiment key: " + str(key)); seen.add(key)
        require(type(r["seed"]) is int and r["seed"] in seeds and r["arm"] in ("ppo", "ppo_8x8", "grpo_g8", "grpo_g2"), "unexpected arm/seed")
        choice = selected_by_file[path.name]
        source = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), texts_sha256=hashlib.sha256(tx_path.read_bytes()).hexdigest())
        for field in ("run_id", "attempt_id", "protocol_version", "environment_fingerprint"):
            require(isinstance(r.get(field), str) and bool(r[field]), "missing provenance: " + field)
            source[field] = r[field]
            require(choice.get(field) == r[field], "selected-run provenance mismatch: " + field)
        require(r["environment_fingerprint"] == protocol.get("environment_fingerprint") and r["protocol_version"] == protocol.get("run_protocol_version"), "same-stack environment/protocol mismatch")
        require(r["run_id"] not in run_ids and r["attempt_id"] not in attempts, "duplicate run/attempt identity")
        run_ids.add(r["run_id"]); attempts.add(r["attempt_id"])
        require(choice.get("sha256") == source["sha256"] and choice.get("texts_sha256") == source["texts_sha256"], "selected-run hash mismatch")
        require(r.get("dataset_revision") == dataset["revision"] and tx.get("dataset_revision") == dataset["revision"], "dataset revision mismatch")
        require(r.get("eval_item_ids") == tx.get("eval_item_ids") == ids, "Evaluation identity/order mismatch")
        require(r.get("n_eval") == len(ids), "n_eval mismatch")
        for phase in ("pre", "post"):
            require(isinstance(tx.get(phase), list) and len(tx[phase]) == len(ids) and all(isinstance(t, str) for t in tx[phase]), "Incomplete " + phase + " texts")
            for field in (phase + "_correct", phase + "_trunc"):
                values = r.get(field)
                require(isinstance(values, list) and len(values) == len(ids) and all(type(v) in (int, bool) and v in (0,1) for v in values), "Incomplete or nonbinary " + field)
        require(r.get("step_log") and all(all(type(s.get(k)) in (float, int) and math.isfinite(s[k]) for k in ("trunc_frac", "mean_comp_len")) for s in r["step_log"]), "finite training diagnostics required")
        require(all(type(r.get(k)) in (float,int) and math.isfinite(r[k]) and r[k] >= 0 for k in ("elapsed_seconds", "eval_seconds")), "finite timing required")
        validated.append((r, tx, source))
    return validated, gold, gold_sha

def last_boxed(text):
    """Independently scan backwards for the last complete exact boxed command."""
    if not isinstance(text, str):
        return None
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    for match in starts[-1:]:
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
    wrapper = re.fullmatch(r"\\(?:text|mathrm)\{([^{}]*)\}", value)
    if wrapper:
        value = wrapper[1].strip()
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
    if "," in value:
        if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d*)?(?:[eE][+-]?\d+)?", value) is None:
            return None
        value = value.replace(",", "")
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value) is None:
        return None
    try:
        number = Decimal(value)
        return Fraction(number) if number.is_finite() and abs(number.adjusted()) <= 10000 else None
    except (InvalidOperation, ValueError, OverflowError, ZeroDivisionError):
        return None


def grade(text, gold):
    boxed = last_boxed(text)
    if boxed is None:
        return 0, "unclosed_box" if isinstance(text, str) and re.search(r"\\boxed\s*\{", text) else "nobox"
    predicted, expected = to_num(boxed), to_num(gold)
    if predicted is None:
        return 0, "invalid_fraction" if "frac" in boxed or "/" in boxed else "unparsable"
    if expected is None:
        raise ValueError("gold answer must be a finite exact number")
    return int(predicted == expected), "ok"



def analyze(data, gold_path, protocol):
    validated_runs, gold, gold_sha = validated_inputs(data, gold_path, protocol)
    runs = {}
    for r, tx, source in validated_runs:
        f = source["file"]
        mine = {k: np.array([grade(t, g)[0] for t, g in zip(tx[k], gold)]) for k in ("pre", "post")}
        nobox = {k: np.mean([last_boxed(t) is None for t in tx[k]]) for k in ("pre", "post")}
        runs.setdefault(r["arm"], {})[r["seed"]] = dict(
            stored_post=np.array(r["post_correct"], dtype=int), stored_pre=np.array(r["pre_correct"], dtype=int),
            mine_post=mine["post"], mine_pre=mine["pre"], nobox=nobox, source=source,
            post_trunc=np.mean(r["post_trunc"]), pre_trunc=np.mean(r["pre_trunc"]),
            train_trunc=np.mean([s["trunc_frac"] for s in r["step_log"]]),
            train_len=np.mean([s["mean_comp_len"] for s in r["step_log"]]),
            elapsed=r["elapsed_seconds"], eval_s=r["eval_seconds"],
            disagree_post=[i for i in np.where(mine["post"] != np.array(r["post_correct"]))[0].tolist()],
            disagree_pre=int((mine["pre"] != np.array(r["pre_correct"])).sum()))

    R = {"schema_version": ANALYSIS_VERSION, "grader_version": GRADER_VERSION, "protocol": protocol, "protocol_sha256": object_hash(protocol), "arms": {}, "contrasts": {}, "item_level": {}, "power": {}, "selected_inputs": [s for _, _, s in validated_runs]}
    for a, d in runs.items():
        v = list(d.values())
        R["arms"][a] = {
            "n_seeds": len(d), "seeds": sorted(d),
            "base_pre_acc_stored_mean": np.mean([x["stored_pre"].mean() for x in v]),
            "post_acc_stored_mean": np.mean([x["stored_post"].mean() for x in v]),
            "post_acc_mine_mean": np.mean([x["mine_post"].mean() for x in v]),
            "post_sd_across_seeds": float(np.std([x["stored_post"].mean() for x in v], ddof=1)) if len(v) > 1 else None,
            "item_disagreements_post_total": sum(len(x["disagree_post"]) for x in v),
            "item_disagreements_pre_total": sum(x["disagree_pre"] for x in v),
            "items_graded": sum(len(x["mine_post"]) for x in v),
            "items_graded_by_phase": {phase: sum(len(x["mine_" + phase]) for x in v) for phase in ("pre", "post")},
            "eval_trunc_pre": np.mean([x["pre_trunc"] for x in v]),
            "eval_trunc_post": np.mean([x["post_trunc"] for x in v]),
            "eval_nobox_post": np.mean([x["nobox"]["post"] for x in v]),
            "train_trunc_mean": np.mean([x["train_trunc"] for x in v]),
            "train_comp_len_mean": np.mean([x["train_len"] for x in v]),
            "elapsed_s_mean": np.mean([x["elapsed"] for x in v]), "elapsed_s_max": np.max([x["elapsed"] for x in v]),
        }
    # sample disagreements for inspection
    ex = []
    for a, d in runs.items():
        for s, x in d.items():
            for i in x["disagree_post"][:2]:
                ex.append((a, s, i, int(x["stored_post"][i]), int(x["mine_post"][i]), gold[i]))
    R["disagreement_examples"] = ex[:12]

    for key in ("stored", "mine"):
        post = {a: {s: x[f"{key}_post"].mean() for s, x in d.items()} for a, d in runs.items()}
        pairs = {"g8-ppo":("grpo_g8", "ppo"), "g8-g2":("grpo_g8", "grpo_g2"), "g2-ppo":("grpo_g2", "ppo")}
        C = {name:paired(post.get(a, {}), post.get(b, {})) for name,(a,b) in pairs.items()}
        if "ppo_8x8" in post:
            C["g8-ppo_8x8"] = paired(post.get("grpo_g8", {}), post["ppo_8x8"])
        fam = protocol["primary_contrasts"]
        missing = [k for k in fam if k not in C or C[k]["seeds"] != protocol["expected_seeds"]]
        R["family_status"] = "INCOMPLETE_PRIMARY_FAMILY" if missing else "COMPLETE_PRIMARY_FAMILY"
        R["confirmatory_publication_allowed"] = not missing
        R["missing_primary_contrasts"] = missing
        ht = holm({k: C[k]["p_t"] if k in C and C[k]["p_t"] is not None else 1.0 for k in fam})
        hs = holm({k: C[k]["p_signflip"] if k in C and C[k]["p_signflip"] is not None else 1.0 for k in fam})
        for k in fam:
            if k in C:
                C[k]["holm_p_t"], C[k]["holm_p_signflip"] = ht[k], hs[k]
        R["holm_family"] = fam
        R["contrasts"][key] = C

    # also delta-vs-base (post - pre, same run) contrasts as a covariate-adjusted check
    post_d = {a: {s: x["stored_post"].mean() - x["stored_pre"].mean() for s, x in d.items()} for a, d in runs.items()}
    R["contrasts"]["gain_over_own_pre"] = {"g8-ppo": paired(post_d.get("grpo_g8", {}), post_d.get("ppo", {}))}

    # item-level secondary
    rng = np.random.default_rng(1)
    ITEM = {"g8-ppo": ("grpo_g8", "ppo"), "g8-g2": ("grpo_g8", "grpo_g2"), "g2-ppo": ("grpo_g2", "ppo")}
    if "ppo_8x8" in runs:
        ITEM["g8-ppo_8x8"] = ("grpo_g8", "ppo_8x8")
    for k, (x, y) in ITEM.items():
        seeds = sorted(set(runs.get(x, {})) & set(runs.get(y, {})))
        if not seeds:
            R["item_level"][k] = {"status":"unavailable", "reason":"no shared seeds"}
            continue
        X = np.stack([runs[x][s]["stored_post"] for s in seeds])  # seeds x items
        Y = np.stack([runs[y][s]["stored_post"] for s in seeds])
        D = X - Y
        b, c = int((D == 1).sum()), int((D == -1).sum())
        # per-seed McNemar
        per_seed = [mcnemar_exact(int((D[i] == 1).sum()), int((D[i] == -1).sum())) for i in range(len(seeds))]
        # two-way clustered bootstrap: resample seeds and items independently
        B, nS, nI = 4000, D.shape[0], D.shape[1]
        bs = np.empty(B)
        for j in range(B):
            si = rng.integers(0, nS, nS); ii = rng.integers(0, nI, nI)
            bs[j] = D[np.ix_(si, ii)].mean()
        # items-only bootstrap (seeds fixed) -> conditional on these seeds
        bi = np.array([D[:, rng.integers(0, nI, nI)].mean() for _ in range(4000)])
        item_mean = D.mean(0)
        nz = item_mean[item_mean != 0]
        R["item_level"][k] = {"n_seeds": nS, "mean_diff": D.mean(), "pooled_x_only": b, "pooled_y_only": c,
                              "pooled_mcnemar_naive_p": mcnemar_exact(b, c),
                              "per_seed_mcnemar_p": per_seed, "discordant_pairs": b + c,
                              "status": "no_discordance" if b + c == 0 else "available",
                              "twoway_boot_ci": list(np.percentile(bs, [2.5, 97.5])),
                              "items_only_boot_ci": list(np.percentile(bi, [2.5, 97.5])),
                              "wilcoxon_items_seedavg_p": wilcoxon(nz).pvalue if len(nz) else None}

    for k, c in R["contrasts"]["stored"].items():
        sd = c["sd"]
        power = {"scope": "unadjusted paired-t planning sensitivity; not actual family or equivalence design power",
                 "sd_diff": sd, "minimum_two_sided_signflip_p": 2 / 2 ** c["n"] if c["n"] else None,
                 "minimum_first_holm_p": min(1.0, len(protocol["primary_contrasts"]) * 2 / 2 ** c["n"]) if c["n"] else None}
        if sd is None or sd <= 0:
            power.update(status="unavailable", reason="zero_or_undefined_pilot_sd")
        else:
            power.update(status="available", mde_n7=mde(sd, 7), mde_n10=mde(sd, 10),
                         conservative_family_mde_n7=mde(sd, 7, alpha=.05 / len(protocol["primary_contrasts"])),
                         sd_sensitivity=[{"sd":sd * multiplier, "conservative_family_mde_n7":mde(sd * multiplier, 7, alpha=.05/len(protocol["primary_contrasts"]))} for multiplier in (.5,1.,2.)],
                     seeds_for_0_05=n_needed(sd, .05), seeds_for_0_01=n_needed(sd, .01))
        power["tost_0.02_p"] = c.get("tost_p")
        power["equivalence_status"] = "unavailable" if c.get("tost_p") is None else "margin_sensitivity_only"
        R["power"][k] = power
    family = protocol["primary_contrasts"]
    tost = holm({k: (R["power"][k]["tost_0.02_p"] if k in R["power"] and R["power"][k]["tost_0.02_p"] is not None else 1.0) for k in family})
    for k in family:
        if k in R["power"]: R["power"][k]["holm_tost_p"] = tost[k]
    R["actual_design_power"] = {"status": "not_established", "reason": "Requires prospective joint family simulation and separately justified equivalence margin"}
    R["claim_evidence"] = {"level": "independent_per_item_regrade_and_seed_statistics", "unit": "accuracy_fraction", "rounding": "unrounded", "dataset_sha256": gold_sha, "analysis_version": ANALYSIS_VERSION, "grader_version": GRADER_VERSION, "selected_input_hashes": [s["sha256"] for _, _, s in validated_runs]}
    R["claim_evidence"]["dependency_versions"] = {"python":platform.python_version(), "numpy":np.__version__, "scipy":scipy.__version__}
    R["claim_evidence"]["dataset_revision"] = protocol["dataset"]["revision"]
    R["claim_evidence"]["protocol_sha256"] = R["protocol_sha256"]
    R["claim_evidence"]["completion_status"] = R["family_status"]
    R["claim_receipts"] = []
    for arm, values in R["arms"].items():
        for field in values:
            R["claim_receipts"].append({"claim":arm + ":" + field, "result_field":"/arms/" + arm + "/" + field,
                "evidence_ref":"/claim_evidence", "rounding":"unrounded JSON; displayed rounding belongs to renderer"})
    for source, contrasts in R["contrasts"].items():
        for name, values in contrasts.items():
            for field in values:
                R["claim_receipts"].append({"claim":source + ":" + name + ":" + field,
                    "result_field":"/contrasts/" + source + "/" + name + "/" + field,
                    "evidence_ref":"/claim_evidence", "rounding":"unrounded JSON"})
    return R

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args(argv)
    require(len(args.paths) in (2, 3), "provide DATA GOLD, optionally preceded by SCRIPTS")
    print(json.dumps(analyze(*args.paths[-2:], load(args.protocol)), indent=2, allow_nan=False, default=float))

if __name__ == "__main__": main()
