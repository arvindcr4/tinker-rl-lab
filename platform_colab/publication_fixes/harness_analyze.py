"""Claim C harness-check analysis: 2x2 (checkpoint x harness) + crossed reward rescoring.

  python3 -I harness_analyze.py gen_Qwen3-8B-Base.json gen_Qwen3-8B.json OUT_JSON
Primary cell value: accuracy of sample 0 per item (n=500), Wilson 95% CI.
Secondary: mean over 4 samples/item, item-cluster bootstrap 95% CI.
Reference training numbers (registry / audit recompute): TRL Qwen3-8B last10 = 0.050 (step1 0.0);
Tinker Qwen3-8B-Base last10 = 0.856 (step1 0.6875); Tinker Qwen3-8B last10 = 0.338.
"""
import json, sys
import numpy as np
from scipy.stats import norm

REF = {"trl_qwen3_8b_last10": 0.050, "tinker_qwen3_8b_base_last10": 0.856, "tinker_qwen3_8b_last10": 0.338}


def wilson(k, n, z=norm.ppf(0.975)):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def boot(M, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    return list(np.percentile(M[rng.integers(0, len(M), (B, len(M)))].mean((1, 2)), [2.5, 97.5]))


def boot_diff(A, Bm, B=10000, seed=0):
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(A), (B, len(A)))
    return list(np.percentile(A[idx].mean((1, 2)) - Bm[idx].mean((1, 2)), [2.5, 97.5]))


out = {"cells": {}, "decomposition": {}, "reference_training_numbers": REF}
data = {}
for f in sys.argv[1:3]:
    G = json.load(open(f))
    name = "base" if G["model"].endswith("Base") else "instruct"
    out.setdefault("env", {})[name] = G["env"]
    out.setdefault("probe", {})[name] = {k: G["probe"][k] for k in ("trl_generation_kwargs", "trl_eos_token_ids", "trl_uses_parse_response",
                                                                     "tokenizer_has_chat_template", "chat_template_kwargs", "versions",
                                                                     "trl_prompt_example", "tinker_prompt_example", "model_generation_config")}
    for cell, c in G["cells"].items():
        for rf in ("r_trl", "r_tinker"):
            M = np.array(c[rf])  # items x samples
            data[(name, cell, rf)] = M
            k0 = int(M[:, 0].sum())
            out["cells"][f"{name}|{cell}|{rf}"] = {
                "acc_sample0": k0 / len(M), "wilson95_sample0": wilson(k0, len(M)),
                "acc_mean4": float(M.mean()), "boot95_mean4": boot(M[:, :, None]),
            }
        texts = c["texts"]
        flat = [t for ts in texts for t in ts]
        out["cells"][f"{name}|{cell}|diag"] = {
            "sampling": c["sampling"],
            "trunc_frac": float(np.mean([x == "length" for fs in c["finish"] for x in fs])),
            "mean_tokens": float(np.mean(c["n_tokens"])),
            "frac_has_think_open": float(np.mean(["<think>" in t for t in flat])),
            "frac_has_think_close": float(np.mean(["</think>" in t for t in flat])),
            "frac_has_boxed": float(np.mean(["\\boxed{" in t for t in flat])),
        }

native = {"trl": "r_trl", "tinker": "r_tinker", "trl_modelgen": "r_trl"}
tab = {}
for name in ("base", "instruct"):
    for cell in ("trl", "tinker", "trl_modelgen"):
        tab[(name, cell)] = data[(name, cell, native[cell])]
out["table_2x2_native_reward"] = {f"{n}|{c}": out["cells"][f"{n}|{c}|{native[c]}"] for (n, c) in tab}

# Path from TRL-instruct (the 5% run's configuration) to Tinker-base (the 85.6% run's configuration).
I_trl, I_trl_tkR = data[("instruct", "trl", "r_trl")], data[("instruct", "trl", "r_tinker")]
I_tk, B_tk = data[("instruct", "tinker", "r_tinker")], data[("base", "tinker", "r_tinker")]
B_trl = data[("base", "trl", "r_trl")]
steps = {"1_reward_fn (TRL texts, TRL->Tinker reward)": (I_trl_tkR, I_trl),
         "2_prompt+decoding (Instruct, TRL->Tinker harness, Tinker reward)": (I_tk, I_trl_tkR),
         "3_checkpoint (Tinker harness, Instruct->Base)": (B_tk, I_tk)}
total = B_tk.mean() - I_trl.mean()
out["decomposition"]["path_instruct_trl_to_base_tinker"] = {
    k: {"delta": float(a.mean() - b.mean()), "ci95": boot_diff(a[:, :, None], b[:, :, None]), "share_of_total": float((a.mean() - b.mean()) / total)}
    for k, (a, b) in steps.items()}
out["decomposition"]["total_inference_gap"] = float(total)
out["decomposition"]["alt_order_checkpoint_first"] = {
    "checkpoint (TRL harness, Instruct->Base)": float(B_trl.mean() - I_trl.mean()),
    "harness (Base, TRL->Tinker, native rewards)": float(B_tk.mean() - B_trl.mean())}
out["decomposition"]["training_residual"] = {
    "trl_instruct: registry last10 - step0 inference (TRL harness)": REF["trl_qwen3_8b_last10"] - float(I_trl.mean()),
    "tinker_base: registry last10 - step0 inference (Tinker harness)": REF["tinker_qwen3_8b_base_last10"] - float(B_tk.mean()),
    "tinker_instruct: registry last10 - step0 inference (Tinker harness)": REF["tinker_qwen3_8b_last10"] - float(I_tk.mean()),
    "observed_gap_0.856_minus_0.050": REF["tinker_qwen3_8b_base_last10"] - REF["trl_qwen3_8b_last10"]}
json.dump(out, open(sys.argv[3], "w"), indent=1)
for k, v in out["table_2x2_native_reward"].items():
    print(f"{k:24s} s0={v['acc_sample0']:.3f} [{v['wilson95_sample0'][0]:.3f},{v['wilson95_sample0'][1]:.3f}] mean4={v['acc_mean4']:.3f}")
print(json.dumps(out["decomposition"], indent=1))
for k, v in out["cells"].items():
    if k.endswith("diag"): print(k, {x: v[x] for x in ("trunc_frac", "mean_tokens", "frac_has_think_close", "frac_has_boxed")})
