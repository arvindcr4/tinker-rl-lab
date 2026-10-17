"""Corrected same-stack runner, new v3 protocol (legacy v2 artifacts untouched).

Arms retain their historical short labels. The PPO-labelled variant is a
prompt-end detached linear terminal-reward baseline with no GAE, not a claim
of general PPO/GRPO equivalence. G8 policy-update prompt schedules match;
critic warm-up adds PPO exposure and compute. This protocol has no new results.

Modal: modal run experiments/modal/modal_samestack_gsm8k_cot_v2.py [--smoke]
Colab: set SAMESTACK_EXECUTION_ENV=colab, then call run_arm_local explicitly.
Importing in Colab never creates a Modal stub or launches an experiment.
"""
import itertools
import math
import os
from pathlib import Path
import sys
import time

# Works both from repository root and platform_hybrid's historical launch cwd.
REPO_ROOT = next((parent for parent in Path(__file__).resolve().parents
                  if (parent / "platform_hybrid/experiments/samestack_protocol.py").is_file()),
                 Path("/root"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from platform_hybrid.experiments.samestack_protocol import (
    PROTOCOL_ID, PROTOCOL_HASH, PROTOCOL_SPEC, ANALYSIS_PLAN_ID, GRADING_SPEC_ID,
    boxed_reward, budget_accounting, canonical_hash, clear_inherited_generation_defaults, generation_configs,
    prompt_schedules, resolved_token_ids, strict_json_dump,
)
from platform_hybrid.experiments.samestack_training import (
    clipped_response_loss, completion_log_probs, rollout_attention,
)
from platform_hybrid.experiments.samestack_manifest import (
    capture_source, example_ids, hardware_manifest, resolve_revisions, utc_now, verify_source_files,
)
from platform_hybrid.experiments.samestack_contract import load_completed, write_completion
from platform_hybrid.experiments.samestack_dependencies import REQUIREMENTS, lock_manifest, verify_runtime

EXECUTION_ENVIRONMENT = os.environ.get("SAMESTACK_EXECUTION_ENV", "modal")
RESULTS_DIR = os.environ.get("SAMESTACK_RESULTS_DIR", "/results")
app = results_vol = image = None
if EXECUTION_ENVIRONMENT == "modal":
    import modal
    app = modal.App("tinkerrl-samestack-gsm8k-cot-v3")
    results_vol = modal.Volume.from_name("tinkerrl-results", create_if_missing=True)
    image = modal.Image.debian_slim(python_version="3.11").pip_install(*REQUIREMENTS)
    for name in ("samestack_protocol", "samestack_training", "samestack_manifest",
                 "samestack_contract", "samestack_dependencies"):
        image = image.add_local_file(
            str(REPO_ROOT / "platform_hybrid/experiments" / f"{name}.py"),
            f"/root/platform_hybrid/experiments/{name}.py")
    image = image.add_local_file(str(Path(__file__).resolve()),
                                "/root/platform_hybrid/experiments/modal/modal_samestack_gsm8k_cot_v2.py")
elif EXECUTION_ENVIRONMENT != "colab":
    raise ValueError("SAMESTACK_EXECUTION_ENV must be modal or colab")

SEEDS = [42, 123, 456, 789, 1024, 2048, 3141, 4096, 5555, 7777]
ARMS = ["grpo_g8", "grpo_g2", "ppo", "ppo_8x8"]
PRIMARY = [("grpo_g8", "ppo"), ("grpo_g8", "ppo_8x8"), ("grpo_g8", "grpo_g2")]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET = "openai/gsm8k"
N_STEPS, N_GEN = 30, 64
GROUP = {"grpo_g8": 8, "grpo_g2": 2, "ppo": 1, "ppo_8x8": 8}
K_EPOCHS, CLIP, LR, MAX_NEW = 2, 0.2, 1e-5, 512
N_EVAL, EVAL_BS, EPS, CHUNK = None, 64, 1e-6, 4
V_LR, V_ITERS, W_WARM = 1e-3, 50, 4
MARGIN = 0.02


def run_arm_local(arm: str, seed: int, smoke: bool = False, source_provenance=None) -> dict:
    """Execute only when explicitly invoked; resume only complete immutable bundles."""
    if arm not in GROUP or seed not in SEEDS:
        raise ValueError("Unknown prespecified arm or seed")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import gzip
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dependencies = verify_runtime()
    started_at = utc_now()
    source = source_provenance or capture_source(REPO_ROOT)
    if source.get("status") != "observed" or not source.get("commit") or not source.get("dirty_hash"):
        raise ValueError("Verified source commit and dirty hash required before execution")
    verify_source_files(source, REPO_ROOT)
    environment = EXECUTION_ENVIRONMENT
    n_steps, n_eval, w_warm = (2, 32, 1) if smoke else (N_STEPS, N_EVAL, W_WARM)
    is_ppo = arm.startswith("ppo")
    w_warm = w_warm if is_ppo else 0
    G, n_prompts = GROUP[arm], N_GEN // GROUP[arm]
    revisions = resolve_revisions(
        MODEL, DATASET, os.environ.get("SAMESTACK_MODEL_REVISION", "main"),
        os.environ.get("SAMESTACK_TOKENIZER_REVISION"),
        os.environ.get("SAMESTACK_DATASET_REVISION", "main"))
    ds = load_dataset(DATASET, "main", revision=revisions["dataset"]["revision"])
    train = ds["train"]
    all_test_ids = example_ids(ds["test"], "test")
    test_indices = list(range(len(ds["test"])))
    if n_eval is not None:
        test_indices = np.random.default_rng(0).permutation(len(test_indices))[:n_eval].tolist()
    test = [ds["test"][i] for i in test_indices]
    if len(test) != (32 if smoke else 1319):
        raise ValueError("Resolved dataset does not have the required evaluation size")
    test_ids = [all_test_ids[i] for i in test_indices]
    train_ids = example_ids(train, "train")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=revisions["tokenizer"]["revision"])
    tok.padding_side = "left"
    # Load config only for EOS resolution before allocating model weights/resuming.
    from transformers import AutoConfig, GenerationConfig
    from types import SimpleNamespace
    config = AutoConfig.from_pretrained(MODEL, revision=revisions["model"]["revision"])
    config._attn_implementation = "sdpa"
    model_generation = GenerationConfig.from_pretrained(MODEL, revision=revisions["model"]["revision"])
    eos_ids, pad_id = resolved_token_ids(tok, SimpleNamespace(config=config, generation_config=model_generation))
    train_generation, eval_generation = generation_configs(eos_ids, pad_id, MAX_NEW)
    generations = {"train": train_generation.to_dict(), "eval": eval_generation.to_dict()}
    critic = {"variant": "prompt-end detached linear terminal-reward baseline; no GAE",
              "input": "detached prompt-end hidden state", "init": "zeros", "v_lr": V_LR,
              "v_iters": V_ITERS, "warm_batches": w_warm, "warm_generations": w_warm * N_GEN} if is_ppo else None
    effective = {"model": MODEL, "tokenizer": MODEL, "dataset": DATASET, "dataset_config": "main",
                 "model_revision": revisions["model"]["revision"],
                 "tokenizer_revision": revisions["tokenizer"]["revision"],
                 "dataset_revision": revisions["dataset"]["revision"],
                 "arm": arm, "seed": seed, "smoke": smoke, "group": G,
                 "prompts_per_step": n_prompts, "n_gen": N_GEN, "n_steps": n_steps,
                 "n_eval": len(test), "k_epochs": K_EPOCHS, "clip": CLIP, "lr": LR,
                 "max_new": MAX_NEW, "eval_bs": EVAL_BS, "eps": EPS, "chunk": CHUNK,
                 "critic": critic, "generation": generations,
                 "policy_optimizer": {"name": "Adam", "betas": [0.9, 0.999], "eps": 1e-8,
                                      "weight_decay": 0, "max_grad_norm": 1.0},
                 "lora": {"r": 16, "alpha": 32, "dropout": 0.0,
                          "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
                 "loss_normalization": "per-response mean over first-EOS-inclusive tokens",
                 "reference_kl": None, "policy_dropout_enabled": False,
                 "attention_backend": "sdpa",
                 "precision": {"model": "torch.bfloat16", "logprobs": "torch.float32",
                               "value_head": "torch.float32"},
                 "advantage_normalization": "batch sample-std" if is_ppo else "group population-std"}
    config_hash = canonical_hash(effective)
    hardware = hardware_manifest(torch, SimpleNamespace(config=config), model_dtype="torch.bfloat16")
    environment_hash = canonical_hash({"environment": environment, "dependencies": dependencies, "hardware": hardware})
    identity = {"protocol_hash": PROTOCOL_HASH, "config_hash": config_hash,
                "environment": environment, "environment_hash": environment_hash, "source_commit": source["commit"],
                "source_dirty_hash": source["dirty_hash"]}
    run_id = canonical_hash(identity)
    expected = dict(identity, protocol_id=PROTOCOL_ID, run_id=run_id)
    sub = PROTOCOL_ID + ("_smoke" if smoke else "")
    out_path = Path(RESULTS_DIR) / sub / f"{arm}_s{seed}.json"
    if out_path.exists():
        completed = load_completed(out_path, expected=expected)
        if completed is not None:
            return completed
        # Do not overwrite mismatched/corrupt/incomplete evidence or legacy results.
        raise ValueError(f"Existing run is not a matching complete artifact bundle: {out_path}")

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = "cuda"
    model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(MODEL, revision=revisions["model"]["revision"],
                                           dtype=torch.bfloat16, attn_implementation="sdpa").to(device),
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    # Eval mode disables dropout, not gradients: scoring and sampling see one policy.
    model.eval()
    clear_inherited_generation_defaults(model, train_generation)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    vhead = vopt = None
    if is_ppo:
        vhead = nn.Linear(model.config.hidden_size, 1).to(device=device, dtype=torch.float32)
        nn.init.zeros_(vhead.weight); nn.init.zeros_(vhead.bias)
        vopt = torch.optim.Adam(vhead.parameters(), lr=V_LR)
    schedules = prompt_schedules(seed, len(train), n_steps, w_warm, n_prompts, N_GEN)
    schedules["policy_item_ids"] = [[train_ids[i] for i in row] for row in schedules["policy"]]
    schedules["critic_warmup_item_ids"] = [[train_ids[i] for i in row] for row in schedules["critic_warmup"]]
    if hardware_manifest(torch, model) != hardware:
        raise ValueError("Observed model backend/precision differs from the effective runtime manifest")
    manifest = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "protocol_hash": PROTOCOL_HASH,
                "protocol_spec": PROTOCOL_SPEC, "grading_spec_id": GRADING_SPEC_ID,
                "analysis_plan_id": ANALYSIS_PLAN_ID, "run_id": run_id,
                "environment": environment, "source": source, "dependencies": dependencies,
                "dependency_lock": lock_manifest(), "hardware": hardware,
                "environment_hash": environment_hash,
                "revisions": revisions, "chat_template_hash": canonical_hash(tok.chat_template),
                "train_item_ids": train_ids, "test_item_ids": test_ids,
                "ordered_test_item_hash": canonical_hash(test_ids), "generation": generations,
                "schedules": schedules, "budget": budget_accounting(n_steps, N_GEN, G, w_warm),
                "effective_config": effective, "config_hash": config_hash,
                "timestamps": {"started_at": started_at}}

    def gold_of(answer):
        return answer.split("####")[-1].strip()

    def build(question):
        messages = [{"role": "user", "content": question + "\nThink step by step, then give the final answer as \\boxed{ANSWER}."}]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def truncated(ids):
        return [not any(int(token) in eos_ids for token in row) for row in ids]

    def gen_batch(prompts, greedy):
        encoded = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        plen = encoded["input_ids"].shape[1]
        with torch.no_grad():
            generated = model.generate(**encoded, generation_config=eval_generation if greedy else train_generation)
        return encoded, plen, generated[:, plen:]

    def heldout():
        correct, trunc, texts = [], [], []
        for i in range(0, len(test), EVAL_BS):
            rows = test[i:i + EVAL_BS]
            _, _, completion = gen_batch([build(row["question"]) for row in rows], greedy=True)
            trunc.extend(int(value) for value in truncated(completion))
            for row, text in zip(rows, tok.batch_decode(completion, skip_special_tokens=True)):
                correct.append(int(boxed_reward(text, gold_of(row["answer"]))))
                texts.append(text)
        return correct, trunc, texts

    def forward(full, attention, plen, start, end, want_features):
        return completion_log_probs(model, full[start:end], attention[start:end], plen, want_features)

    def batch_for(indices):
        prompts, golds = [], []
        for index in indices:
            row = train[index]
            for _ in range(G):
                prompts.append(build(row["question"])); golds.append(gold_of(row["answer"]))
        return prompts, golds

    def rollout(prompts, golds):
        encoded, plen, completion = gen_batch(prompts, greedy=False)
        texts = tok.batch_decode(completion, skip_special_tokens=True)
        rewards = np.array([boxed_reward(text, gold) for text, gold in zip(texts, golds)], dtype=np.float32)
        full = torch.cat([encoded["input_ids"], completion], dim=1)
        attention, mask = rollout_attention(encoded["attention_mask"], completion, eos_ids)
        return full, attention, plen, completion, mask.float(), rewards

    def fit_value(features, rewards):
        for _ in range(V_ITERS):
            vopt.zero_grad()
            loss = 0.5 * ((vhead(features).squeeze(-1) - rewards) ** 2).mean()
            loss.backward(); vopt.step()
        return float(loss.item())

    t0 = time.time()
    pre_correct, pre_trunc, pre_texts = heldout()
    eval_seconds = time.time() - t0
    # Separate warm-up and policy sampling RNG; no warm-up shift of policy RNG.
    torch.manual_seed(schedules["warmup_sampling_seed"])
    torch.cuda.manual_seed_all(schedules["warmup_sampling_seed"])
    warm_log = []
    for warm_step, indices in enumerate(schedules["critic_warmup"]):
        full, attention, plen, _, _, rewards = rollout(*batch_for(indices))
        rew_t = torch.tensor(rewards, device=device)
        with torch.no_grad():
            features = torch.cat([forward(full, attention, plen, start, min(start + CHUNK, len(full)), True)[1]
                                  for start in range(0, len(full), CHUNK)])
            pre_mse = float(0.5 * ((vhead(features).squeeze(-1) - rew_t) ** 2).mean())
        warm_log.append({"warm_step": warm_step, "mean_reward": float(rewards.mean()),
                         "value_loss_before_fit": pre_mse, "value_loss_after_fit": fit_value(features, rew_t)})
    torch.manual_seed(schedules["policy_sampling_seed"])
    torch.cuda.manual_seed_all(schedules["policy_sampling_seed"])
    step_log = []
    for step, indices in enumerate(schedules["policy"]):
        full, attention, plen, completion, mask, rewards = rollout(*batch_for(indices))
        rew_t, batch_size = torch.tensor(rewards, device=device), full.shape[0]
        with torch.no_grad():
            old_parts, features = [], []
            for start in range(0, batch_size, CHUNK):
                lp, feat = forward(full, attention, plen, start, min(start + CHUNK, batch_size), is_ppo)
                old_parts.append(lp)
                if is_ppo:
                    features.append(feat)
            old_lp = torch.cat(old_parts)
        vl_pre = vl_post = None
        if is_ppo:
            features = torch.cat(features)
            with torch.no_grad():
                old_value = vhead(features).squeeze(-1)
            vl_pre = float(0.5 * ((old_value - rew_t) ** 2).mean())
            advantages = rew_t - old_value
            advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)
            zvf = None
        else:
            grouped = rewards.reshape(n_prompts, G)
            group_std = grouped.std(1, keepdims=True)
            zvf = float((group_std[:, 0] <= EPS).mean())
            normalized = (grouped - grouped.mean(1, keepdims=True)) / (group_std + EPS)
            advantages = torch.tensor(normalized.reshape(-1), dtype=torch.float32, device=device)
        for _ in range(K_EPOCHS):
            opt.zero_grad()
            for start in range(0, batch_size, CHUNK):
                end = min(start + CHUNK, batch_size)
                lp, _ = forward(full, attention, plen, start, end, False)
                loss = clipped_response_loss(lp, old_lp[start:end], advantages[start:end],
                                             mask[start:end], CLIP, batch_size)
                loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
        if is_ppo:
            vl_post = fit_value(features, rew_t)
        comp_len = float(mask.sum(1).mean())
        trunc_frac = float(np.mean(truncated(completion)))
        step_log.append({"step": step, "mean_reward": float(rewards.mean()),
                         "constant_mean_inbatch_half_mse": float(0.5 * rewards.var()), "zvf": zvf,
                         "mean_comp_len": comp_len, "trunc_frac": trunc_frac, "grad_norm": grad_norm,
                         "value_loss_pre_fit": vl_pre, "value_loss_post_fit": vl_post})
        print(f"[{arm} s{seed}] step {step} r={rewards.mean():.3f} len={comp_len:.0f} trunc={trunc_frac:.2f}", flush=True)
    eval_start = time.time()
    post_correct, post_trunc, post_texts = heldout()
    eval_seconds += time.time() - eval_start
    pre, post = np.array(pre_correct), np.array(post_correct)
    manifest["timestamps"]["finished_at"] = utc_now()
    result = {**effective, "experiment": PROTOCOL_ID, "execution_environment": environment,
              "manifest": manifest, "heldout_pre_acc": float(pre.mean()), "heldout_post_acc": float(post.mean()),
              "heldout_pre_trunc": float(np.mean(pre_trunc)), "heldout_post_trunc": float(np.mean(post_trunc)),
              "wrong_to_right": int(((pre == 0) & (post == 1)).sum()),
              "right_to_wrong": int(((pre == 1) & (post == 0)).sum()),
              "last10_avg": float(np.mean([row["mean_reward"] for row in step_log[-10:]])),
              "eval_seconds": eval_seconds, "elapsed_seconds": time.time() - t0,
              "pre_correct": pre_correct, "post_correct": post_correct,
              "pre_trunc": pre_trunc, "post_trunc": post_trunc, "warm_log": warm_log, "step_log": step_log}
    adapter_dir = out_path.parent / "adapters" / out_path.stem
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    for name, generation in generations.items():
        with (adapter_dir / f"{name}_generation_config.json").open("w") as handle:
            strict_json_dump(generation, handle, indent=2)
    with gzip.open(out_path.with_name(out_path.stem + "_texts.json.gz"), "wt") as handle:
        strict_json_dump({"pre": pre_texts, "post": post_texts, "test_item_ids": test_ids}, handle)
    with out_path.open("w") as handle:
        strict_json_dump(result, handle)
    write_completion(out_path, expected=expected)
    if results_vol is not None:
        results_vol.commit()
    return result


run_arm = (app.function(image=image, gpu="A10G", timeout=6 * 3600,
                        volumes={RESULTS_DIR: results_vol}, retries=1,
                        secrets=[modal.Secret.from_name("huggingface-secret")])(run_arm_local)
           if app is not None else run_arm_local)


def _t_cdf(t, df):
    """Student-t CDF via the regularized incomplete beta (scipy-free)."""
    if not math.isfinite(df) or df <= 0 or math.isnan(t):
        raise ValueError("Student-t requires positive finite df and non-NaN t")
    if t == 0:
        return 0.5
    if math.isinf(t):
        return 1.0 if t > 0 else 0.0
    # Compute df/(df+t*t) without overflowing t*t. Keep log(x) even
    # when x underflows: for heavy tails the probability may still be finite.
    log_ratio = math.log(abs(t)) - 0.5 * math.log(df)
    if log_ratio > 0:
        log_x = -2 * log_ratio - math.log1p(math.exp(-2 * log_ratio))
        x = math.exp(log_x)
    else:
        x = 1 / (1 + (t / math.sqrt(df)) ** 2)
    if x < 1e-100:
        a = df / 2.0
        # I_x(a, 1/2) = x**a / (a*B(a,1/2)) * (1 + O(x)).
        # In this branch the omitted relative correction is below 1e-100.
        log_tail = (math.lgamma(a + 0.5) - math.lgamma(a)
                    - math.lgamma(0.5) - math.log(a) - math.log(2) + a * log_x)
        tail = math.exp(log_tail)
        return 1 - tail if t > 0 else tail
    if x == 1:
        # Near zero, squaring t loses the complement of x to rounding.
        # The local linear expansion avoids log(0); cubic error is negligible.
        density = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
        return 0.5 + t * density
    def betacf(a, b, x):
        qab, qap, qam = a + b, a + 1, a - 1
        c, d = 1.0, 1 - qab * x / qap
        d = 1 / (d if abs(d) > 1e-300 else 1e-300); h = d
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1 / (1 + aa * d or 1e-300); c = 1 + aa / c or 1e-300; h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1 / (1 + aa * d or 1e-300); c = 1 + aa / c or 1e-300; de = d * c; h *= de
            if abs(de - 1) < 3e-14:
                break
        return h
    a, b = df / 2.0, 0.5
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    ib = bt * betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * betacf(b, a, 1 - x) / b
    return 1 - 0.5 * ib if t > 0 else 0.5 * ib


def _t_ppf(p, df):
    if not 0 < p < 1 or not math.isfinite(df) or df <= 0:
        raise ValueError("Student-t quantile requires 0 < p < 1 and positive finite df")
    if p == 0.5:
        return 0.0
    # Invert the lower tail on both sides. Comparing an upper CDF near 1
    # loses relative accuracy in the requested tail through subtraction.
    q = min(p, 1 - p)
    sign = -1.0 if p < 0.5 else 1.0
    lo, hi = 0.0, 1.0
    largest = float.fromhex("0x1.fffffffffffffp+1023")
    while _t_cdf(-hi, df) > q:
        lo = hi
        if hi == largest:
            # The mathematical quantile exceeds the finite float range.
            return sign * float("inf")
        hi = min(2 * hi, largest)
    for _ in range(200):
        mid = lo / 2 + hi / 2
        if mid == lo or mid == hi:
            break
        lo, hi = (mid, hi) if _t_cdf(-mid, df) > q else (lo, mid)
    return sign * (lo / 2 + hi / 2)


def signflip_p(d):
    """Exact two-sided sign-flip permutation p on the mean of paired differences."""
    obs = abs(sum(d))
    hits = sum(abs(sum(s * x for s, x in zip(signs, d))) >= obs - 1e-12
               for signs in itertools.product([1, -1], repeat=len(d)))
    return hits / 2 ** len(d)


def paired(a: dict, b: dict) -> dict:
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    n = len(d)
    if n < 2:
        raise ValueError("paired requires at least two shared seeds")
    if not all(math.isfinite(x) for x in d):
        raise ValueError("paired requires finite paired differences")
    md = d[0] if all(x == d[0] for x in d) else math.fsum(d) / n
    sd = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n)
    df = n - 1
    t = md / se if se > 0 else (math.copysign(float("inf"), md) if md else 0.0)
    p = 2 * _t_cdf(-abs(t), df) if se > 0 else (0.0 if md else 1.0)
    tc = _t_ppf(0.975, df)
    tc90 = _t_ppf(0.95, df)
    # Use one-sided limiting probabilities at zero SE. At the equivalence
    # boundary the null is not rejected; outside it equivalence is false.
    p_lo = _t_cdf(-(md + MARGIN) / se, df) if se > 0 else (0.0 if md > -MARGIN else 0.5 if md == -MARGIN else 1.0)
    p_hi = _t_cdf((md - MARGIN) / se, df) if se > 0 else (0.0 if md < MARGIN else 0.5 if md == MARGIN else 1.0)
    return {"n_seeds": n, "seeds": seeds, "diffs": d, "mean_diff": md, "sd": sd,
            "ci95": [md - tc * se, md + tc * se],
            "ci90": [md - tc90 * se, md + tc90 * se], "t": t, "df": df, "p_two_sided": p,
            "signflip_p": signflip_p(d),
            "tost_margin": MARGIN, "tost_p": max(p_lo, p_hi),
            "equivalent_at_margin": max(p_lo, p_hi) < 0.05}


def holm(ps: dict) -> dict:
    order = sorted(ps, key=ps.get)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * ps[k]))
        out[k] = running
    return out


def main(smoke: bool = False):
    """Launch explicit jobs; summary remains descriptive until independent analysis."""
    if app is None:
        raise RuntimeError("Use run_arm_local explicitly in Colab")
    seeds = SEEDS[:1] if smoke else SEEDS
    source = capture_source(REPO_ROOT)
    jobs = [(arm, seed, smoke, source) for arm in ARMS for seed in seeds]
    results, errors = [], []
    for result in run_arm.starmap(jobs, return_exceptions=True):
        (results if isinstance(result, dict) else errors).append(result)
    summary = {}
    for arm in ARMS:
        rows = [row for row in results if row["arm"] == arm]
        if not rows:
            continue
        summary[arm] = {"n_seeds": len(rows),
                        "heldout_pre_mean": sum(row["heldout_pre_acc"] for row in rows) / len(rows),
                        "heldout_post_mean": sum(row["heldout_post_acc"] for row in rows) / len(rows)}
        if arm.startswith("ppo"):
            steps = [step for row in rows for step in row["step_log"]]
            summary[arm]["critic_value_loss_pre_fit_mean"] = sum(step["value_loss_pre_fit"] for step in steps) / len(steps)
            summary[arm]["constant_mean_inbatch_half_mse_mean"] = sum(step["constant_mean_inbatch_half_mse"] for step in steps) / len(steps)
    output = {"protocol_id": PROTOCOL_ID, "protocol_hash": PROTOCOL_HASH,
              "analysis_plan_id": ANALYSIS_PLAN_ID,
              "analysis_status": "descriptive-only; run the independent fixed-family analysis",
              "confirmatory_verdict": None, "smoke": smoke,
              "effective_configs": [row["manifest"]["effective_config"] for row in results],
              "summary": summary, "n_failed": len(errors), "errors": [str(error) for error in errors],
              "runs": results}
    stem = Path(RESULTS_DIR) / (PROTOCOL_ID + ("_smoke" if smoke else "")) / "aggregate.json"
    # Local aggregates are distinct from immutable per-run bundles and historical results.
    local = REPO_ROOT / "platform_hybrid/experiments/results" / stem.parent.name / "aggregate.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    with local.open("w") as handle:
        strict_json_dump(output, handle, indent=2)
    print(f"Saved descriptive aggregate: {local}; failed runs: {len(errors)}")


if app is not None:
    main = app.local_entrypoint()(main)
