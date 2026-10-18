"""Tinker side of the matched TRL-vs-Tinker run (runs locally against the Tinker API).

Per seed: Qwen3-8B LoRA r32, 30 steps x 2 prompts x 8,
temp 0.8 / top_p 0.95 / 512 tok, stop on TRL's eos ids, campaign_v2 reward and group
advantage (population std + 1e-8). On-policy: sampler weights refreshed EVERY step
(campaign_v2 refreshed every 10; TRL is on-policy)
LoRA: r32/alpha32 on q,k,v,o,gate,up,down; unembed NOT trained (TRL 1.16 cannot train lm_head LoRA).. Loss: -sum_i A_i * sum_t logp(response
tokens) / N  (campaign_v2 summed over prompt tokens too and had a dummy target; both removed).
Adam lr 1e-5, betas (0.9, 0.95), eps 1e-8, wd 0, no grad clipping.

  python matched_tinker.py OUT_DIR
"""

import json, os, sys, time, platform
from concurrent.futures import ThreadPoolExecutor

import torch
import tinker
import tinker.types as T
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matched_common import *  # noqa

OUT = sys.argv[1]
EOS = [
    151645,
    151643,
]  # TRL eos_token_ids for Qwen3-8B (probe_Qwen3-8B.json); dedup of [151645, 151645, 151643]


def run(seed):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    stream = prompt_stream(seed, gsm8k("train"))
    svc = tinker.ServiceClient()
    tc = svc.create_lora_training_client(
        base_model=MODEL, rank=32, seed=seed, train_mlp=True, train_attn=True, train_unembed=False
    )
    log = []
    for step, batch in enumerate(stream):
        sc = tc.save_weights_and_get_sampling_client()
        data, advs, masks, rs, lens, trunc = [], [], [], [], [], []
        futs = []
        for j, (q, ans) in enumerate(batch):
            pid = tok.encode(build_prompt(q), add_special_tokens=False)
            sp = T.SamplingParams(
                max_tokens=MAX_TOK,
                temperature=TEMP,
                top_p=TOP_P,
                stop=EOS,
                seed=seed * 1000 + step * 10 + j,
            )
            futs.append(
                (
                    pid,
                    ans,
                    sc.sample(T.ModelInput.from_ints(pid), num_samples=GROUP, sampling_params=sp),
                )
            )
        for pid, ans, f in futs:
            resp = f.result()
            rews = [
                reward_fn(tok.decode(list(r.tokens), skip_special_tokens=True), ans)
                for r in resp.sequences
            ]
            mr = sum(rews) / len(rews)
            sr = (sum((r - mr) ** 2 for r in rews) / len(rews)) ** 0.5 + 1e-8
            for r_seq, r in zip(resp.sequences, rews, strict=False):
                rid = list(r_seq.tokens)
                fid = pid + rid
                data.append(
                    T.Datum(
                        model_input=T.ModelInput.from_ints(fid[:-1]),
                        loss_fn_inputs={
                            "target_tokens": T.TensorData(
                                data=fid[1:], dtype="int64", shape=[len(fid) - 1]
                            )
                        },
                    )
                )
                m = torch.zeros(len(fid) - 1)
                m[len(pid) - 1 :] = 1.0
                masks.append(m)
                advs.append((r - mr) / sr)
                rs.append(r)
                lens.append(len(rid))
                trunc.append(int(not any(t in EOS for t in rid)))

        def loss_fn(_data, lp):
            loss = torch.stack(
                [-(advs[i] * (lp[i] * masks[i]).sum()) for i in range(len(lp))]
            ).sum() / len(lp)
            return loss, {"loss": loss.item()}

        tc.forward_backward_custom(data=data, loss_fn=loss_fn).result()
        tc.optim_step(
            T.AdamParams(
                learning_rate=LR,
                beta1=ADAM["beta1"],
                beta2=ADAM["beta2"],
                eps=ADAM["eps"],
                weight_decay=ADAM["weight_decay"],
                grad_clip_norm=0.0,
            )
        ).result()
        g = [rs[i : i + GROUP] for i in range(0, len(rs), GROUP)]
        log.append(
            {
                "step": step,
                "mean_reward": sum(rs) / len(rs),
                "zvf": sum(len(set(x)) == 1 for x in g) / len(g),
                "mean_len": sum(lens) / len(lens),
                "trunc_frac": sum(trunc) / len(trunc),
                "questions": [q[:60] for q, _ in batch],
            }
        )
        print(
            f"[tinker s{seed}] step {step} r={log[-1]['mean_reward']:.3f} len={log[-1]['mean_len']:.0f} trunc={log[-1]['trunc_frac']:.2f}",
            flush=True,
        )
    final = tc.save_weights_for_sampler(name=f"m{seed}_final").result().path
    res = {
        "side": "tinker",
        "seed": seed,
        "model": MODEL,
        "step_log": log,
        "final_sampler_path": final,
        "archive_url": None,  # signed URLs not stored; regenerate from final_sampler_path
        "run_id": tc.model_id if hasattr(tc, "model_id") else None,
        "config": {
            "steps": STEPS,
            "prompts_per_step": PROMPTS_PER_STEP,
            "group": GROUP,
            "lr": LR,
            "adam": ADAM,
            "lora": "Tinker rank=32, train_mlp=True, train_attn=True, train_unembed=False (alpha 32, all-linear)",
            "temp": TEMP,
            "top_p": TOP_P,
            "max_tokens": MAX_TOK,
            "stop": EOS,
            "on_policy": "refresh every step",
            "loss": "-sum_i A_i sum_t logp_t(response) / N; A=(r-mean)/(pop std+1e-8)",
        },
        "env": {
            "backend": "Tinker API",
            "tinker_sdk": tinker.__version__,
            "python": platform.python_version(),
            "client_host": platform.node(),
            "wallclock_s": time.time() - t0,
            "seeds": {"stream": seed, "lora_init": seed, "sampling": "seed*1000+step*10+j"},
        },
    }
    json.dump(res, open(f"{OUT}/tinker_s{seed}.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with ThreadPoolExecutor(3) as ex:
        for r in ex.map(run, SEEDS):
            print("done", r["seed"], sum(s["mean_reward"] for s in r["step_log"][-10:]) / 10)
