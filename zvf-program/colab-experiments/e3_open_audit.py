"""E3: reproducibility audit in ONE controlled open trainer + live adaptive-G.

Colab-only: the original head-to-head ran on CLOSED Tinker (loss kernel
unauditable, not swappable). Here every loss arm is re-implemented in the same
open loop with identical sampler/precision/KL handling, so we can see which
algorithmic gains survive the stack being held fixed (MIN-REPORT Pillar 4), and
whether a zvf-triage-style adaptive-G actually reduces ZVF live (Pillar 3).

Arms (all share old-policy logprob caching + 2 inner epochs so ratio/clip engage):
  grpo            : adv=(r-mean)/(std+eps), symmetric clip 0.2
  drgrpo          : adv=(r-mean)  [NO /std], symmetric clip 0.2
  dapo            : adv=(r-mean)/(std+eps), asymmetric clip [0.2, 0.28],
                    dynamic sampling (resample zero-variance groups)
  grpo_adaptiveG  : grpo + zvf-triage controller raises G when recent ZVF high

Run:  colab run --gpu T4 --timeout 1200 e3_open_audit.py
"""
import argparse
import json, os, re, random, statistics, tempfile
from pathlib import Path

ARM_NAMES = ["grpo", "drgrpo", "dapo", "grpo_adaptiveG"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARM_NAMES, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--g0", type=int, default=4)
    parser.add_argument("--gmax", type=int, default=10)
    parser.add_argument("--max-new", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--inner", type=int, default=2)
    parser.add_argument("--heldout-n", type=int, default=20)
    parser.add_argument("--unit-fingerprint")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="e1-open-audit-pilot")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--hf-repo")
    parser.add_argument("--hf-path")
    parser.add_argument("--require-remote-tracking", action="store_true")
    return parser.parse_args()


ARGS = parse_args()
MODEL = ARGS.model
SEEDS = ARGS.seed or [0, 1]
ARMS = ARGS.arm or ARM_NAMES
BATCH, G0, GMAX = ARGS.batch, ARGS.g0, ARGS.gmax
MAX_NEW, LR = ARGS.max_new, ARGS.learning_rate
STEPS, INNER, HELDOUT_N = ARGS.steps, ARGS.inner, ARGS.heldout_n
TRACK_REMOTE = bool(ARGS.wandb_project or ARGS.hf_repo or ARGS.require_remote_tracking)

if ARGS.require_remote_tracking:
    missing = []
    if not ARGS.wandb_project:
        missing.append("--wandb-project")
    if not ARGS.hf_repo:
        missing.append("--hf-repo")
    if not ARGS.hf_path:
        missing.append("--hf-path")
    if not ARGS.unit_fingerprint:
        missing.append("--unit-fingerprint")
    if not os.environ.get("WANDB_API_KEY"):
        missing.append("WANDB_API_KEY")
    if not os.environ.get("HF_TOKEN"):
        missing.append("HF_TOKEN")
    if missing:
        raise SystemExit("remote tracking required but missing: " + ", ".join(missing))

import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

if TRACK_REMOTE:
    import wandb
    from huggingface_hub import HfApi

DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda":
    MODEL_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    MODEL_DTYPE = torch.float32

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
PAD = tok.pad_token_id

def problem():
    a, b = random.randint(11, 60), random.randint(11, 60)
    return f"{a} + {b}", a + b

def prompt_of(q):
    return tok.apply_chat_template(
        [{"role": "user", "content": f"Compute {q}. Reason briefly, then put the final integer after '####'."}],
        tokenize=False, add_generation_prompt=True)

def parse(text):
    m = re.findall(r"-?\d+", text.split("####")[-1])
    return int(m[0]) if m else None

def seq_logprob(model, pids, gen_row, grad):
    gen_row = gen_row[gen_row != PAD]
    if gen_row.numel() == 0:
        return None
    ids = torch.cat([pids, gen_row]).unsqueeze(0)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        logits = model(ids).logits[:, :-1, :].float()
        tgt = ids[:, 1:]
        lp = F.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        return lp[:, pids.shape[0] - 1:].sum()

def gen_group(model, prompt, gold, g):
    model.eval()
    enc = tok([prompt] * g, return_tensors="pt", padding=True).to(DEV)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=0.95,
                             max_new_tokens=MAX_NEW, pad_token_id=PAD)
    gens = out[:, enc.input_ids.shape[1]:]
    rewards = [1.0 if parse(t) == gold else 0.0 for t in tok.batch_decode(gens, skip_special_tokens=True)]
    pids = enc.input_ids[0]
    old = [seq_logprob(model, pids, gens[i], grad=False) for i in range(g)]
    return pids, gens, rewards, old

@torch.no_grad()
def heldout_acc(model, evalset):
    model.eval(); c = 0
    for q, gold in evalset:
        enc = tok([prompt_of(q)], return_tensors="pt", padding=True).to(DEV)
        out = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW, pad_token_id=PAD)
        if parse(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)) == gold:
            c += 1
    return c / len(evalset)

def advantages(rewards, arm):
    m = sum(rewards) / len(rewards)
    v = statistics.pvariance(rewards); s = v ** 0.5
    if v == 0.0:
        return None
    if arm == "drgrpo":
        return [r - m for r in rewards]                 # no /std
    return [(r - m) / (s + 1e-6) for r in rewards]      # grpo / dapo

def clipped_loss(new_lp, old_lp, adv, arm):
    ratio = torch.exp(new_lp - old_lp.detach())
    lo, hi = (0.2, 0.28) if arm == "dapo" else (0.2, 0.2)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - lo, 1 + hi) * adv
    return -torch.min(unclipped, clipped)


def tracking_config(arm, seed):
    return {
        "evidence_class": "pilot-not-confirmatory",
        "confirmatory_contract": "zvf-program/audit/preregistration.json",
        "arm": arm,
        "seed": seed,
        "model": MODEL,
        "batch": BATCH,
        "g0": G0,
        "gmax": GMAX,
        "max_new": MAX_NEW,
        "learning_rate": LR,
        "steps": STEPS,
        "inner": INNER,
        "heldout_n": HELDOUT_N,
        "unit_fingerprint": ARGS.unit_fingerprint,
        "device": DEV,
        "dtype": str(MODEL_DTYPE),
    }


def upload_checkpoint(model, optimizer, result, tracking_run):
    if not TRACK_REMOTE:
        return {}
    with tempfile.TemporaryDirectory(prefix="e3-checkpoint-") as tmp:
        checkpoint_dir = Path(tmp)
        model.save_pretrained(checkpoint_dir, safe_serialization=True, max_shard_size="2GB")
        tok.save_pretrained(checkpoint_dir)
        state = {
            "schema_version": "e3-open-audit-final-checkpoint-v1",
            "evidence_class": "pilot-not-confirmatory",
            "unit_fingerprint": ARGS.unit_fingerprint,
            "result": result,
            "run_config": tracking_config(result["arm"], result["seed"]),
            "optimizer_class": type(optimizer).__name__,
            "optimizer_state_included": False,
            "optimizer_state_note": "Final evaluation checkpoint; optimizer tensors omitted to bound artifact size.",
            "wandb_run_id": tracking_run.id,
            "wandb_run_url": tracking_run.url,
        }
        manifest_path = checkpoint_dir / "run_manifest.json"
        manifest_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(repo_id=ARGS.hf_repo, repo_type="model", private=True, exist_ok=True)
        commit = api.upload_folder(
            repo_id=ARGS.hf_repo,
            repo_type="model",
            folder_path=checkpoint_dir,
            path_in_repo=ARGS.hf_path,
            commit_message=(
                f"E1 pilot {result['arm']} seed {result['seed']} "
                f"{ARGS.unit_fingerprint[:12]}"
            ),
        )
        hf_url = f"https://huggingface.co/{ARGS.hf_repo}/tree/main/{ARGS.hf_path}"

        artifact = wandb.Artifact(
            name=f"e1-pilot-manifest-{result['arm']}-s{result['seed']}",
            type="run-manifest",
            metadata={
                "unit_fingerprint": ARGS.unit_fingerprint,
                "hf_repo": ARGS.hf_repo,
                "hf_path": ARGS.hf_path,
                "hf_commit": commit.oid,
            },
        )
        artifact.add_file(str(manifest_path), name="run_manifest.json")
        tracking_run.log_artifact(artifact).wait()
        return {
            "hf_repo": ARGS.hf_repo,
            "hf_path": ARGS.hf_path,
            "hf_commit": commit.oid,
            "hf_checkpoint_url": hf_url,
            "wandb_run_id": tracking_run.id,
            "wandb_run_url": tracking_run.url,
        }

def run(arm, seed):
    random.seed(seed); torch.manual_seed(seed)
    tracking_run = None
    if TRACK_REMOTE:
        tracking_run = wandb.init(
            project=ARGS.wandb_project,
            entity=ARGS.wandb_entity,
            group=ARGS.wandb_group,
            name=ARGS.wandb_run_name,
            job_type=arm,
            tags=["colab", "e1-pilot", arm, "pilot-not-confirmatory"],
            config=tracking_config(arm, seed),
            mode="online",
        )
    evalset = [problem() for _ in range(HELDOUT_N)]
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=MODEL_DTYPE).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        pre = heldout_acc(model, evalset)
        g, recent_zvf, traj, rollouts = G0, [], [], 0
        if tracking_run:
            tracking_run.log({"eval/heldout_pre": pre}, step=0)
        for step in range(1, STEPS + 1):
            if arm == "grpo_adaptiveG" and len(recent_zvf) >= 2 and statistics.mean(recent_zvf[-2:]) > 0.4:
                g = min(GMAX, g + 2)                         # zvf-triage: escalate G under collapse
            groups, zv = [], 0
            for _ in range(BATCH):
                q, gold = problem()
                tries = 0
                while True:
                    pids, gens, rewards, old = gen_group(model, prompt_of(q), gold, g)
                    rollouts += g
                    adv = advantages(rewards, arm)
                    if adv is not None or arm != "dapo" or tries >= 2:
                        break
                    tries += 1                               # DAPO dynamic sampling: resample dead group
                if adv is None:
                    zv += 1; continue
                groups.append((pids, gens, adv, old))
            for _ in range(INNER):
                opt.zero_grad(set_to_none=True)
                losses = []
                for pids, gens, adv, old in groups:
                    for i, a in enumerate(adv):
                        if a == 0:
                            continue
                        new = seq_logprob(model, pids, gens[i], grad=True)
                        if new is not None and old[i] is not None:
                            losses.append(clipped_loss(new, old[i], a, arm))
                if losses:
                    torch.stack(losses).sum().backward(); opt.step()
            zvf = zv / BATCH
            recent_zvf.append(zvf)
            # rough mean reward proxy: recompute correctness count from last groups not stored; track via zvf-comp
            traj.append({"step": step, "zvf": round(zvf, 3), "G": g})
            if tracking_run:
                tracking_run.log({
                    "train/zvf": zvf,
                    "train/group_size": g,
                    "train/rollouts_cumulative": rollouts,
                    "train/nonzero_groups": len(groups),
                }, step=step)
            print(f"[e3:{arm[:8]:8s} s{seed}] step={step:2d} G={g} ZVF={zvf:.2f}", flush=True)
        post = heldout_acc(model, evalset)
        out = {"arm": arm, "seed": seed, "heldout_pre": round(pre, 3), "heldout_post": round(post, 3),
               "heldout_delta": round(post - pre, 3), "mean_zvf": round(statistics.mean(t["zvf"] for t in traj), 3),
               "final_G": g, "total_rollouts": rollouts}
        remote = upload_checkpoint(model, opt, out, tracking_run) if tracking_run else {}
        out.update(remote)
        if tracking_run:
            tracking_run.summary.update({
                "eval/heldout_post": post,
                "eval/heldout_delta": post - pre,
                "train/mean_zvf": out["mean_zvf"],
                "train/final_group_size": g,
                "train/total_rollouts": rollouts,
                **remote,
            })
            tracking_run.finish(exit_code=0)
        del model, opt; torch.cuda.empty_cache()
        return out
    except BaseException:
        if tracking_run:
            tracking_run.finish(exit_code=1)
        raise

results = []
for arm in ARMS:
    for seed in SEEDS:
        results.append(run(arm, seed))

agg = {}
for arm in ARMS:
    rs = [r for r in results if r["arm"] == arm]
    agg[arm] = {"mean_heldout_delta": round(statistics.mean(r["heldout_delta"] for r in rs), 3),
                "mean_zvf": round(statistics.mean(r["mean_zvf"] for r in rs), 3),
                "mean_rollouts": round(statistics.mean(r["total_rollouts"] for r in rs), 1)}
print("E3_RESULT " + json.dumps({"experiment": "E3_open_audit", "model": MODEL,
                                 "seeds": SEEDS, "arms": ARMS,
                                 "run_config": {
                                     "device": DEV, "dtype": str(MODEL_DTYPE),
                                     "batch": BATCH, "g0": G0, "gmax": GMAX,
                                     "max_new": MAX_NEW, "learning_rate": LR,
                                     "steps": STEPS, "inner": INNER,
                                     "heldout_n": HELDOUT_N,
                                 },
                                 "units": results, "by_arm": agg}), flush=True)
