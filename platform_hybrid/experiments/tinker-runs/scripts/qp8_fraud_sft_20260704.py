#!/usr/bin/env python3
"""qp8-fraud-sft: supervised fine-tune a small Tinker model on serialized fraud rows.

Data comes from platform_local/train_xgboost.py's synthetic generator (train_data.csv / test_data.csv).
Rows are serialized as "field: value" lines -> label yes/no; 1 epoch SFT via
cross_entropy loss; eval accuracy/AUC by sampling (k votes -> P(yes) score).

Usage:
  python3 qp8_fraud_sft_20260704.py --smoke
  python3 qp8_fraud_sft_20260704.py
"""
import argparse, csv, json, os, random, sys, time

import tinker
import tinker.types as T

DATA_DIR = "/tmp/claude-1001/-home-claude-tinker-rl-lab/30780cda-8d14-4b43-8179-c46d6b7ba4dc/scratchpad"
OUT_DIR = "/home/claude/tinker-rl-lab/experiments/results/quick_20260704"
EXP = "qp8-fraud-sft"
SYS_PROMPT = ("You are a fraud detection assistant. Given the transaction features, "
              "answer with exactly one word: yes if the transaction is fraudulent, no otherwise.")


def load_rows(path, limit=None):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            label = int(float(row.pop("Class")))
            feats = [(k, float(v)) for k, v in row.items()]
            rows.append((feats, label))
            if limit and len(rows) >= limit:
                break
    return rows


def serialize(feats):
    body = "\n".join(f"{k}: {v:.3f}" for k, v in feats)
    return (f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nTransaction:\n{body}\nIs this transaction fraudulent? "
            f"Answer yes or no.<|im_end|>\n<|im_start|>assistant\n")


def auc_score(labels, scores):
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # rank-sum AUC with tie handling
    ranks, i = {}, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # average of 1-based ranks i+1..j
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    rank_sum_pos = sum(ranks[k] for k in range(len(pairs)) if pairs[k][1] == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate(sc, tok, rows, k, max_par=64):
    sp = T.SamplingParams(max_tokens=4, temperature=1.0, top_p=0.95)
    labels, scores, correct = [], [], 0
    for start in range(0, len(rows), max_par):
        chunk = rows[start:start + max_par]
        futs = [sc.sample(T.ModelInput.from_ints(tok.encode(serialize(f), add_special_tokens=False)),
                          num_samples=k, sampling_params=sp) for f, _ in chunk]
        for (feats, label), fut in zip(chunk, futs):
            try:
                resp = fut.result()
                yes = 0
                for s in resp.sequences:
                    txt = tok.decode(list(s.tokens), skip_special_tokens=True).strip().lower()
                    if txt.startswith("yes"):
                        yes += 1
                score = yes / max(1, len(resp.sequences))
            except Exception as e:
                print(f"eval sample failed: {e}"); score = 0.0
            labels.append(label); scores.append(score)
            if (score >= 0.5) == (label == 1):
                correct += 1
    return correct / len(rows), auc_score(labels, scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--train-rows", type=int, default=2000)
    ap.add_argument("--eval-rows", type=int, default=500)
    ap.add_argument("--mid-eval-rows", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=16)
    ap.add_argument("--votes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.train_rows, args.eval_rows, args.mid_eval_rows = 64, 16, 16
        args.batch, args.eval_every, args.votes = 32, 1, 4

    os.makedirs(OUT_DIR, exist_ok=True)
    step_tsv = os.path.join(OUT_DIR, f"{EXP}.tsv")
    final_tsv = os.path.join(OUT_DIR, "qp8_fraud.tsv")
    if not os.path.exists(step_tsv):
        with open(step_tsv, "w") as f:
            f.write("step\tphase\tloss\tn\taccuracy\tauc\telapsed_s\n")
    if not os.path.exists(final_tsv):
        with open(final_tsv, "w") as f:
            f.write("model\tsplit\tn\taccuracy\tauc\n")

    train_all = load_rows(os.path.join(DATA_DIR, "train_data.csv"), limit=20000)
    test_all = load_rows(os.path.join(DATA_DIR, "test_data.csv"), limit=10000)
    rng = random.Random(args.seed)

    # Balanced training subset (<= train_rows): all positives up to half, rest negatives
    pos = [r for r in train_all if r[1] == 1]
    neg = [r for r in train_all if r[1] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    n_pos = min(len(pos), args.train_rows // 2)
    train = pos[:n_pos] + neg[:args.train_rows - n_pos]
    rng.shuffle(train)
    # Held-out eval: positive-enriched slice of test set (AUC is prevalence-invariant;
    # natural 1-2% fraud rate would give only ~4 positives in 500 rows)
    tpos = [r for r in test_all if r[1] == 1]
    tneg = [r for r in test_all if r[1] == 0]
    rng.shuffle(tpos); rng.shuffle(tneg)
    ep = min(len(tpos), args.eval_rows // 5)
    test_rows = tpos[:ep] + tneg[:args.eval_rows - ep]
    rng.shuffle(test_rows)
    mp = min(len(tpos), args.mid_eval_rows // 5)
    mid_rows = tpos[:mp] + tneg[:args.mid_eval_rows - mp]
    rng.shuffle(mid_rows)
    print(f"train={len(train)} (pos={n_pos})  eval={len(test_rows)} "
          f"(pos={sum(l for _, l in test_rows)})", flush=True)

    run_name = f"{EXP}-20260704" + ("-smoke" if args.smoke else "")
    wandb = None
    if not args.no_wandb:
        import wandb as _w; wandb = _w
        wandb.init(project="tinker-new-research", name=run_name, config=vars(args),
                   tags=["qp8", "fraud-sft", "20260704"])

    svc = tinker.ServiceClient(base_url=None)
    tc = svc.create_lora_training_client(base_model=args.model, rank=args.rank)
    tok = tc.get_tokenizer()

    manifest = {
        "loss_form": "token-level cross_entropy (Tinker loss_fn='cross_entropy'), weights=0 on prompt, 1 on answer tokens",
        "ref_policy_kl_handling": "none (pure SFT, no reference policy / KL term)",
        "sampler_backend_precision": "unknown/closed-stack (Tinker managed sampler)",
        "per_step_zvf_path": "n/a (SFT, no group rollouts); per-step loss in " + step_tsv,
        "group_size_schedule": f"n/a for SFT; eval voting k={args.votes} fixed",
        "heldout_split": f"test_data.csv from platform_local/train_xgboost.py split (random_state=42, stratified); eval subset positive-enriched to 20% fraud ({args.eval_rows} rows) since natural rate ~1.4% gives too few positives for stable AUC",
        "decontamination_notes": "synthetic data (sklearn make_classification seed 42); no overlap with pretraining corpora possible; train/test split disjoint by construction",
    }
    with open(os.path.join(OUT_DIR, f"{EXP}_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    t0 = time.time()
    n_steps = (len(train) + args.batch - 1) // args.batch
    adam = T.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)

    def log_row(step, phase, loss, n, acc, auc):
        with open(step_tsv, "a") as f:
            f.write(f"{step}\t{phase}\t{loss}\t{n}\t{acc}\t{auc}\t{time.time()-t0:.1f}\n")

    def run_eval(step, rows, tag):
        w = tc.save_weights_for_sampler(name=f"eval_{step}").result()
        sc = tc.create_sampling_client(model_path=w.path)
        acc, auc = evaluate(sc, tok, rows, args.votes)
        log_row(step, tag, "", len(rows), f"{acc:.4f}", f"{auc:.4f}")
        if wandb: wandb.log({f"{tag}/accuracy": acc, f"{tag}/auc": auc}, step=step)
        print(f"[{tag}] step {step}: acc={acc:.4f} auc={auc:.4f}", flush=True)
        return acc, auc

    run_eval(0, mid_rows, "eval_mid")  # pre-training baseline

    for step in range(n_steps):
        batch = train[step * args.batch:(step + 1) * args.batch]
        data = []
        for feats, label in batch:
            prompt_ids = tok.encode(serialize(feats), add_special_tokens=False)
            ans_ids = tok.encode(("yes" if label else "no") + "<|im_end|>", add_special_tokens=False)
            full = prompt_ids + ans_ids
            weights = [0.0] * (len(prompt_ids) - 1) + [1.0] * len(ans_ids)
            data.append(T.Datum(
                model_input=T.ModelInput.from_ints(full[:-1]),
                loss_fn_inputs={
                    "target_tokens": T.TensorData(data=full[1:], dtype="int64", shape=[len(full) - 1]),
                    "weights": T.TensorData(data=weights, dtype="float32", shape=[len(weights)]),
                }))
        res = tc.forward_backward(data=data, loss_fn="cross_entropy").result()
        tc.optim_step(adam).result()
        try:
            loss = res.metrics.get("loss:sum", None) or res.metrics.get("loss", None)
        except AttributeError:
            loss = getattr(res.metrics, "loss", None)
        log_row(step + 1, "train", loss, len(batch), "", "")
        if wandb: wandb.log({"train/loss": loss}, step=step + 1)
        print(f"[train] {step+1}/{n_steps} loss={loss}", flush=True)
        if (step + 1) % args.eval_every == 0 and step + 1 < n_steps:
            run_eval(step + 1, mid_rows, "eval_mid")

    acc, auc = run_eval(n_steps, test_rows, "eval_final")
    with open(final_tsv, "a") as f:
        f.write(f"{args.model}-sft\ttest_enriched20pct\t{len(test_rows)}\t{acc:.4f}\t{auc:.4f}\n")
    if wandb: wandb.finish()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
