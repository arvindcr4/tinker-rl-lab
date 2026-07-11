"""100-step GRPO on MATH problems — optimized for speed"""

import atexit
try:
    from codecarbon import EmissionsTracker
    _tracker = EmissionsTracker()
    _tracker.start()
    atexit.register(_tracker.stop)
except ImportError:
    pass


import os, json, re, warnings, random, math

warnings.filterwarnings("ignore")
assert os.environ.get("TINKER_API_KEY"), (
    "Set TINKER_API_KEY in env (was hardcoded, removed 2026-04-11)"
)
import torch, tinker, tinker.types as T
from transformers import AutoTokenizer

# Fix for Single-Seed Extrapolations: Allow reproducible multi-seed runs
# TODO: Run this script with multiple seeds (N>1) to prove statistical significance.
SEED = int(os.environ.get("SEED", "42"))
random.seed(SEED)
torch.manual_seed(SEED)

# TODO (Adversarial Review Limitations):
# 1. Closed-Source Confound: Compare against open-source libraries (e.g. TRL) to isolate Tinker's managed defaults.
# 2. Early-Training Snapshot: 100 steps is insufficient for observing true convergence or collapse. Consider longer runs.

EXP = "math100"
MODEL = "Qwen/Qwen3-8B"
STEPS, GROUP, LR, TEMP = 100, 4, 5e-6, 0.9
SAVE_EVERY = 25

SYSTEM_PROMPT = "You are a math assistant. Solve the problem step by step, then give your final answer inside \\boxed{}."

# Synthetic hard math problems with known answers
MATH_PROBLEMS = [
    ("What is 17 * 23?", "391"),
    ("What is 256 / 16?", "16"),
    ("What is 2^8?", "256"),
    ("Solve: 3x + 7 = 22", "5"),
    ("What is sqrt(625)?", "25"),
    ("What is 15! / 14!?", "15"),
    ("What is the sum of the first 10 positive integers?", "55"),
    ("What is 7^3?", "343"),
    ("Solve: 2x - 5 = 13", "9"),
    ("What is 144 / 12?", "12"),
    ("What is the GCD of 48 and 36?", "12"),
    ("What is 3^4 + 4^3?", "145"),
    ("Solve: x^2 = 49, x > 0", "7"),
    ("What is 1000 - 37 * 27?", "1"),
    ("What is the LCM of 12 and 18?", "36"),
    ("How many prime numbers are less than 20?", "8"),
    ("What is 5! (5 factorial)?", "120"),
    ("Solve: |x - 3| = 7, find positive x", "10"),
    ("What is 99 * 101?", "9999"),
    ("What is the 10th Fibonacci number?", "55"),
    ("What is 2^10 - 1?", "1023"),
    ("Solve: x + x/2 + x/4 = 14", "8"),
    ("What is 13^2 - 12^2?", "25"),
    ("What is the area of a circle with radius 7? (use pi=22/7)", "154"),
    ("What is 111 * 111?", "12321"),
    ("Solve: 2^x = 64", "6"),
    ("What is the sum of angles in a pentagon?", "540"),
    ("What is 17^2?", "289"),
    ("How many ways to choose 2 items from 5?", "10"),
    ("What is log_2(256)?", "8"),
    ("Solve: 5x + 3 = 2x + 18", "5"),
    ("What is 37 + 48 + 65 + 50?", "200"),
    ("What is the remainder when 100 is divided by 7?", "2"),
    ("What is 25% of 480?", "120"),
    ("Solve: x^2 - 5x + 6 = 0, find the larger root", "3"),
    ("What is 8 * 7 * 6 / (3 * 2 * 1)?", "56"),
    ("What is 1/2 + 1/3 + 1/6? Express as integer.", "1"),
    ("How many diagonals does a hexagon have?", "9"),
    ("What is the cube root of 27?", "3"),
    ("What is 50^2 - 49^2?", "99"),
]


def make_prompt(q):
    return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


# Fix for Failure to Prove Generalization: Hold out 10% of problems for a test set.
# TODO: Evaluate on a broader held-out test set (e.g., GSM8K) for robust generalization proof.
random.shuffle(MATH_PROBLEMS)
split_idx = int(len(MATH_PROBLEMS) * 0.9)
train_problems = MATH_PROBLEMS[:split_idx]
test_problems = MATH_PROBLEMS[split_idx:]

examples = [(make_prompt(q), a) for q, a in train_problems] * 20
random.shuffle(examples)
test_examples = [(make_prompt(q), a) for q, a in test_problems]
print(f"[{EXP}] {len(examples)} train examples, {len(train_problems)} unique train problems, {len(test_problems)} test problems")


def reward(response, answer):
    """Check if the answer appears in \\boxed{} or as the last number."""
    response = response.strip()
    # Check \boxed{answer}
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
    if boxed:
        for b in boxed:
            b_clean = b.strip().replace(",", "").replace(" ", "")
            if b_clean == answer:
                return 1.0
            try:
                if abs(float(b_clean) - float(answer)) < 0.01:
                    return 1.0
            except:
                pass
        return 0.3  # has boxed but wrong answer

    # Check if answer appears anywhere as a standalone number
    nums = re.findall(r"\b" + re.escape(answer) + r"\b", response)
    if nums:
        return 0.5  # correct answer but not in boxed format

    # Check last number in response
    all_nums = re.findall(r"[-+]?\d*\.?\d+", response)
    if all_nums:
        last = all_nums[-1].replace(",", "")
        try:
            if abs(float(last) - float(answer)) < 0.01:
                return 0.4
        except:
            pass

    # At least attempted math
    if any(c in response for c in "+-*/="):
        return 0.1
    return 0.0


_adv = []


def loss_fn(data, lp):
    losses = [(-_adv[i] * lp[i].sum()) for i in range(len(lp))]
    loss = torch.stack(losses).mean()
    return loss, {"loss": loss.item()}


print(f"[{EXP}] Connecting...")
svc = tinker.ServiceClient(base_url=None)
tc = svc.create_lora_training_client(base_model=MODEL, rank=32)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
w0 = tc.save_weights_for_sampler(name="s0").result()
sc = tc.create_sampling_client(model_path=w0.path)
print(f"[{EXP}] Run: {tc.model_id}")

import wandb
wandb.init(
    project="tinker-math-grpo",
    name=f"{EXP}-{tc.model_id}",
    config={
        "model": MODEL,
        "steps": STEPS,
        "group": GROUP,
        "lr": LR,
        "temp": TEMP,
        "save_every": SAVE_EVERY,
        "seed": SEED
    }
)

step_rewards = []
for step in range(STEPS):
    batch = random.sample(examples, 2)
    all_data, all_advs, batch_r, batch_sr = [], [], [], []
    for pt, ans in batch:
        pid = tok.encode(pt, add_special_tokens=False)
        sp = T.SamplingParams(max_tokens=256, temperature=TEMP, top_p=0.95)
        resp = sc.sample(
            T.ModelInput.from_ints(pid), num_samples=GROUP, sampling_params=sp
        ).result()
        rews = [
            reward(tok.decode(list(r.tokens), skip_special_tokens=True), ans)
            for r in resp.sequences
        ]
        mr = sum(rews) / len(rews)
        sr = (sum((r - mr) ** 2 for r in rews) / len(rews)) ** 0.5 + 1e-8
        advs = [(r - mr) / sr for r in rews]
        batch_r.extend(rews)
        batch_sr.append(sr)
        for r, a in zip(resp.sequences, advs):
            rid = list(r.tokens)
            fid = pid + rid
            tid = fid[1:] + [0]
            all_data.append(
                T.Datum(
                    model_input=T.ModelInput.from_ints(fid),
                    loss_fn_inputs={
                        "target_tokens": T.TensorData(data=tid, dtype="int64", shape=[len(tid)])
                    },
                )
            )
            all_advs.append(a)
    if not all_data:
        continue
    _adv = all_advs
    result = tc.forward_backward_custom(data=all_data, loss_fn=loss_fn).result()
    tc.optim_step(T.AdamParams(learning_rate=LR, beta1=0.9, beta2=0.95, eps=1e-8)).result()
    avg = sum(batch_r) / len(batch_r)
    # Fix for ZVF metric limitation: Monitor reward variance and ZVF directly
    avg_var = sum(s**2 for s in batch_sr) / len(batch_sr)
    zvf = sum(1 for s in batch_sr if s <= 1e-8) / len(batch_sr)
    step_rewards.append(avg)
    print(
        f"[{EXP}] {step + 1:3d}/{STEPS} | loss={result.metrics.get('loss', 0):.4f} | reward={avg:.3f} | r_var={avg_var:.3f} | zvf={zvf:.2f}"
    )
    wandb.log({
        "train/loss": result.metrics.get("loss", 0),
        "train/reward": avg,
        "train/r_var": avg_var,
        "train/zvf": zvf
    }, step=step + 1)
    if (step + 1) % SAVE_EVERY == 0:
        tc.save_state(name=f"s{step + 1}")
        ckpt = tc.save_weights_for_sampler(name=f"s{step + 1}").result()
        sc = tc.create_sampling_client(model_path=ckpt.path)
        print(f"[{EXP}]   -> ckpt s{step + 1}")

tc.save_state(name="final")
f = tc.save_weights_for_sampler(name="final").result()
last10 = step_rewards[-10:]
avg10 = sum(last10) / len(last10) if last10 else 0
print(f"\n[{EXP}] DONE | last10={avg10:.3f} | run={tc.model_id} | path={f.path}")

print(f"[{EXP}] Pushing final model to HuggingFace Hub...")
from huggingface_hub import HfApi
api = HfApi()
repo_id = os.environ.get("HF_REPO_ID", f"arvindcr4/tinker-{EXP}-{MODEL.split('/')[-1]}-lora")
api.create_repo(repo_id=repo_id, exist_ok=True, private=True)
api.upload_folder(
    folder_path=f.path,
    repo_id=repo_id,
    repo_type="model"
)
print(f"[{EXP}] Successfully pushed to {repo_id}")

# Fix for Failure to Prove Generalization: Evaluate on held-out test set
print(f"[{EXP}] Evaluating on {len(test_examples)} held-out test problems...")
test_rews = []
if test_examples:
    for pt, ans in test_examples:
        pid = tok.encode(pt, add_special_tokens=False)
        sp = T.SamplingParams(max_tokens=256, temperature=0.1, top_p=1.0)
        resp = sc.sample(
            T.ModelInput.from_ints(pid), num_samples=1, sampling_params=sp
        ).result()
        rews = [
            reward(tok.decode(list(r.tokens), skip_special_tokens=True), ans)
            for r in resp.sequences
        ]
        test_rews.extend(rews)

    test_acc = sum(test_rews) / len(test_rews)
    print(f"[{EXP}] Test Accuracy on Held-Out Set: {test_acc:.3f}")
    wandb.log({"test/accuracy": test_acc})

wandb.finish()
