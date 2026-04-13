"""
Modal GPU Runner for TinkerRL Lab Experiments
==============================================
Runs all benchmark experiments on Modal's GPU infrastructure.
Uses ungated models (Qwen2.5 series) to avoid auth issues.

Usage:
  modal run scripts/modal_run_experiments.py
"""

import modal
import json
import os
import time

app = modal.App("tinkerrl-benchmark")
results_vol = modal.Volume.from_name("tinkerrl-results", create_if_missing=True)

llm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.3.0",
        "transformers>=4.46.0",
        "trl>=1.0.0",
        "datasets>=3.0.0",
        "accelerate>=1.0.0",
        "peft>=0.13.0",
        "safetensors>=0.4.0",
        "huggingface-hub>=0.26.0",
        "numpy>=1.26.0,<2.0.0",
        "scipy>=1.12.0",
    )
)

rl_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.3.0",
        "numpy>=1.26.0,<2.0.0",
        "scipy>=1.12.0",
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.3.0",
    )
)

SEEDS = [42, 123, 456, 789, 1024]
RESULTS_DIR = "/results"


# ===========================================================================
# Experiment 1: TRL GRPO Math (Arithmetic) — Qwen2.5-0.5B (ungated)
# ===========================================================================
@app.function(
    image=llm_image,
    gpu="L4",
    timeout=3600,
    volumes={RESULTS_DIR: results_vol},
    retries=1,
)
def run_grpo_math(seed: int) -> dict:
    """Run TRL GRPO on arithmetic task with Qwen2.5-0.5B."""
    import re
    import random
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOTrainer, GRPOConfig

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_name = "Qwen/Qwen2.5-0.5B"  # Ungated
    print(f"[GRPO Math] seed={seed}, model={model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )

    # Generate dataset
    problems = []
    for _ in range(500):
        a = random.randint(1, 99)
        b = random.randint(1, 99)
        problems.append({
            "prompt": f"What is {a} + {b}? Answer with just the number.",
            "answer": str(a + b),
        })
    dataset = Dataset.from_list(problems)

    # Reward function — TRL passes list of completions as strings or list of dicts
    def reward_fn(completions, prompts=None, **kwargs):
        rewards = []
        for comp in completions:
            if isinstance(comp, list):  # chat format: list of {role, content}
                text = comp[-1]["content"] if comp else ""
            elif isinstance(comp, dict):
                text = comp.get("content", str(comp))
            else:
                text = str(comp)
            numbers = re.findall(r'\b\d+\b', text)
            rewards.append(1.0 if numbers else 0.0)
        return rewards

    from peft import LoraConfig

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    grpo_config = GRPOConfig(
        output_dir=f"/tmp/grpo_math_seed{seed}",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=4,
        beta=0.1,
        learning_rate=1e-4,
        max_completion_length=8,
        temperature=1.0,
        num_train_epochs=1,
        logging_steps=5,
        save_steps=999999,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        peft_config=peft_config,
    )

    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0

    # Evaluate
    model.eval()
    correct = 0
    total = 200
    for _ in range(total):
        a = random.randint(1, 99)
        b = random.randint(1, 99)
        prompt = f"What is {a} + {b}? Answer with just the number."
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, temperature=0.1, do_sample=True)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        numbers = re.findall(r'\b\d+\b', text)
        if numbers and int(numbers[-1]) == a + b:
            correct += 1

    accuracy = correct / total
    result = {
        "experiment": "trl_grpo_math",
        "seed": seed,
        "final_accuracy": accuracy,
        "train_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else 0,
        "train_steps": train_result.global_step if hasattr(train_result, 'global_step') else 0,
        "elapsed_seconds": elapsed,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "model": model_name,
    }

    os.makedirs(f"{RESULTS_DIR}/grpo_math", exist_ok=True)
    with open(f"{RESULTS_DIR}/grpo_math/seed{seed}.json", "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()

    print(f"[GRPO Math] seed={seed} → accuracy={accuracy:.3f}, elapsed={elapsed:.0f}s")
    return result


# ===========================================================================
# Experiment 2: SB3 PPO Math
# ===========================================================================
@app.function(image=rl_image, gpu="T4", timeout=600, volumes={RESULTS_DIR: results_vol})
def run_sb3_math(seed: int) -> dict:
    """Run Stable Baselines3 PPO on arithmetic environment."""
    import random
    import numpy as np
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback

    random.seed(seed)
    np.random.seed(seed)

    class ArithmeticEnv(gym.Env):
        def __init__(self, max_num=99):
            super().__init__()
            self.max_num = max_num
            self.observation_space = spaces.Box(low=0, high=max_num, shape=(2,), dtype=np.float32)
            self.action_space = spaces.Discrete(max_num * 2 + 1)
            self.a, self.b = 0, 0
        def reset(self, seed=None, **kw):
            super().reset(seed=seed)
            self.a = np.random.randint(1, self.max_num + 1)
            self.b = np.random.randint(1, self.max_num + 1)
            return np.array([self.a, self.b], dtype=np.float32), {}
        def step(self, action):
            correct = self.a + self.b
            r = 1.0 if int(action) == correct else 0.0
            return np.array([self.a, self.b], dtype=np.float32), r, True, False, {"correct": int(action)==correct}

    class AccuracyTracker(BaseCallback):
        def __init__(self): super().__init__(); self.accuracies = []; self.step_at_95 = None
        def _on_step(self):
            if self.n_calls % 2048 == 0:
                env = ArithmeticEnv()
                c = 0
                for _ in range(200):
                    o, _ = env.reset()
                    a, _ = self.model.predict(o, deterministic=True)
                    _, _, _, _, i = env.step(a)
                    if i["correct"]: c += 1
                acc = c / 200
                self.accuracies.append((self.num_timesteps, acc))
                if acc >= 0.95 and self.step_at_95 is None:
                    self.step_at_95 = self.num_timesteps
            return True

    env = DummyVecEnv([lambda: ArithmeticEnv()])
    model = PPO("MlpPolicy", env, learning_rate=1e-4, n_steps=2048, batch_size=64, n_epochs=10, seed=seed, verbose=0)
    tracker = AccuracyTracker()

    t0 = time.time()
    model.learn(total_timesteps=100_000, callback=tracker)
    elapsed = time.time() - t0

    # Final eval
    test_env = ArithmeticEnv()
    correct = 0
    for _ in range(1000):
        obs, _ = test_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, _, _, _, info = test_env.step(action)
        if info["correct"]: correct += 1

    accuracy = correct / 1000
    result = {
        "experiment": "sb3_ppo_math", "seed": seed, "final_accuracy": accuracy,
        "steps_to_95": tracker.step_at_95, "elapsed_seconds": elapsed,
        "learning_curve": tracker.accuracies,
    }

    os.makedirs(f"{RESULTS_DIR}/sb3_math", exist_ok=True)
    with open(f"{RESULTS_DIR}/sb3_math/seed{seed}.json", "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    print(f"[SB3 PPO] seed={seed} → accuracy={accuracy:.3f}, steps_to_95={tracker.step_at_95}")
    return result


# ===========================================================================
# Experiment 3: CleanRL PPO Math
# ===========================================================================
@app.function(image=rl_image, gpu="T4", timeout=600, volumes={RESULTS_DIR: results_vol})
def run_cleanrl_math(seed: int) -> dict:
    """Run CleanRL-style PPO on arithmetic environment."""
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions.categorical import Categorical

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_num = 99
    max_answer = max_num * 2 + 1

    class Agent(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
            self.actor = nn.Linear(256, max_answer)
            self.critic = nn.Linear(256, 1)
        def get_value(self, x): return self.critic(self.net(x))
        def get_action_and_value(self, x, action=None):
            h = self.net(x); logits = self.actor(h); probs = Categorical(logits=logits)
            if action is None: action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), self.critic(h)

    agent = Agent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=1e-4, eps=1e-5)

    num_steps, num_envs, total_timesteps = 128, 8, 100_000
    minibatch_size, update_epochs, clip_coef = 256, 4, 0.2
    gamma, gae_lambda = 0.99, 0.95
    num_updates = total_timesteps // (num_steps * num_envs)

    accuracies = []; step_at_95 = None; global_step = 0
    t0 = time.time()

    for update in range(1, num_updates + 1):
        obs_buf = torch.zeros((num_steps, num_envs, 2)).to(device)
        act_buf = torch.zeros((num_steps, num_envs)).to(device)
        logp_buf = torch.zeros((num_steps, num_envs)).to(device)
        rew_buf = torch.zeros((num_steps, num_envs)).to(device)
        val_buf = torch.zeros((num_steps, num_envs)).to(device)

        for step in range(num_steps):
            a = np.random.randint(1, max_num+1, size=num_envs)
            b = np.random.randint(1, max_num+1, size=num_envs)
            obs = torch.FloatTensor(np.stack([a,b], axis=1)).to(device)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs)
            rewards = (action.cpu().numpy() == a+b).astype(np.float32)
            obs_buf[step]=obs; act_buf[step]=action; logp_buf[step]=logprob
            rew_buf[step]=torch.FloatTensor(rewards).to(device); val_buf[step]=value.flatten()
            global_step += num_envs

        with torch.no_grad():
            advantages = torch.zeros_like(rew_buf); lastgaelam = 0
            for t in reversed(range(num_steps)):
                nv = val_buf[t+1] if t+1 < num_steps else torch.zeros(num_envs).to(device)
                delta = rew_buf[t] + gamma*nv - val_buf[t]
                advantages[t] = lastgaelam = delta + gamma*gae_lambda*lastgaelam
            returns = advantages + val_buf

        b_obs=obs_buf.reshape(-1,2); b_act=act_buf.reshape(-1); b_logp=logp_buf.reshape(-1)
        b_adv=advantages.reshape(-1); b_ret=returns.reshape(-1)
        batch_size=num_steps*num_envs; b_inds=np.arange(batch_size)

        for _ in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb = b_inds[start:start+minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb], b_act[mb].long())
                ratio = (newlogprob - b_logp[mb]).exp()
                mb_adv = b_adv[mb]; mb_adv = (mb_adv - mb_adv.mean())/(mb_adv.std()+1e-8)
                loss = torch.max(-mb_adv*ratio, -mb_adv*torch.clamp(ratio,1-clip_coef,1+clip_coef)).mean()
                loss += 0.5*((newvalue.flatten()-b_ret[mb])**2).mean() - 0.01*entropy.mean()
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5); optimizer.step()

        # Track accuracy every 5 updates
        if update % 5 == 0:
            agent.eval(); c = 0
            for _ in range(200):
                aa = np.random.randint(1, max_num+1); bb = np.random.randint(1, max_num+1)
                o = torch.FloatTensor([[aa,bb]]).to(device)
                with torch.no_grad(): act, _, _, _ = agent.get_action_and_value(o)
                if act.item() == aa+bb: c += 1
            acc = c/200; accuracies.append((global_step, acc))
            if acc >= 0.95 and step_at_95 is None: step_at_95 = global_step
            agent.train()

    elapsed = time.time() - t0

    # Final eval
    agent.eval(); correct = 0
    for _ in range(1000):
        a = np.random.randint(1, max_num+1); b = np.random.randint(1, max_num+1)
        o = torch.FloatTensor([[a,b]]).to(device)
        with torch.no_grad(): act, _, _, _ = agent.get_action_and_value(o)
        if act.item() == a+b: correct += 1

    accuracy = correct/1000
    result = {
        "experiment": "cleanrl_ppo_math", "seed": seed, "final_accuracy": accuracy,
        "steps_to_95": step_at_95, "elapsed_seconds": elapsed,
        "learning_curve": accuracies,
    }

    os.makedirs(f"{RESULTS_DIR}/cleanrl_math", exist_ok=True)
    with open(f"{RESULTS_DIR}/cleanrl_math/seed{seed}.json", "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    print(f"[CleanRL PPO] seed={seed} → accuracy={accuracy:.3f}, steps_to_95={step_at_95}")
    return result


# ===========================================================================
# Experiment 4: Tianshou PPO Math (manual impl for compat)
# ===========================================================================
@app.function(image=rl_image, gpu="T4", timeout=600, volumes={RESULTS_DIR: results_vol})
def run_tianshou_math(seed: int) -> dict:
    """Run PPO on arithmetic (Tianshou-style architecture)."""
    import random
    import numpy as np
    import torch
    import torch.nn as nn

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_num = 99; max_answer = max_num*2+1

    # Tianshou-style network (wider)
    class TianshouAgent(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2,512), nn.ReLU(), nn.Linear(512,512), nn.ReLU(), nn.Linear(512,256), nn.ReLU())
            self.actor = nn.Linear(256, max_answer)
            self.critic = nn.Linear(256, 1)
        def forward(self, x):
            h = self.net(x)
            return self.actor(h), self.critic(h)

    agent = TianshouAgent().to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)

    accuracies = []; step_at_95 = None; global_step = 0
    t0 = time.time()

    for epoch in range(100):
        batch_size = 1000
        a = np.random.randint(1, max_num+1, size=batch_size)
        b = np.random.randint(1, max_num+1, size=batch_size)
        obs = torch.FloatTensor(np.stack([a,b], axis=1)).to(device)
        answers = a + b

        logits, values = agent(obs)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        rewards = (actions.cpu().numpy() == answers).astype(np.float32)
        rewards_t = torch.FloatTensor(rewards).to(device)
        advantages = rewards_t - values.squeeze()

        # PPO-style loss
        ratio = torch.ones_like(log_probs)  # First pass
        pg_loss = -(advantages.detach() * log_probs).mean()
        v_loss = 0.5 * ((values.squeeze() - rewards_t) ** 2).mean()
        entropy_loss = -dist.entropy().mean()
        loss = pg_loss + 0.5 * v_loss + 0.01 * entropy_loss

        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()
        global_step += batch_size

        # Track accuracy
        if (epoch + 1) % 5 == 0:
            agent.eval()
            c = 0
            for _ in range(200):
                aa = np.random.randint(1, max_num+1); bb = np.random.randint(1, max_num+1)
                o = torch.FloatTensor([[aa,bb]]).to(device)
                with torch.no_grad(): lo, _ = agent(o)
                if lo.argmax().item() == aa+bb: c += 1
            acc = c/200; accuracies.append((global_step, acc))
            if acc >= 0.95 and step_at_95 is None: step_at_95 = global_step
            agent.train()

    elapsed = time.time() - t0

    # Final eval
    agent.eval(); correct = 0
    for _ in range(1000):
        aa = np.random.randint(1, max_num+1); bb = np.random.randint(1, max_num+1)
        o = torch.FloatTensor([[aa,bb]]).to(device)
        with torch.no_grad(): lo, _ = agent(o)
        if lo.argmax().item() == aa+bb: correct += 1

    accuracy = correct/1000
    result = {
        "experiment": "tianshou_ppo_math", "seed": seed, "final_accuracy": accuracy,
        "steps_to_95": step_at_95, "elapsed_seconds": elapsed,
        "learning_curve": accuracies,
    }

    os.makedirs(f"{RESULTS_DIR}/tianshou_math", exist_ok=True)
    with open(f"{RESULTS_DIR}/tianshou_math/seed{seed}.json", "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    print(f"[Tianshou PPO] seed={seed} → accuracy={accuracy:.3f}, steps_to_95={step_at_95}")
    return result


# ===========================================================================
# Main entrypoint — fan out all experiments in parallel
# ===========================================================================
@app.local_entrypoint()
def main():
    import numpy as np

    print("=" * 60)
    print("TinkerRL Lab — Running Benchmark Experiments on Modal")
    print(f"Seeds: {SEEDS}")
    print(f"Total jobs: {len(SEEDS) * 4}")
    print("=" * 60)

    # Launch classic RL experiments first (simpler, faster)
    print("\n[1/4] SB3 PPO Math ...")
    sb3_results = list(run_sb3_math.map(SEEDS))

    print("\n[2/4] CleanRL PPO Math ...")
    cleanrl_results = list(run_cleanrl_math.map(SEEDS))

    print("\n[3/4] Tianshou PPO Math ...")
    tianshou_results = list(run_tianshou_math.map(SEEDS))

    print("\n[4/4] GRPO Math (Qwen2.5-0.5B) ...")
    grpo_results = list(run_grpo_math.map(SEEDS))

    all_results = {
        "trl_grpo_math": grpo_results,
        "sb3_ppo_math": sb3_results,
        "cleanrl_ppo_math": cleanrl_results,
        "tianshou_ppo_math": tianshou_results,
    }

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for exp_name, results in all_results.items():
        accs = [r["final_accuracy"] for r in results if "final_accuracy" in r]
        steps = [r.get("steps_to_95") for r in results if r.get("steps_to_95") is not None]
        if accs:
            mean_acc = np.mean(accs)
            se_acc = np.std(accs) / np.sqrt(len(accs))
            ci_lo = mean_acc - 1.96 * se_acc
            ci_hi = mean_acc + 1.96 * se_acc
            steps_str = f", steps_to_95={np.mean(steps):.0f}" if steps else ""
            print(f"  {exp_name}: {mean_acc:.3f} ± {se_acc:.3f}  95%CI=[{ci_lo:.3f}, {ci_hi:.3f}]{steps_str}")

    # Save combined results
    with open("modal_results_all.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nResults saved to modal_results_all.json")
