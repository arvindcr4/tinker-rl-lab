import re

with open("platform_tinker/atropos/train_grpo_unsloth.py", "r") as f:
    code = f.read()

# 1. Imports and logging
code = code.replace(
"""import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import yaml""",
"""import argparse
import logging
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import List

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("train_grpo_unsloth")"""
)

# 2. _last_zvf global hack
code = code.replace("""except ImportError:
    pass

_last_zvf = 0.0


# ── reward helpers""", """except ImportError:
    pass


# ── reward helpers""")

# 3. unsloth print
code = code.replace("""            from unsloth import FastLanguageModel

            print(f"  Loading {model_name} with Unsloth ({params_b}B, 4-bit={load_in_4bit}) ...")""", """            from unsloth import FastLanguageModel

            logger.info(f"Loading {model_name} with Unsloth ({params_b}B, 4-bit={load_in_4bit}) ...")""")

# 4. unsloth except print
code = code.replace("""        except Exception as exc:
            print(f"  Unsloth load failed, falling back to Transformers/PEFT: {exc}")""", """        except Exception as exc:
            logger.warning(f"Unsloth load failed, falling back to Transformers/PEFT: {exc}")""")

# 5. PEFT print
code = code.replace("""        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"  Loading {model_name} with Transformers/PEFT ({params_b}B, 4-bit={load_in_4bit}) ...")""", """        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading {model_name} with Transformers/PEFT ({params_b}B, 4-bit={load_in_4bit}) ...")""")

# 6. make_reward_fn replacement
old_make_reward = """def make_reward_fn(group_size: int, evaluation_mode: str = "static"):
    \"\"\"Return a reward function compatible with TRL GRPOTrainer.\"\"\"

    def reward_fn(completions: List[str], prompts=None, **kwargs) -> List[float]:
        # GRPOTrainer passes gold answers via kwargs when the dataset has them.
        # We store gold_boxed in the dataset column and TRL surfaces it here.
        gold_list = kwargs.get("gold_boxed", None)
        rewards = []
        comp_texts = []
        for i, completion in enumerate(completions):
            if gold_list is not None:
                gold = gold_list[i] if isinstance(gold_list[i], str) else gold_list[i][0]
            else:
                gold = ""   # fallback (shouldn't happen)
            
            comp_text = _completion_to_text(completion)
            comp_texts.append(comp_text)

            # Outcome Reward Model (ORM) based on Evaluation Mode
            if evaluation_mode == "generative":
                # Extract prompt text
                prompt_text = ""
                if prompts is not None and len(prompts) > i:
                    if isinstance(prompts[i], str):
                        prompt_text = prompts[i]
                    elif isinstance(prompts[i], list):
                        prompt_text = prompts[i][-1].get("content", "") if isinstance(prompts[i][-1], dict) else str(prompts[i][-1])
                base_reward = _generative_score_response(prompt_text, comp_text, gold)
            elif evaluation_mode == "execution":
                base_reward = _execution_score_response(comp_text, gold)
            else:
                base_reward = _score_response(comp_text, gold)
            
            # Process Reward Model (PRM)
            text_lower = comp_text.lower()
            step_reward = 0.0
            steps_found = text_lower.count("step") + text_lower.count("first") + text_lower.count("then")
            if steps_found > 0:
                step_reward += min(0.5, 0.1 * steps_found)
                
            rewards.append(base_reward + step_reward)
        
        # calculate zvf, advantage variance, and ACR
        zvf_sum = 0.0
        advantage_variances = []
        n_groups = len(rewards) // group_size
        if n_groups > 0:
            for idx in range(n_groups):
                chunk = rewards[idx*group_size:(idx+1)*group_size]
                var = np.var(chunk)
                advantage_variances.append(var)
                mr = sum(chunk) / group_size
                if all(abs(r - mr) < 1e-6 for r in chunk):
                    zvf_sum += 1.0
                    
            global _last_zvf, _last_adv_var, _last_acr
            _last_zvf = zvf_sum / n_groups
            _last_adv_var = float(np.mean(advantage_variances))
            _last_acr = 1.0 if _last_adv_var < 1e-4 else 0.0
            
        # Length confounding panel
        lengths = [len(t) for t in comp_texts]
        correct_lens = [l for l, r in zip(lengths, rewards) if r > 0.5]
        incorrect_lens = [l for l, r in zip(lengths, rewards) if r <= 0.5]
        
        global _last_mean_len_correct, _last_mean_len_incorrect, _last_len_reward_corr
        _last_mean_len_correct = float(np.mean(correct_lens)) if correct_lens else 0.0
        _last_mean_len_incorrect = float(np.mean(incorrect_lens)) if incorrect_lens else 0.0
        
        if correct_lens and incorrect_lens:
            mean_y1 = _last_mean_len_correct
            mean_y0 = _last_mean_len_incorrect
            n1 = len(correct_lens)
            n0 = len(incorrect_lens)
            n = n1 + n0
            std_y = np.std(lengths)
            _last_len_reward_corr = float(((mean_y1 - mean_y0) / (std_y + 1e-8)) * np.sqrt((n1 * n0) / (n * (n - 1))))
        else:
            _last_len_reward_corr = 0.0
            
        return rewards

    return reward_fn"""

new_make_reward = """class StatefulRewardFunction:
    \"\"\"A stateful reward function compatible with TRL GRPOTrainer, tracking metrics thread-safely.\"\"\"
    
    def __init__(self, group_size: int, evaluation_mode: str = "static"):
        self.group_size = group_size
        self.evaluation_mode = evaluation_mode
        self.lock = threading.Lock()
        self.metrics = {}

    def __call__(self, completions: List[str], prompts=None, **kwargs) -> List[float]:
        # GRPOTrainer passes gold answers via kwargs when the dataset has them.
        # We store gold_boxed in the dataset column and TRL surfaces it here.
        gold_list = kwargs.get("gold_boxed", None)
        rewards = []
        comp_texts = []
        for i, completion in enumerate(completions):
            if gold_list is not None:
                gold = gold_list[i] if isinstance(gold_list[i], str) else gold_list[i][0]
            else:
                gold = ""   # fallback (shouldn't happen)
            
            comp_text = _completion_to_text(completion)
            comp_texts.append(comp_text)

            # Outcome Reward Model (ORM) based on Evaluation Mode
            if self.evaluation_mode == "generative":
                # Extract prompt text
                prompt_text = ""
                if prompts is not None and len(prompts) > i:
                    if isinstance(prompts[i], str):
                        prompt_text = prompts[i]
                    elif isinstance(prompts[i], list):
                        prompt_text = prompts[i][-1].get("content", "") if isinstance(prompts[i][-1], dict) else str(prompts[i][-1])
                base_reward = _generative_score_response(prompt_text, comp_text, gold)
            elif self.evaluation_mode == "execution":
                base_reward = _execution_score_response(comp_text, gold)
            else:
                base_reward = _score_response(comp_text, gold)
            
            # Process Reward Model (PRM)
            text_lower = comp_text.lower()
            step_reward = 0.0
            steps_found = text_lower.count("step") + text_lower.count("first") + text_lower.count("then")
            if steps_found > 0:
                step_reward += min(0.5, 0.1 * steps_found)
                
            rewards.append(base_reward + step_reward)
        
        # calculate zvf, advantage variance, and ACR
        zvf_sum = 0.0
        advantage_variances = []
        n_groups = len(rewards) // self.group_size
        if n_groups > 0:
            for idx in range(n_groups):
                chunk = rewards[idx*self.group_size:(idx+1)*self.group_size]
                var = np.var(chunk)
                advantage_variances.append(var)
                mr = sum(chunk) / self.group_size
                if all(abs(r - mr) < 1e-6 for r in chunk):
                    zvf_sum += 1.0
                    
            mean_adv_var = float(np.mean(advantage_variances))
            acr = 1.0 if mean_adv_var < 1e-4 else 0.0
            zvf = zvf_sum / n_groups
        else:
            mean_adv_var = 0.0
            acr = 0.0
            zvf = 0.0
            
        # Length confounding panel
        lengths = [len(t) for t in comp_texts]
        correct_lens = [l for l, r in zip(lengths, rewards) if r > 0.5]
        incorrect_lens = [l for l, r in zip(lengths, rewards) if r <= 0.5]
        
        mean_len_correct = float(np.mean(correct_lens)) if correct_lens else 0.0
        mean_len_incorrect = float(np.mean(incorrect_lens)) if incorrect_lens else 0.0
        
        if correct_lens and incorrect_lens:
            mean_y1 = mean_len_correct
            mean_y0 = mean_len_incorrect
            n1 = len(correct_lens)
            n0 = len(incorrect_lens)
            n = n1 + n0
            std_y = np.std(lengths)
            len_reward_corr = float(((mean_y1 - mean_y0) / (std_y + 1e-8)) * np.sqrt((n1 * n0) / (n * (n - 1))))
        else:
            len_reward_corr = 0.0

        with self.lock:
            self.metrics["zvf"] = zvf
            self.metrics["diagnostics/advantage_variance"] = mean_adv_var
            self.metrics["diagnostics/advantage_collapse_rate"] = acr
            self.metrics["diagnostics/mean_len_correct"] = mean_len_correct
            self.metrics["diagnostics/mean_len_incorrect"] = mean_len_incorrect
            self.metrics["diagnostics/length_reward_corr"] = len_reward_corr
            
        return rewards

    def get_metrics_and_reset(self) -> dict:
        with self.lock:
            metrics = self.metrics.copy()
            return metrics"""

if old_make_reward not in code:
    print("WARNING: make_reward_fn not found!")
code = code.replace(old_make_reward, new_make_reward)

# 7. StepLogger replacement
old_steplogger = """class StepLogger:
    \"\"\"Accumulates per-completion scores and logs step-level metrics.\"\"\"

    def __init__(self, run_name: str):
        self.run_name = run_name
        self.step_scores: list[float] = []
        self.step_log: list[float] = []   # per-step mean rewards

    def record(self, scores: List[float]):
        self.step_scores.extend(scores)

    def flush(self, step: int):
        if not self.step_scores:
            return
        mean_r = sum(self.step_scores) / len(self.step_scores)
        self.step_log.append(mean_r)
        metrics = {
            "train/percent_correct": mean_r,
            "train/step": step,
        }
        wandb.log(metrics, step=step)
        print(f"  step {step:3d}  mean_reward={mean_r:.4f}  "
              f"n={len(self.step_scores)}")
        self.step_scores = []

    def save_csv(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("step,mean_reward\\n")
            for i, r in enumerate(self.step_log):
                f.write(f"{i},{r:.6f}\\n")
        print(f"  Reward log saved → {path}")"""

new_steplogger = """class StepTracker:
    \"\"\"Accumulates per-completion scores and logs step-level metrics.\"\"\"

    def __init__(self, run_name: str):
        self.run_name = run_name
        self.step_scores: list[float] = []
        self.step_log: list[float] = []   # per-step mean rewards

    def record(self, scores: List[float]):
        self.step_scores.extend(scores)

    def flush(self, step: int):
        if not self.step_scores:
            return
        mean_r = sum(self.step_scores) / len(self.step_scores)
        self.step_log.append(mean_r)
        metrics = {
            "train/percent_correct": mean_r,
            "train/step": step,
        }
        wandb.log(metrics, step=step)
        logger.info(f"step {step:3d}  mean_reward={mean_r:.4f}  n={len(self.step_scores)}")
        self.step_scores = []

    def save_csv(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("step,mean_reward\\n")
            for i, r in enumerate(self.step_log):
                f.write(f"{i},{r:.6f}\\n")
        logger.info(f"Reward log saved → {path}")"""

if old_steplogger not in code:
    print("WARNING: StepLogger not found!")
code = code.replace(old_steplogger, new_steplogger)

# 8. maybe_push_to_hub print
code = code.replace("""        commit_message=f"Upload adapter for {cfg['model_name']} ({cfg['wandb_run_name']})",
    )
    print(f"  Hugging Face upload complete → {repo_id}")""", """        commit_message=f"Upload adapter for {cfg['model_name']} ({cfg['wandb_run_name']})",
    )
    logger.info(f"Hugging Face upload complete → {repo_id}")""")

# 9. train definition
old_train_def = """def train(config_path: str, seed: int = 42, wandb_api_key: str | None = None,
          total_token_budget: int | None = None, tokens_per_sample: int = 2048,
          group_size_override: int | None = None, batch_size_override: int | None = None):
    cfg = load_config(config_path)
    if group_size_override:
        cfg['group_size'] = group_size_override
    if batch_size_override:
        cfg['batch_size'] = batch_size_override
        
    if total_token_budget:
        total_samples = total_token_budget / tokens_per_sample
        effective_batch_size = cfg['group_size']
        cfg['total_steps'] = max(1, int(total_samples / effective_batch_size))
        print(f"[P1 Ablation] Matching token budget: {total_token_budget} tokens -> total_steps={cfg['total_steps']}")
        
    print(f"\\n{'='*60}")
    print(f"  GRPO (Unsloth) — {cfg['model_name']}")
    print(f"  Config: {config_path}")
    print(f"  Steps: {cfg['total_steps']}  |  batch: {cfg['batch_size']}  "
          f"|  group: {cfg['group_size']}  |  seed: {seed}")
    print(f"{'='*60}\\n")"""

new_train_def = """def train(config_path: str, seed: int = 42, wandb_api_key: str | None = None,
          total_token_budget: int | None = None, tokens_per_sample: int = 2048,
          group_size_override: int | None = None, batch_size_override: int | None = None,
          evaluation_mode_override: str | None = None):
    cfg = load_config(config_path)
    if group_size_override:
        cfg['group_size'] = group_size_override
    if batch_size_override:
        cfg['batch_size'] = batch_size_override
        
    if total_token_budget:
        total_samples = total_token_budget / tokens_per_sample
        effective_batch_size = cfg['group_size']
        cfg['total_steps'] = max(1, int(total_samples / effective_batch_size))
        logger.info(f"[P1 Ablation] Matching token budget: {total_token_budget} tokens -> total_steps={cfg['total_steps']}")
        
    logger.info("\\n" + "="*60)
    logger.info(f"GRPO (Unsloth) — {cfg['model_name']}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Steps: {cfg['total_steps']}  |  batch: {cfg['batch_size']}  |  group: {cfg['group_size']}  |  seed: {seed}")
    logger.info("="*60 + "\\n")"""

if old_train_def not in code:
    print("WARNING: train def not found!")
code = code.replace(old_train_def, new_train_def)

# 10. StepTracker init
code = code.replace("""        config={**cfg, "seed": seed, "config_file": config_path},
    )
    logger = StepLogger(cfg["wandb_run_name"])""", """        config={**cfg, "seed": seed, "config_file": config_path},
    )
    step_tracker = StepTracker(cfg["wandb_run_name"])""")

# 11. reward_fn init
code = code.replace("""        remove_unused_columns=False,
    )

    reward_fn = make_reward_fn(cfg["group_size"], evaluation_mode=cfg.get("evaluation_mode", "static"))

    from transformers import EarlyStoppingCallback""", """        remove_unused_columns=False,
    )

    eval_mode = evaluation_mode_override if evaluation_mode_override else cfg.get("evaluation_mode", "static")
    reward_fn = StatefulRewardFunction(cfg["group_size"], evaluation_mode=eval_mode)

    from transformers import EarlyStoppingCallback""")

# 12. RewardLogCallback
old_reward_callback = """    class RewardLogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            step = state.global_step
            # TRL logs reward/mean from the reward function
            mean_r = logs.get("reward/mean", logs.get("rewards/mean", None))
            if mean_r is not None:
                logger.step_log.append(float(mean_r))
                logger.flush(step)
            
            global _last_zvf, _last_adv_var, _last_acr, _last_mean_len_correct, _last_mean_len_incorrect, _last_len_reward_corr
            metrics_to_log = {}
            if '_last_zvf' in globals():
                metrics_to_log["zvf"] = _last_zvf
            if '_last_adv_var' in globals():
                metrics_to_log["diagnostics/advantage_variance"] = _last_adv_var
            if '_last_acr' in globals():
                metrics_to_log["diagnostics/advantage_collapse_rate"] = _last_acr
            if '_last_mean_len_correct' in globals():
                metrics_to_log["diagnostics/mean_len_correct"] = _last_mean_len_correct
            if '_last_mean_len_incorrect' in globals():
                metrics_to_log["diagnostics/mean_len_incorrect"] = _last_mean_len_incorrect
            if '_last_len_reward_corr' in globals():
                metrics_to_log["diagnostics/length_reward_corr"] = _last_len_reward_corr
                
            if metrics_to_log:
                wandb.log(metrics_to_log, step=step, commit=False)"""

new_reward_callback = """    class RewardLogCallback(TrainerCallback):
        def __init__(self, step_tracker, reward_fn: StatefulRewardFunction):
            self.step_tracker = step_tracker
            self.reward_fn = reward_fn

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            step = state.global_step
            # TRL logs reward/mean from the reward function
            mean_r = logs.get("reward/mean", logs.get("rewards/mean", None))
            if mean_r is not None:
                self.step_tracker.step_log.append(float(mean_r))
                self.step_tracker.flush(step)
            
            metrics_to_log = self.reward_fn.get_metrics_and_reset()
            if metrics_to_log:
                wandb.log(metrics_to_log, step=step, commit=False)"""

if old_reward_callback not in code:
    print("WARNING: RewardLogCallback not found!")
code = code.replace(old_reward_callback, new_reward_callback)

# 13. End of train replacements
old_end_train = """    trainer.add_callback(RewardLogCallback())
    trainer.add_callback(GoodputCallback())

    # ── run ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\\n  Training complete in {elapsed/60:.1f} min")

    # Save reward log as CSV for offline analysis
    csv_path = os.path.join(cfg["checkpoint_dir"], "reward_log.csv")
    logger.save_csv(csv_path)

    final_dir = os.path.join(cfg["checkpoint_dir"], "final")
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    shutil.copy2(config_path, os.path.join(final_dir, "training_config.yaml"))
    shutil.copy2(csv_path, os.path.join(final_dir, "reward_log.csv"))
    maybe_push_to_hub(final_dir, cfg, config_path, seed)

    wandb.finish()

    return logger.step_log"""

new_end_train = """    trainer.add_callback(RewardLogCallback(step_tracker, reward_fn))
    trainer.add_callback(GoodputCallback())

    # ── run ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    logger.info(f"\\nTraining complete in {elapsed/60:.1f} min")

    # Save reward log as CSV for offline analysis
    csv_path = os.path.join(cfg["checkpoint_dir"], "reward_log.csv")
    step_tracker.save_csv(csv_path)

    final_dir = os.path.join(cfg["checkpoint_dir"], "final")
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    shutil.copy2(config_path, os.path.join(final_dir, "training_config.yaml"))
    shutil.copy2(csv_path, os.path.join(final_dir, "reward_log.csv"))
    maybe_push_to_hub(final_dir, cfg, config_path, seed)

    wandb.finish()

    return step_tracker.step_log"""

if old_end_train not in code:
    print("WARNING: end_train not found!")
code = code.replace(old_end_train, new_end_train)

# 14. Main total_steps
code = code.replace("""        cfg.env.total_steps = max(1, int(total_samples / effective_batch_size))
        print(f"[P1 Ablation] Matching token budget: {args.total_token_budget} tokens -> total_steps={cfg.env.total_steps}")""", """        cfg.env.total_steps = max(1, int(total_samples / effective_batch_size))
        logger.info(f"[P1 Ablation] Matching token budget: {args.total_token_budget} tokens -> total_steps={cfg.env.total_steps}")""")


with open("platform_tinker/atropos/train_grpo_unsloth.py", "w") as f:
    f.write(code)

print("Done replacing.")

