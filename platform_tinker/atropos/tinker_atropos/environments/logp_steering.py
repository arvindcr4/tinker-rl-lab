"""
Logprob-steering RLHF environment.

Uses WildChat first-turn prompts.  The "teacher" is the *same model* being
trained, but with a system prompt prepended (e.g. "respond in a
Bataille/Deleuze ontology").  Teacher logprobs come from tinker's /logprobs
endpoint — no separate teacher server needed.

The per-token logp_teacher is delivered as distillation arrays so the
trainer applies on-policy distillation (advantage = logp_teacher - logp_student).
"""

import asyncio
import math
import copy
from typing import Dict, List, Optional, Tuple

import aiohttp
from datasets import load_dataset

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item
from tinker_atropos.config import TinkerAtroposConfig

CONFIG_PATH = "configs/logp_steering_qwen3_30b.yaml"

# ── steering prompt ──────────────────────────────────────────────────────
# This gets prepended as a system message when fetching teacher logprobs.
# The student sees the conversation *without* this prompt, so the training
# signal is: "shift your distribution toward responses that would be
# produced under this ontological framing."
STEERING_SYSTEM_PROMPT = "Use many emojis in your response, like at least 10% of the tokens."


class LogpSteeringEnv(BaseEnv):
    """
    RLHF env that steers a model via logprob distillation from itself
    under a different system prompt.

    Flow per item:
      1. Pick a first-turn user message from WildChat.
      2. Generate group_size completions (student, no steering prompt).
      3. For each completion, build the full token sequence *with* the
         steering system prompt and call tinker /logprobs to get teacher
         per-token logprobs.
      4. Pack distill arrays so the trainer applies on-policy distillation
         (per-token advantage = logp_teacher − logp_student).
    """

    name = "logp-steering"

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.dataset = None
        self.iter = 0
        # URL of the tinker-atropos FastAPI server (where /logprobs lives)
        # This is the same server that serves /v1/ and /generate
        self.tinker_api_url = str(server_configs[0].base_url).removesuffix("/v1").rstrip("/")

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        config = (
            TinkerAtroposConfig.from_yaml(CONFIG_PATH) if CONFIG_PATH else TinkerAtroposConfig()
        )

        env_config = BaseEnvConfig(
            tokenizer_name=config.base_model,
            group_size=config.group_size,
            use_wandb=config.use_wandb,
            rollout_server_url=config.atropos_api_url,
            total_steps=config.num_steps,
            batch_size=config.batch_size,
            steps_per_eval=config.steps_per_eval,
            max_token_length=config.max_token_env_length,
            max_num_workers=config.max_num_workers,
            max_batches_offpolicy=config.max_batches_offpolicy,
            wandb_name=f"{config.wandb_run_name}-env",
            ensure_scores_are_not_same=config.ensure_scores_are_not_same,
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            # We handle distillation ourselves via /logprobs
            distillation_enabled=False,
        )
        server_configs = [
            APIServerConfig(
                model_name=config.base_model,
                base_url=config.inference_api_url + "/v1",
                api_key="x",
                server_type="sglang",
                num_requests_for_eval=config.num_requests_for_eval,
            ),
        ]

        return env_config, server_configs

    async def setup(self):
        self.dataset = load_dataset("allenai/WildChat", split="train")
        self.iter = 0
        self._http_session = aiohttp.ClientSession()

        # Pre-tokenize the steering system prompt prefix once.
        # This is the token sequence for: <|im_start|>system\n{STEERING}<|im_end|>\n
        sys_tpl = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": STEERING_SYSTEM_PROMPT}],
            tokenize=True,
            add_generation_prompt=False,
        )
        self._steering_prefix_ids = (
            sys_tpl["input_ids"] if hasattr(sys_tpl, "input_ids") else list(sys_tpl)
        )
        print(f"[SETUP] Steering prefix: {len(self._steering_prefix_ids)} tokens")

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def rollout_and_score_eval(self, question, answer):
        pass

    async def evaluate(self, *args, **kwargs):
        pass

    # ── helpers ───────────────────────────────────────────────────────

    def _extract_first_turn(self, item) -> Optional[List[Dict]]:
        """
        Pull out the first user message from a WildChat conversation.
        Returns a list of chat messages up to (and including) the first
        user turn, or None if the conversation is empty/invalid.
        """
        conversation = item.get("conversation", [])
        chat: List[Dict] = []
        for msg in conversation:
            chat.append({"role": msg["role"], "content": msg["content"]})
            if msg["role"] == "user":
                break
        if not chat or chat[-1]["role"] != "user":
            return None
        return chat

    async def _get_teacher_logprobs_from_tinker(self, token_ids: List[int]) -> List[float]:
        """
        Call tinker-atropos /logprobs endpoint to get per-token logprobs
        for a token sequence (with steering prompt baked in).
        """
        payload = {"input_ids": token_ids}
        for attempt in range(3):
            try:
                async with self._http_session.post(
                    f"{self.tinker_api_url}/logprobs",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status in {429, 502, 503, 504}:
                        raise RuntimeError(f"transient teacher status {resp.status}")
                    resp.raise_for_status()
                    data = await resp.json()
                entries = data["logprobs"]
                if len(entries) != len(token_ids) or [e["token_id"] for e in entries] != token_ids:
                    raise ValueError("Teacher token alignment mismatch")
                return {
                    "values": [e["logprob"] for e in entries],
                    "policy_version": data.get("policy_version"),
                }
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError):
                if attempt < 2:
                    await asyncio.sleep(0.25 * 2**attempt)
            except (ValueError, KeyError, TypeError):
                break
        return None  # unavailable is unscored, never probability one

    # ── main trajectory collection ───────────────────────────────────

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, list[Item]]:
        first_turn = self._extract_first_turn(item)
        if first_turn is None:
            return None, []

        # Student chat: no steering prompt
        student_chat = copy.deepcopy(first_turn)

        # Check length before generating
        prompt_tpl = self.tokenizer.apply_chat_template(
            student_chat, tokenize=True, add_generation_prompt=True
        )
        prompt_tokens = prompt_tpl["input_ids"] if hasattr(prompt_tpl, "input_ids") else prompt_tpl
        if len(prompt_tokens) >= self.config.max_token_length - 256:
            return None, []

        # Generate group_size completions via managed server (sglang /generate)
        max_new_tokens = min(
            self.config.max_token_length // 2,
            self.config.max_token_length - len(prompt_tokens) - 16,
        )

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            chat_completions = await managed.chat_completion(
                messages=student_chat,
                n=self.config.group_size,
                max_tokens=max_new_tokens,
                temperature=1.0,
            )
            state = managed.get_state()
            nodes = state["nodes"]

        # Build scored data
        scored_data = ScoredDataGroup()
        scored_data["tokens"] = []
        scored_data["masks"] = []
        scored_data["scores"] = []
        scored_data["inference_logprobs"] = []
        scored_data["distill_token_ids"] = []
        scored_data["distill_logprobs"] = []

        # For each completion, get teacher logprobs
        teacher_tasks = []

        for i, choice in enumerate(chat_completions.choices):
            student_tokens = nodes[i].tokens
            student_masks = nodes[i].masked_tokens
            student_logprobs = nodes[i].logprobs

            # Build teacher token sequence by prepending the steering system
            # prompt tokens to the actual model-generated tokens.
            # This avoids detokenize→retokenize round-trip loss.
            teacher_token_ids = self._steering_prefix_ids + student_tokens

            teacher_tasks.append(self._get_teacher_logprobs_from_tinker(teacher_token_ids))

            scored_data["tokens"].append(student_tokens)
            scored_data["masks"].append(student_masks)
            scored_data["inference_logprobs"].append(student_logprobs)

        # Gather all teacher logprob calls concurrently
        teacher_results = await asyncio.gather(*teacher_tasks)

        versions = {result["policy_version"] for result in teacher_results if result is not None}
        if (
            None in versions
            or len(versions) != 1
            or any(result is None for result in teacher_results)
        ):
            return None, []
        scored_data["teacher_policy_version"] = next(iter(versions))
        teacher_results = [result["values"] for result in teacher_results]

        # Fail the group closed if any generated token lacks an aligned teacher value.
        for i, values in enumerate(teacher_results):
            tokens = scored_data["tokens"][i]
            if values is None or len(values) != len(self._steering_prefix_ids) + len(tokens):
                return None, []
            aligned = values[-len(tokens) :]
            behavior = scored_data["inference_logprobs"][i]
            if len(behavior) != len(tokens):
                return None, []
            for lp, teacher_lp in zip(behavior, aligned):
                if lp != 1.0 and (
                    teacher_lp is None or not math.isfinite(teacher_lp) or teacher_lp > 0
                ):
                    return None, []
            # Unavailable prompt positions remain null and are masked by the preprocessor.
            teacher_results[i] = aligned

        all_length_penalty = True
        for i, choice in enumerate(chat_completions.choices):
            teacher_lps_full = teacher_results[i]
            student_tokens = scored_data["tokens"][i]

            # The teacher token sequence is longer (has system prompt tokens).
            # Align: take the LAST len(student_tokens) positions from the
            # teacher logprobs so the completion portion lines up.
            teacher_lps_aligned = teacher_lps_full

            # distill format: [seq_len, K] where K=1 for tinker
            scored_data["distill_token_ids"].append([[tid] for tid in student_tokens])
            scored_data["distill_logprobs"].append([[lp] for lp in teacher_lps_aligned])

            # Score: mean per-token (teacher - student) logprob diff on the
            # generated (non-prompt) portion. Penalize length truncation.
            if choice.finish_reason == "length":
                scored_data["scores"].append(-1.0)
            else:
                all_length_penalty = False
                student_lps = scored_data["inference_logprobs"][i]
                n = min(len(teacher_lps_aligned), len(student_lps))
                if n > 0:
                    diffs = []
                    for j in range(n):
                        # prompt tokens have logprob=1.0 as sentinel
                        if student_lps[j] != 1.0:
                            diffs.append(teacher_lps_aligned[j] - student_lps[j])
                    score = sum(diffs) / max(len(diffs), 1)
                else:
                    score = 0.0
                scored_data["scores"].append(score)

        if all_length_penalty:
            return None, []

        return scored_data, []

    async def get_next_item(self):
        next_item = self.dataset[self.iter % len(self.dataset)]
        self.iter += 1
        return next_item


if __name__ == "__main__":
    LogpSteeringEnv.cli()
