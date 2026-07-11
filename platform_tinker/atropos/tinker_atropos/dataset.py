import time
import json
import numpy as np
import requests
import torch
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Dict, Any, List, Tuple

import tinker

class DatasetPreprocessor:
    """Handles dataset fetching, padding, and distillation processing."""
    
    def __init__(self, atropos_api_url: str):
        self.atropos_api_url = atropos_api_url
        self.logprob_stats = {}
        self.advantage_stats = {}
        self.distil_stats = {}
        self.erf_stats = {}
        self.zvf = 0.0
        self.group_mean_rewards = []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
    def get_batch(self):
        """Fetch a batch of rollouts from Atropos API with retry logic."""
        data = requests.get(f"{self.atropos_api_url}/batch", timeout=10).json()
        return data

    @staticmethod
    def _validate_distil_field(field_data, field_name: str, seq_len: int):
        """Validate distillation fields."""
        if field_data is None:
            return None

        arr = np.array(field_data)

        if arr.ndim == 1:
            raise ValueError(
                f"Distillation field '{field_name}' has shape {arr.shape} (1D). "
                f"Expected 2D shape [seq_len, K]. The distillation API always "
                f"uses [seq_len, K] format — pass K=1 for per-token distillation."
            )
        if arr.ndim != 2:
            raise ValueError(
                f"Distillation field '{field_name}' has unexpected ndim={arr.ndim}, "
                f"shape={arr.shape}. Expected 2D shape [seq_len, K]."
            )
        if arr.shape[0] != seq_len:
            raise ValueError(
                f"Distillation field '{field_name}' has {arr.shape[0]} positions "
                f"but expected {seq_len} (seq_len)."
            )
        if arr.shape[1] != 1:
            raise ValueError(
                f"Distillation field '{field_name}' has K={arr.shape[1]} (shape {arr.shape}). "
                f"Tinker only supports K=1 — its forward_backward_custom only provides "
                f"per-token logprobs, not full vocab distribution. "
                f"Use torchtitan for top-K distillation (K>1)."
            )

        return arr.squeeze(axis=1)

    def pad_data_to_good_offset(
        self, data: Dict[str, Any]
    ) -> tuple[List[tinker.Datum], List[float], bool, float]:
        batch = data["batch"]

        datums = []
        group_mean_rewards = []
        all_reference_logprobs = []
        all_advantages = []
        has_distil_data = False
        skipped_count = 0
        total_trajectories = 0
        
        all_teacher_logprobs = []
        all_student_logprobs_for_distil = []
        all_per_token_advantages = []

        for item in batch:
            total_trajectories += len(item["tokens"])
            scores = np.array(item["scores"])
            original_mean = np.mean(scores)
            advantages = scores - original_mean

            group_mean_rewards.append(original_mean)

            if len(scores) > 1 and np.all(advantages == 0.0):
                skipped_count += 1
                continue

            if item.get("overrides") is not None:
                for i in range(len(item["overrides"])):
                    if item["overrides"][i].get("set_advantage_to_zero", False):
                        advantages[i] = 0.0

            item_has_distil = (
                item.get("distill_token_ids") is not None
                and item.get("distill_logprobs") is not None
            )
            if item_has_distil:
                has_distil_data = True

            for i in range(len(item["tokens"])):
                tokens = item["tokens"][i]
                trajectory_logprobs = item["inference_logprobs"][i]
                advantage = advantages[i]

                all_advantages.append(advantage)

                input_tokens = tokens[:-1]
                target_tokens = tokens[1:]

                all_logprobs = trajectory_logprobs[1:]
                all_advantages_padded = [0.0 if lp == 1.0 else advantage for lp in all_logprobs]
                all_reference_logprobs.extend(all_logprobs)

                seq_len = len(target_tokens)

                assert (
                    len(input_tokens) == seq_len == len(all_logprobs) == len(all_advantages_padded)
                ), f"Length mismatch: input={len(input_tokens)}, target={seq_len}, logprobs={len(all_logprobs)}, advantages={len(all_advantages_padded)}"

                if item_has_distil:
                    raw_distil_ids = item["distill_token_ids"][i]
                    raw_distil_lps = item["distill_logprobs"][i]

                    raw_distil_ids = (
                        raw_distil_ids[1:] if len(raw_distil_ids) > seq_len else raw_distil_ids
                    )
                    raw_distil_lps = (
                        raw_distil_lps[1:] if len(raw_distil_lps) > seq_len else raw_distil_lps
                    )

                    self._validate_distil_field(raw_distil_ids, "distil_token_ids", seq_len)
                    distil_lps = self._validate_distil_field(
                        raw_distil_lps, "distil_logprobs", seq_len
                    )

                    all_advantages_padded = [
                        0.0 if lp == 1.0 else float(t_lp - lp)
                        for lp, t_lp in zip(all_logprobs, distil_lps)
                    ]

                    for lp, t_lp, adv in zip(all_logprobs, distil_lps, all_advantages_padded):
                        if lp != 1.0:
                            all_teacher_logprobs.append(float(t_lp))
                            all_student_logprobs_for_distil.append(float(lp))
                            all_per_token_advantages.append(adv)

                datum = tinker.Datum(
                    model_input=tinker.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs={
                        "target_tokens": tinker.TensorData.from_torch(
                            torch.tensor(target_tokens, dtype=torch.int64)
                        ),
                        "logprobs": tinker.TensorData.from_torch(
                            torch.tensor(all_logprobs, dtype=torch.float32)
                        ),
                        "advantages": tinker.TensorData.from_torch(
                            torch.tensor(all_advantages_padded, dtype=torch.float32)
                        ),
                    },
                )
                datums.append(datum)

        if all_reference_logprobs:
            logprob_array = np.array(all_reference_logprobs)
            logprob_array_actual = logprob_array[(logprob_array != 0.0) & (logprob_array != 1.0)]
            if len(logprob_array_actual) > 0:
                self.logprob_stats = {
                    "logprobs/mean": float(np.mean(logprob_array_actual)),
                    "logprobs/std": float(np.std(logprob_array_actual)),
                    "logprobs/min": float(np.min(logprob_array_actual)),
                    "logprobs/p50": float(np.percentile(logprob_array_actual, 50)),
                }
            else:
                self.logprob_stats = {}
        else:
            self.logprob_stats = {}

        if all_advantages:
            advantages_array = np.array(all_advantages)
            if np.std(advantages_array) > 1e-6:
                self.advantage_stats = {
                    "advantages/mean": float(np.mean(advantages_array)),
                    "advantages/std": float(np.std(advantages_array)),
                    "advantages/sum": float(np.sum(advantages_array)),
                }
            else:
                self.advantage_stats = {}
        else:
            self.advantage_stats = {}

        if all_teacher_logprobs:
            teacher_arr = np.array(all_teacher_logprobs)
            student_arr = np.array(all_student_logprobs_for_distil)
            adv_arr = np.array(all_per_token_advantages)
            self.distil_stats = {
                "distil/teacher_logp_mean": float(np.mean(teacher_arr)),
                "distil/teacher_logp_std": float(np.std(teacher_arr)),
                "distil/teacher_logp_min": float(np.min(teacher_arr)),
                "distil/student_logp_mean": float(np.mean(student_arr)),
                "distil/student_logp_std": float(np.std(student_arr)),
                "distil/advantage_mean": float(np.mean(adv_arr)),
                "distil/advantage_std": float(np.std(adv_arr)),
                "distil/advantage_abs_mean": float(np.mean(np.abs(adv_arr))),
                "distil/kl_approx": float(np.mean(student_arr - teacher_arr)),
                "distil/num_tokens": len(all_teacher_logprobs),
            }
        else:
            self.distil_stats = {}

        if total_trajectories > 0:
            self.erf_stats = {
                "train/erf": float(len(datums)) / total_trajectories
            }
        else:
            self.erf_stats = {}

        if skipped_count > 0:
            print(f"Skipped {skipped_count} groups with zero advantages")

        zvf = skipped_count / max(1, len(batch))
        return datums, group_mean_rewards, has_distil_data, zvf

    def get_data(self) -> tuple[List[tinker.Datum], bool]:
        while True:
            data = self.get_batch()

            if data.get("batch") is not None:
                with open("temp.json", "w", encoding="utf-8") as f:
                    json.dump(data, f)

                datums, group_mean_rewards, has_distil, zvf = self.pad_data_to_good_offset(data)
                self.group_mean_rewards = group_mean_rewards
                self.zvf = zvf
                return datums, has_distil
            else:
                time.sleep(1)
