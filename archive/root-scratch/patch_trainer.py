import sys

file_path = "/Users/arvind/.gemini/antigravity-cli/brain/d11d0942-494f-4f18-92dc-091dc26c5df9/.system_generated/worktrees/subagent-ERF-Logger-self-a4a31886/atropos/tinker_atropos/trainer.py"
with open(file_path, "r") as f:
    content = f.read()

target1 = """        datums = []
        group_mean_rewards = []
        all_reference_logprobs = []
        all_advantages = []
        has_distil_data = False
        skipped_count = 0
        # Distil-specific tracking
        all_teacher_logprobs = []
        all_student_logprobs_for_distil = []
        all_per_token_advantages = []

        for item in batch:
            # Calculate advantages"""

replacement1 = """        datums = []
        group_mean_rewards = []
        all_reference_logprobs = []
        all_advantages = []
        has_distil_data = False
        skipped_count = 0
        total_trajectories = 0
        # Distil-specific tracking
        all_teacher_logprobs = []
        all_student_logprobs_for_distil = []
        all_per_token_advantages = []

        for item in batch:
            total_trajectories += len(item["tokens"])
            # Calculate advantages"""

content = content.replace(target1, replacement1)


target2 = """        else:
            self.distil_stats = {}

        if skipped_count > 0:
            print(f"Skipped {skipped_count} groups with zero advantages")

        return datums, group_mean_rewards, has_distil_data"""

replacement2 = """        else:
            self.distil_stats = {}

        if total_trajectories > 0:
            self.erf_stats = {
                "train/erf": float(len(datums)) / total_trajectories
            }
        else:
            self.erf_stats = {}

        if skipped_count > 0:
            print(f"Skipped {skipped_count} groups with zero advantages")

        return datums, group_mean_rewards, has_distil_data"""

content = content.replace(target2, replacement2)

target3 = """            if hasattr(self, "training_logprob_stats"):
                wandb_metrics.update(self.training_logprob_stats)
            if hasattr(self, "advantage_stats"):
                wandb_metrics.update(self.advantage_stats)"""

replacement3 = """            if hasattr(self, "training_logprob_stats"):
                wandb_metrics.update(self.training_logprob_stats)
            if hasattr(self, "advantage_stats"):
                wandb_metrics.update(self.advantage_stats)
            if hasattr(self, "erf_stats"):
                wandb_metrics.update(self.erf_stats)"""

content = content.replace(target3, replacement3)

with open(file_path, "w") as f:
    f.write(content)
print("Patch applied successfully.")
