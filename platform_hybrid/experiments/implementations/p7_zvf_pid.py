import numpy as np
import wandb
import os
import json
from huggingface_hub import HfApi

class ZVF_PIDController:
    def __init__(self, target_zvf=0.5, kp=1.0, ki=0.1, kd=0.05):
        self.target = target_zvf
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.integral = 0.0
        self.prev_error = 0.0
        
    def step(self, current_zvf):
        error = self.target - current_zvf
        self.integral += error
        derivative = error - self.prev_error
        
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        self.prev_error = error
        
        # Output can map to Group Size G
        # Base G is 8, adjustment based on output
        new_g = int(8 + output)
        return max(2, min(16, new_g)) # Clamp between 2 and 16

    def push_to_hub(self, repo_id):
        save_directory = "./tmp_pid_model"
        os.makedirs(save_directory, exist_ok=True)
        config = {
            "target": self.target,
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "integral": self.integral,
            "prev_error": self.prev_error,
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)
            
        api = HfApi()
        api.create_repo(repo_id, exist_ok=True)
        api.upload_folder(
            folder_path=save_directory,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"Pushed model to Hugging Face Hub: {repo_id}")

if __name__ == "__main__":
    wandb.init(project="tinker-rl-lab", name="p7-zvf-pid")
    
    controller = ZVF_PIDController(target_zvf=0.4)
    # Simulate a loop
    zvf_mock = [0.8, 0.7, 0.5, 0.3, 0.2]
    print("Simulating live PID loop for ZVF Controller")
    for z in zvf_mock:
        g = controller.step(z)
        print(f"Observed ZVF: {z:.2f} -> Controller outputs Group Size G={g}")
        wandb.log({"zvf": z, "group_size": g})
        
    # Example checkpoint push
    # controller.push_to_hub("arvindcr4/zvf-pid-controller")
    wandb.finish()
