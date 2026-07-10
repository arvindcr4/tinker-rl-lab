import numpy as np

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

if __name__ == "__main__":
    controller = ZVF_PIDController(target_zvf=0.4)
    # Simulate a loop
    zvf_mock = [0.8, 0.7, 0.5, 0.3, 0.2]
    print("Simulating live PID loop for ZVF Controller")
    for z in zvf_mock:
        g = controller.step(z)
        print(f"Observed ZVF: {z:.2f} -> Controller outputs Group Size G={g}")
