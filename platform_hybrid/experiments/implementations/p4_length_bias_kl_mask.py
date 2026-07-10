import numpy as np

def run_p4_kl_surprise_mask():
    """
    P4: length-bias / KL-surprise mask experiment.
    White-box Colab implementation.
    """
    print("Initializing P4 Length-bias / KL-surprise mask experiment...")
    print("Applying KL-surprise masks to reward trajectories...")
    # Mocking
    lengths = np.random.randint(50, 200, size=10)
    kl_surprises = np.random.uniform(0.1, 1.5, size=10)
    for l, kl in zip(lengths, kl_surprises):
        mask_threshold = 0.5
        applied = "Yes" if kl > mask_threshold else "No"
        print(f"Traj length: {l}, KL: {kl:.2f} -> Mask applied: {applied}")

if __name__ == "__main__":
    run_p4_kl_surprise_mask()
