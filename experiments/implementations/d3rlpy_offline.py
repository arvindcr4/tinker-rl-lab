"""
d3rlpy Offline RL Implementation
=================================
Port of Tinker distillation concepts to offline RL with d3rlpy.

d3rlpy specializes in offline RL algorithms:
- CQL (Conservative Q-Learning)
- IQL (Implicit Q-Learning)
- TD3+BC, AWAC, etc.

Use case: Train from pre-collected trajectories without online interaction.
"""

import numpy as np
import d3rlpy
from d3rlpy.dataset import MDPDataset


def create_arithmetic_dataset(
    num_episodes: int = 10000,
    max_num: int = 99,
    expert_ratio: float = 0.7,
) -> MDPDataset:
    """
    Create offline dataset for arithmetic task.

    Args:
        num_episodes: Number of episodes
        max_num: Maximum number in addition
        expert_ratio: Fraction of correct (expert) trajectories
    """
    observations = []
    actions = []
    rewards = []
    terminals = []

    for _ in range(num_episodes):
        # Generate problem
        num1 = np.random.randint(1, max_num + 1)
        num2 = np.random.randint(1, max_num + 1)
        correct_answer = num1 + num2

        obs = np.array([num1 / max_num, num2 / max_num], dtype=np.float32)

        # Expert vs random action
        if np.random.random() < expert_ratio:
            action = correct_answer  # Expert action
            reward = 1.0
        else:
            # Random incorrect action
            action = np.random.randint(0, max_num * 2 + 1)
            reward = 1.0 if action == correct_answer else 0.0

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        terminals.append(True)  # Single-step episodes

    return MDPDataset(
        observations=np.array(observations),
        actions=np.array(actions).reshape(-1, 1),
        rewards=np.array(rewards),
        terminals=np.array(terminals),
    )


def train_cql(dataset: MDPDataset):
    """
    Train with Conservative Q-Learning (CQL).

    CQL adds a conservative penalty to prevent overestimation
    of Q-values for out-of-distribution actions.
    """
    print("Training CQL...")

    cql = d3rlpy.algos.DiscreteCQLConfig(
        learning_rate=3e-4,
        batch_size=256,
        gamma=0.99,
        # CQL-specific
        alpha=4.0,  # Conservative penalty weight
    ).create(device="cuda:0" if d3rlpy.torch_utility.get_device() else "cpu:0")

    cql.fit(
        dataset,
        n_steps=10000,
        n_steps_per_epoch=1000,
        show_progress=True,
    )

    cql.save_model("cql_arithmetic.d3")
    return cql


def train_iql(dataset: MDPDataset):
    """
    Train with Implicit Q-Learning (IQL).

    IQL avoids querying OOD actions by using expectile regression.
    Good for offline RL when you can't interact with environment.
    """
    print("Training IQL...")

    iql = d3rlpy.algos.DiscreteIQLConfig(
        learning_rate=3e-4,
        batch_size=256,
        gamma=0.99,
        # IQL-specific
        expectile=0.7,  # Asymmetric loss parameter
        weight_temp=3.0,  # Temperature for advantage weighting
    ).create(device="cuda:0" if d3rlpy.torch_utility.get_device() else "cpu:0")

    iql.fit(
        dataset,
        n_steps=10000,
        n_steps_per_epoch=1000,
        show_progress=True,
    )

    iql.save_model("iql_arithmetic.d3")
    return iql


def train_bc(dataset: MDPDataset):
    """
    Behavior Cloning baseline.

    Simple imitation learning - matches Tinker's off-policy distillation
    when applied to expert demonstrations.
    """
    print("Training Behavior Cloning...")

    bc = d3rlpy.algos.DiscreteBCConfig(
        learning_rate=1e-4,
        batch_size=256,
    ).create(device="cuda:0" if d3rlpy.torch_utility.get_device() else "cpu:0")

    bc.fit(
        dataset,
        n_steps=5000,
        n_steps_per_epoch=1000,
        show_progress=True,
    )

    bc.save_model("bc_arithmetic.d3")
    return bc


def evaluate_model(model, num_eval: int = 1000, max_num: int = 99):
    """Evaluate trained model on arithmetic task."""
    correct = 0

    for _ in range(num_eval):
        num1 = np.random.randint(1, max_num + 1)
        num2 = np.random.randint(1, max_num + 1)
        expected = num1 + num2

        obs = np.array([[num1 / max_num, num2 / max_num]], dtype=np.float32)
        action = model.predict(obs)[0]

        if action == expected:
            correct += 1

    accuracy = correct / num_eval * 100
    return accuracy


def main():
    print("=" * 60)
    print("d3rlpy Offline RL for Math Task")
    print("=" * 60)

    # Create offline dataset
    print("\nCreating offline dataset...")
    dataset = create_arithmetic_dataset(
        num_episodes=10000,
        expert_ratio=0.7,  # 70% expert demonstrations
    )
    print(f"Dataset size: {len(dataset)} episodes")

    # Train different algorithms
    print("\n--- Training Algorithms ---")

    # 1. Behavior Cloning (baseline, like off-policy distillation)
    bc_model = train_bc(dataset)
    bc_acc = evaluate_model(bc_model)
    print(f"Behavior Cloning accuracy: {bc_acc:.1f}%")

    # 2. CQL (conservative offline RL)
    cql_model = train_cql(dataset)
    cql_acc = evaluate_model(cql_model)
    print(f"CQL accuracy: {cql_acc:.1f}%")

    # 3. IQL (implicit Q-learning)
    iql_model = train_iql(dataset)
    iql_acc = evaluate_model(iql_model)
    print(f"IQL accuracy: {iql_acc:.1f}%")

    print("\n--- Results Summary ---")
    print(f"Behavior Cloning: {bc_acc:.1f}%")
    print(f"CQL:              {cql_acc:.1f}%")
    print(f"IQL:              {iql_acc:.1f}%")
    print("\nExpected: All should approach ~70% (expert ratio in dataset)")


if __name__ == "__main__":
    main()
