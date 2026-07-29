"""
Fail-Closed Verification & Diagnostic Tooling Suite for Category 10 (Ideas 10.1 - 10.5)
Verification script to test static analysis DAGs, dynamic invariants, differential fuzzing, cryptographic audit ledgers, and tensor memory bounds.
"""

import math
import hashlib
import random
import time
from typing import Dict, List, Tuple, Any, Optional

# ==========================================
# Idea 10.1: Static Analysis Abstract Interpretation & Fail-Closed DAG
# ==========================================

class AbstractDomain:
    """Abstract interval domain [low, high] for state variable bounds."""
    def __init__(self, low: float, high: float, is_bot: bool = False):
        self.low = low
        self.high = high
        self.is_bot = is_bot

    def join(self, other: 'AbstractDomain') -> 'AbstractDomain':
        if self.is_bot: return other
        if other.is_bot: return self
        return AbstractDomain(min(self.low, other.low), max(self.high, other.high))

    def meets_invariant(self, safe_min: float, safe_max: float) -> bool:
        if self.is_bot: return False
        return self.low >= safe_min and self.high <= safe_max

    def __repr__(self):
        return "⊥" if self.is_bot else f"[{self.low:.2f}, {self.high:.2f}]"

class DAGNode:
    def __init__(self, name: str, transfer_fn):
        self.name = name
        self.transfer_fn = transfer_fn
        self.parents: List['DAGNode'] = []
        self.children: List['DAGNode'] = []

    def add_child(self, child: 'DAGNode'):
        self.children.append(child)
        child.parents.append(self)

class StaticFailClosedAnalyzer:
    """Static analysis tool using abstract interpretation on execution DAGs."""
    def __init__(self, nodes: List[DAGNode], safe_bounds: Dict[str, Tuple[float, float]]):
        self.nodes = nodes
        self.safe_bounds = safe_bounds

    def analyze(self, initial_domain: AbstractDomain) -> Tuple[bool, Dict[str, AbstractDomain], List[str]]:
        state_map: Dict[str, AbstractDomain] = {}
        halt_logs: List[str] = []
        fail_closed_triggered = False

        # Topological propagation
        for node in self.nodes:
            if not node.parents:
                input_domain = initial_domain
            else:
                input_domain = AbstractDomain(0, 0, is_bot=True)
                for p in node.parents:
                    input_domain = input_domain.join(state_map[p.name])

            if fail_closed_triggered:
                state_map[node.name] = AbstractDomain(0, 0, is_bot=True)
                continue

            out_domain = node.transfer_fn(input_domain)
            safe_min, safe_max = self.safe_bounds.get(node.name, (-math.inf, math.inf))
            
            if not out_domain.meets_invariant(safe_min, safe_max):
                fail_closed_triggered = True
                halt_logs.append(f"[FAIL-CLOSED TRIGGERED] Node '{node.name}' domain {out_domain} violated safety bounds [{safe_min}, {safe_max}]. Immediate Pipeline Halt.")
                state_map[node.name] = AbstractDomain(0, 0, is_bot=True)
            else:
                state_map[node.name] = out_domain

        return not fail_closed_triggered, state_map, halt_logs


# ==========================================
# Idea 10.2: Dynamic Policy Invariant Runtime Verification
# ==========================================

class DynamicPolicyMonitor:
    """Runtime invariant assertion monitor enforcing policy update bounds."""
    def __init__(self, grad_norm_max: float = 10.0, adv_abs_max: float = 5.0, min_entropy: float = 0.1):
        self.grad_norm_max = grad_norm_max
        self.adv_abs_max = adv_abs_max
        self.min_entropy = min_entropy
        self.history: List[Dict[str, float]] = []

    def verify_step(self, step: int, grad_norm: float, advantages: List[float], logits: List[float]) -> Tuple[bool, List[str]]:
        violations = []
        
        # 1. Gradient Norm Contract
        if math.isnan(grad_norm) or math.isinf(grad_norm) or grad_norm > self.grad_norm_max:
            violations.append(f"Step {step}: Gradient norm breach ({grad_norm} > {self.grad_norm_max})")

        # 2. Advantage Bounds Contract
        max_adv = max(abs(a) for a in advantages) if advantages else 0.0
        if max_adv > self.adv_abs_max:
            violations.append(f"Step {step}: Advantage bound breach (max |A| = {max_adv:.2f} > {self.adv_abs_max})")

        # 3. Policy Entropy Floor Contract
        # Softmax & Entropy calculation
        exp_l = [math.exp(l) for l in logits]
        sum_l = sum(exp_l)
        probs = [p / sum_l for p in exp_l]
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)

        if entropy < self.min_entropy:
            violations.append(f"Step {step}: Entropy collapse breach (Entropy = {entropy:.4f} < {self.min_entropy})")

        is_safe = len(violations) == 0
        self.history.append({"step": step, "grad_norm": grad_norm, "max_adv": max_adv, "entropy": entropy, "safe": is_safe})
        return is_safe, violations


# ==========================================
# Idea 10.3: Differential Fuzzing Engine for Gym Environments
# ==========================================

class MockGymEnv:
    """Environment implementation with optional floating-point or state bugs."""
    def __init__(self, bug_type: Optional[str] = None):
        self.bug_type = bug_type
        self.state = [0.0, 0.0]

    def reset(self, seed: int):
        random.seed(seed)
        self.state = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
        return list(self.state)

    def step(self, action: float) -> Tuple[List[float], float, bool]:
        x, v = self.state
        v_next = v + 0.1 * action - 0.01 * x
        x_next = x + v_next

        if self.bug_type == "state_corruption" and random.random() < 0.1:
            x_next += 0.5  # Silent bug injection!

        self.state = [x_next, v_next]
        reward = - (x_next**2 + 0.1 * action**2)
        done = abs(x_next) > 5.0
        return list(self.state), reward, done

class DifferentialFuzzer:
    """Differential fuzzing engine comparing env rollouts with tolerance bounds."""
    def __init__(self, env1: MockGymEnv, env2: MockGymEnv, tol: float = 1e-4):
        self.env1 = env1
        self.env2 = env2
        self.tol = tol

    def fuzz_rollout(self, seed: int, num_steps: int) -> Tuple[bool, int, str]:
        s1 = self.env1.reset(seed)
        s2 = self.env2.reset(seed)

        random.seed(seed + 999)
        for t in range(num_steps):
            action = random.uniform(-1.0, 1.0)
            next_s1, r1, d1 = self.env1.step(action)
            next_s2, r2, d2 = self.env2.step(action)

            # Check state discrepancy with tolerance
            max_diff = max(abs(a - b) for a, b in zip(next_s1, next_s2))
            if max_diff > self.tol or abs(r1 - r2) > self.tol or d1 != d2:
                msg = f"Discrepancy detected at step {t}: state_diff={max_diff:.5f}, reward_diff={abs(r1-r2):.5f}, done_match={d1==d2}"
                return False, t, msg

        return True, num_steps, "Rollouts identical within tolerance"


# ==========================================
# Idea 10.4: Cryptographically Signed Audit Ledger
# ==========================================

class MerkleAuditLedger:
    """Append-only cryptographic audit ledger for experiment reproducibility."""
    def __init__(self):
        self.ledger: List[Dict[str, Any]] = []
        self.current_root: str = "0" * 64

    def append_record(self, code_commit: str, env_hash: str, params: Dict[str, Any], seed: int) -> str:
        record_str = f"{code_commit}:{env_hash}:{sorted(params.items())}:{seed}"
        rec_hash = hashlib.sha256(record_str.encode('utf-8')).hexdigest()
        
        # Link in MMR hash chain
        new_root = hashlib.sha256(f"{self.current_root}:{rec_hash}".encode('utf-8')).hexdigest()
        signature = f"SIG_ED25519_{new_root[:16]}" # Mocked cryptographic signature
        
        entry = {
            "index": len(self.ledger),
            "timestamp": time.time(),
            "prev_root": self.current_root,
            "record_hash": rec_hash,
            "root": new_root,
            "signature": signature
        }
        self.ledger.append(entry)
        self.current_root = new_root
        return new_root

    def verify_ledger(self) -> bool:
        prev = "0" * 64
        for entry in self.ledger:
            if entry["prev_root"] != prev:
                return False
            expected_root = hashlib.sha256(f"{prev}:{entry['record_hash']}".encode('utf-8')).hexdigest()
            if entry["root"] != expected_root:
                return False
            prev = entry["root"]
        return True


# ==========================================
# Idea 10.5: CUDA / Dynamic Tensor Memory Sanitizer
# ==========================================

class TensorMemorySanitizer:
    """Simulated CUDA memory sanitizer with red-zone canary checking."""
    CANARY = 0xDEADBEEF

    def __init__(self, allocation_size: int):
        self.allocation_size = allocation_size
        # Allocate buffer with front & rear red-zones
        self.buffer = [self.CANARY] * 4 + [0.0] * allocation_size + [self.CANARY] * 4
        self.data_offset = 4

    def write(self, index: int, value: float):
        target_idx = self.data_offset + index
        if target_idx < 4 or target_idx >= 4 + self.allocation_size:
            # Red-zone corruption write!
            self.buffer[target_idx] = value
        else:
            self.buffer[target_idx] = value

    def check_red_zones(self) -> Tuple[bool, List[str]]:
        corruptions = []
        for i in range(4):
            if self.buffer[i] != self.CANARY:
                corruptions.append(f"Front red-zone canary corrupted at index {i}: value={self.buffer[i]}")
        for i in range(4 + self.allocation_size, 4 + self.allocation_size + 4):
            if self.buffer[i] != self.CANARY:
                corruptions.append(f"Rear red-zone canary corrupted at index {i}: value={self.buffer[i]}")

        return len(corruptions) == 0, corruptions


# ==========================================
# Comprehensive Test Runner
# ==========================================

def run_all_verifications():
    print("=== Category 10 Verification Suite Execution ===")
    
    # 1. Verify Idea 10.1: Static Abstract Interpretation DAG
    print("\n--- Testing 10.1: Static Analysis Fail-Closed DAG ---")
    node_a = DAGNode("InputIngest", lambda dom: AbstractDomain(dom.low * 1.0, dom.high * 1.0))
    node_b = DAGNode("FeatureNorm", lambda dom: AbstractDomain(dom.low / 2.0, dom.high / 2.0))
    node_c = DAGNode("PolicyInference", lambda dom: AbstractDomain(dom.low * 5.0, dom.high * 10.0)) # Explodes bound!
    
    node_a.add_child(node_b)
    node_b.add_child(node_c)

    safe_bounds = {
        "InputIngest": (-10.0, 10.0),
        "FeatureNorm": (-5.0, 5.0),
        "PolicyInference": (-15.0, 15.0) # Maximum allowable inference bound
    }
    
    analyzer = StaticFailClosedAnalyzer([node_a, node_b, node_c], safe_bounds)
    passed, state_map, logs = analyzer.analyze(AbstractDomain(-4.0, 4.0))
    print(f"Analysis Pass: {passed}")
    for log in logs:
        print(f"  {log}")
    assert not passed, "Static analyzer should have triggered fail-closed on PolicyInference bound explosion!"
    print("  [SUCCESS] 10.1 Static analysis successfully halts invalid execution DAG.")

    # 2. Verify Idea 10.2: Dynamic Policy Invariants Monitor
    print("\n--- Testing 10.2: Dynamic Policy Invariants Monitor ---")
    monitor = DynamicPolicyMonitor(grad_norm_max=5.0, adv_abs_max=3.0, min_entropy=0.2)
    
    # Normal step
    safe1, v1 = monitor.verify_step(step=1, grad_norm=1.2, advantages=[0.5, -0.2, 1.1], logits=[1.0, 2.0, 0.5])
    assert safe1, f"Step 1 should be safe: {v1}"
    
    # Abnormal step (gradient explosion & entropy collapse)
    safe2, v2 = monitor.verify_step(step=2, grad_norm=12.5, advantages=[0.5, -0.2], logits=[20.0, 0.001, 0.001])
    assert not safe2, "Step 2 should trigger invariant failure"
    print(f"  Captured Violations at Step 2: {v2}")
    print("  [SUCCESS] 10.2 Dynamic monitor accurately traps numerical instability contracts.")

    # 3. Verify Idea 10.3: Differential Fuzzing Engine
    print("\n--- Testing 10.3: Differential Fuzzing Engine ---")
    clean_env1 = MockGymEnv(bug_type=None)
    clean_env2 = MockGymEnv(bug_type=None)
    buggy_env2 = MockGymEnv(bug_type="state_corruption")

    fuzzer_clean = DifferentialFuzzer(clean_env1, clean_env2)
    pass_clean, steps_clean, msg_clean = fuzzer_clean.fuzz_rollout(seed=42, num_steps=50)
    assert pass_clean, "Identical clean envs should pass fuzzing"
    print(f"  Clean Rollout: {msg_clean}")

    fuzzer_buggy = DifferentialFuzzer(clean_env1, buggy_env2)
    pass_buggy, steps_buggy, msg_buggy = fuzzer_buggy.fuzz_rollout(seed=42, num_steps=50)
    assert not pass_buggy, "Buggy env should be caught by fuzzer"
    print(f"  Buggy Rollout Trapped at step {steps_buggy}: {msg_buggy}")
    print("  [SUCCESS] 10.3 Differential fuzzer isolates state transition discrepancy.")

    # 4. Verify Idea 10.4: Cryptographically Signed Audit Ledger
    print("\n--- Testing 10.4: Cryptographically Signed Audit Ledger ---")
    ledger = MerkleAuditLedger()
    root1 = ledger.append_record("commit_a1b2c3", "sha256_env_v1", {"lr": 1e-4, "batch_size": 64}, seed=42)
    root2 = ledger.append_record("commit_a1b2c3", "sha256_env_v1", {"lr": 1e-4, "batch_size": 64}, seed=43)
    
    assert ledger.verify_ledger(), "Ledger should verify integrity"
    print(f"  Ledger Hash Chain Verified. Final Root: {root2[:16]}...")
    
    # Tampering test
    ledger.ledger[0]["record_hash"] = "0" * 64
    assert not ledger.verify_ledger(), "Tampered ledger must fail verification"
    print("  [SUCCESS] 10.4 Cryptographic ledger correctly guarantees immutable audit trail.")

    # 5. Verify Idea 10.5: CUDA / Dynamic Tensor Memory Sanitizer
    print("\n--- Testing 10.5: CUDA Memory Sanitizer & Red-Zone Bounds ---")
    sanitizer = TensorMemorySanitizer(allocation_size=100)
    
    # In-bounds write
    sanitizer.write(50, 3.14159)
    safe_mem1, corrupt1 = sanitizer.check_red_zones()
    assert safe_mem1, f"In-bounds write failed: {corrupt1}"

    # Out-of-bounds write (red-zone overflow)
    sanitizer.write(102, 999.99) # Out of bounds!
    safe_mem2, corrupt2 = sanitizer.check_red_zones()
    assert not safe_mem2, "Out-of-bounds write must be detected by canary red-zones"
    print(f"  Captured Memory Defect: {corrupt2}")
    print("  [SUCCESS] 10.5 Memory sanitizer successfully traps dynamic buffer overflow.")

    print("\n========================================================")
    print(" ALL CATEGORY 10 VERIFICATION TESTS PASSED SUCCESSFULLY! ")
    print("========================================================")

if __name__ == "__main__":
    run_all_verifications()
