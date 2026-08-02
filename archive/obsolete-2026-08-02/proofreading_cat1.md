# ZAI Proofreading Report: Category 1 (Zero-Variance Starvation & GRPO Policy Optimization)

> **Document ID**: `ZAI-PROOFREADING-CAT1-2026`  
> **Target Ideas**: Ideas 1.1 to 1.5  
> **Source Catalog**: `50_research_ideas_catalog.md`  
> **Status**: Verified & Refined (Fail-Closed Provenance)  

---

## Executive Summary

Category 1 addresses **Zero-Variance Starvation (ZVF)** in Group-Relative Policy Optimization (GRPO). ZVF occurs when sampled rollouts within a prompt group $G$ produce identical reward outputs (e.g., all 0s or all 1s on unit tests or math verifiers), causing the empirical reward variance $\sigma_G^2$ to collapse to zero ($\sigma_G^2 \to 0$). In standard GRPO, this leads to an advantage of $0/\epsilon = 0$, freezing policy updates precisely on hard edge cases or trivial prompts.

This proofreading report rigorously audits Ideas 1.1 through 1.5, identifies structural and mathematical flaws in the original drafts (including LaTeX escape corruptions and advantage zero-numerator paradoxes), presents rigorous mathematical derivations for each core mechanism, and records the corrections applied to the master catalog.

---

## Detailed Proofreading Notes & Corrections

### Idea 1.1: Adaptive Group-Relative Advantage Normalization (AGAN-GRPO)

#### 1. Identified Issues & Flaws in Draft
- **LaTeX Encoding Corruption**: The original string contained `\sigma_G^2 \to 0` escaped as `\sigma_G^2 \to 0` (corrupted into tab character), `\tilde{\sigma}_G^2` as `ilde{\sigma}_G^2`, and `\mathbb{D}_{\text{KL}}` as `\mathbb{D}_{ext{KL}}`.
- **Mathematical Zero-Numerator Flaw**: The original draft proposed $\tilde{\sigma}_G^2 = \sigma_G^2 + \epsilon_t \cdot \exp(-\mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}))$. However, when all rewards in group $G$ are identical ($r_i = \mu_G$), the numerator $r_i - \mu_G$ is identically zero. Simply adjusting the denominator variance $\tilde{\sigma}_G$ still leaves $A_i = 0 / \tilde{\sigma}_G = 0$, failing to eliminate Zero-Variance Starvation!

#### 2. Rigorous Reformulation & Mathematical Solution
To resolve the zero-numerator paradox, AGAN-GRPO uses a **dual-scale advantage estimator**. When intra-group variance drops below a threshold $\delta$, the baseline falls back to a running cross-prompt batch baseline $(\mu_B, \sigma_B^2)$:

$$A_i = \frac{r_i - \mu_G}{\sqrt{\sigma_G^2 + \epsilon}} + \mathbb{I}(\sigma_G^2 < \delta) \cdot \frac{\mu_G - \mu_B}{\sqrt{\sigma_B^2 + \epsilon_t \cdot \exp(-\mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}))}}$$

- **Case 1: Intra-Group Variance ($\sigma_G^2 \ge \delta$)**: Standard intra-group relative advantage.
- **Case 2: Zero-Variance Starvation ($\sigma_G^2 < \delta$)**: The intra-group term vanishes, but the group-to-batch term activates. If the group achieves $r_i = 1$ while the global batch mean is $\mu_B = 0.3$, the group receives a positive baseline advantage proportional to $(1.0 - 0.3) = +0.7$. If the group achieves $r_i = 0$, it receives a negative advantage $-0.3$.

#### 3. Key Theoretical Assumptions
- **Sub-Gaussian Value Distribution**: The prompt-level value function $V^*(x)$ across batch prompts follows a sub-Gaussian tail, ensuring $(\mu_G - \mu_B)$ is an unbiased estimate of global advantage under intra-group reward homogeneity.

---

### Idea 1.2: Cross-Group Entropy-Regularized Advantage Projection (CGER-AP)

#### 1. Identified Issues & Flaws in Draft
- **Vague Hilbert Projection**: The original text mentioned projecting rewards onto a Hilbert space without defining the Gram matrix operator or regularizer.
- **LaTeX Escape Errors**: Subscripts and symbol math delimiters were improperly rendered.

#### 2. Rigorous Reformulation & Mathematical Solution
CGER-AP constructs a Reproducing Kernel Hilbert Space (RKHS) $\mathcal{H}_K$ across prompt context embeddings $\phi(x_i)$.
For a training batch of $N$ prompt groups, define the cross-prompt Gram matrix:

$$K_{ij} = k(\phi(x_i), \phi(x_j)) = \exp\left(-\frac{\|\phi(x_i) - \phi(x_j)\|^2}{2\ell^2}\right)$$

The cross-group advantage projection vector $\boldsymbol{A}_{\text{proj}}$ is computed by solving the regularized kernel ridge system:

$$\boldsymbol{A}_{\text{proj}} = (K + \lambda I)^{-1} (\boldsymbol{r} - \boldsymbol{\mu}_{\text{group}})$$

To prevent entropy collapse when batch rewards are zero, an orthogonal KL divergence penalty term is added:

$$\mathcal{L}_{\text{CGER}} = \mathcal{L}_{\text{GRPO}}(\boldsymbol{A}_{\text{proj}}) + \beta \cdot \mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

#### 3. Key Theoretical Assumptions
- **Lipschitz Continuity in RKHS**: The optimal value function $V^*(x)$ is $L$-Lipschitz continuous with respect to the embedding metric $d_\mathcal{X}(x_i, x_j) = \|\phi(x_i) - \phi(x_j)\|_{\mathcal{H}_K}$:
  $$\|V^*(x_i) - V^*(x_j)\| \le L \cdot d_\mathcal{X}(x_i, x_j)$$

---

### Idea 1.3: Information-Theoretic Multi-Sample Policy Variance Recovery (IT-MSVR)

#### 1. Identified Issues & Flaws in Draft
- **Mangled LaTeX Notation**: `\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}` was rendered with ASCII control characters (`rac{\pi_	heta(a|s)}{\pi_{	ext{old}}(a|s)}`).
- **RL Action Notation Inconsistency**: Standard language model policy optimization operates on sequence tokens $y_t \sim \pi_\theta(\cdot | x, y_{<t})$, not generic continuous RL state-action pairs $(a|s)$.

#### 2. Rigorous Reformulation & Mathematical Solution
When all rollouts in group $G$ achieve identical binary rewards, IT-MSVR utilizes token-level conditional KL divergence to create a **token compression pseudo-advantage** $\hat{A}_t$:

$$\hat{A}_t = -\eta \left[ \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{ref}}(y_t | x, y_{<t})} - \mathbb{E}_{y_t' \sim \pi_\theta} \left[ \log \frac{\pi_\theta(y_t' | x, y_{<t})}{\pi_{\text{ref}}(y_t' | x, y_{<t})} \right] \right]$$

- When intra-group reward variance $\sigma_G^2 = 0$, IT-MSVR activates $\hat{A}_t$.
- Tokens that diverge heavily from $\pi_{\text{ref}}$ without contributing to higher reward are assigned negative pseudo-advantage $\hat{A}_t < 0$, penalizing verbose or redundant chain-of-thought tokens while preserving accuracy.

#### 3. Key Theoretical Assumptions
- **Redundancy-Surprise Correlation**: Conditional entropy spikes relative to $\pi_{\text{ref}}$ on equal-reward trajectories correlate positively with token redundancy and hallucination risk.

---

### Idea 1.4: Sobolev Gradient Flow Regularization for ZVF Prevention

#### 1. Identified Issues & Flaws in Draft
- **Severely Corrupted LaTeX Operators**: `\|\nabla_\theta \pi_\theta(x_t | x_{<t})\|_{H^1}^2` was mangled to `\|abla_heta...`.
- **Incomplete Sobolev Operator Specification**: Failed to define how the $H^1$ Sobolev gradient $\nabla_{H^1} \mathcal{L}$ relates to the standard $L^2$ policy gradient $\nabla_{L^2} \mathcal{L}$.

#### 2. Rigorous Reformulation & Mathematical Solution
Standard policy gradient optimization computes parameter updates in $L^2(\Omega)$, leading to point-wise parameter jitters during zero-reward regimes.
Sobolev gradient flow Sobolev space $H^1(\Omega)$ norm incorporates first-order sequence derivatives:

$$\|u\|_{H^1}^2 = \|u\|_{L^2}^2 + \|\nabla_{y_t} u\|_{L^2}^2$$

The Sobolev gradient $\nabla_{H^1} \mathcal{L}$ is obtained by solving the operator equation:

$$(I - \gamma \Delta_{\text{seq}}) \nabla_{H^1} \mathcal{L} = \nabla_{L^2} \mathcal{L}$$

where $\Delta_{\text{seq}}$ is the discrete sequence Laplacian operator across hidden state representations:

$$\Delta_{\text{seq}} h_t = h_{t+1} - 2h_t + h_{t-1}$$

This smooths out loss landscape transitions, preventing high-frequency loss spikes when advantages vanish.

#### 3. Key Theoretical Assumptions
- **Sobolev Riemannian Smoothness**: The policy probability manifold is a smooth Riemannian manifold embedded in Sobolev space $H^1(\Omega)$, ensuring continuous gradient flow along parameter trajectories.

---

### Idea 1.5: Dynamic Variance-Constrained Natural GRPO (DVC-NGRPO)

#### 1. Identified Issues & Flaws in Draft
- **Lack of Mathematical Precision for FIM Degeneracy**: Did not explicitly formalize how Fisher Information Matrix (FIM) degeneracy occurs under ZVF.
- **Unclear Damped Neumann Formulation**: Mentioned Neumann series without defining the expansion terms or condition-number-dependent trust region.

#### 2. Rigorous Reformulation & Mathematical Solution
The empirical Fisher Information Matrix $F_\theta$ is defined as:

$$F_\theta = \mathbb{E}_{x, y \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(y|x) \nabla_\theta \log \pi_\theta(y|x)^T \right]$$

Under ZVF ($\sigma_G^2 = 0$), rollout responses collapse to near-identical token paths, causing the condition number $\kappa(F_\theta) = \frac{\lambda_{\max}(F_\theta)}{\lambda_{\min}(F_\theta)} \to \infty$ to explode.

DVC-NGRPO introduces:
1. **Dynamic Trust-Region Boundary**:
   $$\delta_t = \delta_0 \cdot \min\left(1, \frac{\kappa_0}{\kappa(F_\theta)}\right)$$
2. **Damped Neumann Series Inversion**:
   $$(F_\theta + \lambda I)^{-1} = \frac{1}{\lambda} \sum_{k=0}^K \left(-\frac{1}{\lambda} F_\theta\right)^k$$
   where the damping factor scales dynamically with matrix trace and reward variance:
   $$\lambda = \gamma \frac{\operatorname{Tr}(F_\theta)}{d} + \sigma_G^{-2} \cdot \mathbf{1}_{\sigma_G^2 < \epsilon}$$

#### 3. Key Theoretical Assumptions
- **Bounded Spectral Norm**: The Hessian norm $\|\nabla_\theta^2 \mathcal{L}\|_2$ is bounded along local geodesics, guaranteeing numerical convergence of the $K$-term Neumann series expansion under $\lambda$-damping.

---

## Summary of File Modifications

The file `/Users/arvind/Developer/agentic_repos/tinker-rl-lab/50_research_ideas_catalog.md` has been directly updated to reflect all corrected LaTeX math expressions, sound mathematical mechanisms, explicit theoretical assumptions, and standardized RL notation for Category 1 (Ideas 1.1 - 1.5).

All changes pass fail-closed verification.
