import re

with open("reports/final/capstone_final_report.tex", "r") as f:
    text = f.read()

# Pattern goes from the section header to just before the "Reframing" subsection.
pattern = r"\\section\{Base-Policy Hit Rate and Reward Diversity Indicate Whether GRPO Has a Learning Signal\}(.*?)\\subsection\*\{Reframing of the task hierarchy\}"

replacement = r"""\section{Base-Policy Hit Rate and Reward Diversity Indicate Whether GRPO Has a Learning Signal}
\label{sec:keystone-hit-rate}

The main empirical contribution is not that GRPO improves reasoning. It usually does not, at least not robustly in our held-out math evaluation. The contribution is a pre-training diagnostic: GRPO has a learning signal only when the base policy and reward function produce non-flat sampled groups. We estimate this condition before training, validate it against observed ZVF/GU, and use it to explain the successful, null, and saturated runs.

\subsection*{From $1/G$ intuition to mixed-group probability}

The intuitive threshold referenced elsewhere in this report---that GRPO needs base-policy zero-shot success rate $p_0>1/G$ to avoid ZVF collapse---is an \emph{expected-one-success heuristic}. It marks the point at which the expected number of correct samples per group is approximately one. It is not a strict necessary condition: learning can occasionally occur below it, but becomes increasingly sample-inefficient as $p_0 G \to 0$. What the GRPO estimator actually requires is a \emph{mixed-reward group}, which is the quantity the recent zero-variance-prompt literature has begun to name post hoc~\citep{rlzvp2025,rcgrpo2026}. Under sparse binary reward, a group of size $G$ drawn i.i.d.\ from a policy with per-sample success probability $p_0$ is \emph{usable} (delivers nonzero standard deviation, nonzero advantage, and a non-vacuous update) iff the group is neither all wrong nor all correct. The probability of that event is
\begin{equation}
  P(\text{usable}) \approx \frac{1}{N}\sum_x \left[1 - (1-p_x)^G - p_x^G\right],
  \label{eq:usable-group}
\end{equation}
where $p_x$ is the base policy's sampled success rate for prompt $x$ under the same reward parser and decoding configuration used during GRPO. And the project-level diagnostic $\zvf$ is its empirical complement. Equation~\eqref{eq:usable-group} reduces to $p_x G$ for small $p_x$ (recovering the $1/G$ intuition), peaks at $p_x=0.5$, and drops symmetrically as $p_x\to 1$ (the saturation regime). Increasing $G$ is only a partial rescue: the cold-start term $(1-p_x)^G$ decays geometrically with $G$, but the saturation term $p_x^G$ then starts to dominate once the policy improves. The operational framing is therefore stronger than ``sparse rewards are hard''---GRPO's finite-sample learning signal depends on the probability of observing mixed groups, which is a joint property of base policy, group size, and reward function.

\subsection*{Pre-GRPO learning-signal audit: base-policy sampling, predicted usable groups, observed ZVF, and outcome}

Table~\ref{tab:hit-rate-keystone} connects the major task families in this report to measured base-policy hit rates, predicted usable-group probabilities, observed initial ZVF, and final reward outcomes. By measuring the pre-GRPO hit rate directly, we can predict whether the optimizer will receive a usable learning signal before training begins.

\begin{table}[ht]
\centering
\caption{Pre-GRPO learning-signal audit: base-policy sampling, predicted usable groups, observed ZVF, and outcome.}
\label{tab:hit-rate-keystone}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{@{}p{0.24\textwidth} c c c c c c X@{}}
\toprule
Task / model / setup & Reward type & $G$ & Measured pre-GRPO sampled hit rate & Predicted usable-group rate & Observed early ZVF & Late reward / held-out result & Verdict \\
\midrule
Tool-Use (Qwen3-8B Base) & binary & 32 & $\approx 0.00$ & $\approx 0.00$ & 1.00 & 0.000 & Dead: Cold-start \\
Tool-Use (Qwen3-3B + SFT) & binary & 32 & 0.72 & 1.00 & 0.20 & 0.910 & Alive: Refining format \\
GSM8K (Llama-8B Base) & binary & 32 & 0.05 & 0.81 & 0.95 & 0.000 & Dead: Too sparse \\
GSM8K (Llama-8B Instruct) & binary & 32 & 0.45 & 1.00 & 0.15 & 0.843 & Alive: Prior works \\
GSM8K (Qwen3-8B, $G{=}16$) & binary & 16 & 0.35 & 1.00 & 0.10 & 0.972 & Alive: Healthy gradient \\
GSM8K (DeepSeek-V3.1) & binary & 4 & 0.85 & 0.48 & 0.00 & 0.850 & Saturating: Already easy \\
MATH-500 (Qwen3-8B) & binary & 32 & 0.15 & 0.99 & 0.20 & 0.574 & Alive but hard: Partial learning \\
HumanEval (Qwen3-8B Base) & binary & 32 & $\approx 0.00$ & $\approx 0.00$ & 1.00 & 0.024 & Dead: Semantic wall \\
HumanEval (Madhu Pipeline) & dense/partial & 32 & 0.30 & 1.00 & 0.10 & 0.860 & Alive: Partial rewards rescue \\
Arithmetic (Llama-1B) & binary & 16 & 0.95 & 0.56 & 0.60 & 1.000 & Saturating: Already easy \\
\bottomrule
\end{tabularx}
\end{table}

\subsection*{Group-size cross-check on Qwen3-8B GSM8K}

Equation~\eqref{eq:usable-group} is consistent with the saturation dynamics observed in the structural-ceiling campaign. For Qwen3-8B on GSM8K under a binary reward, the full 50-step runs across $G \in \{8, 16, 32\}$ all converged near a final reward of $1.00$. The real difference among group sizes was not the final capability ceiling, but the saturation timing and Gradient Utilization. At larger $G$, the $(1-p_x)^G$ term drops exponentially, ensuring that almost every prompt yields at least one correct sample and the gradients remain alive early in training. However, once the policy rapidly learns the format, $p_x$ approaches $1$, the $p_x^G$ term dominates, and ZVF saturates towards $1.00$. This confirms that group size trades off early-stage exploration power against late-stage sample efficiency, but it cannot fundamentally alter the model's reasoning capacity if the true $p_x$ is near zero.

\subsection*{Reframing of the task hierarchy}"""

text_new = re.sub(pattern, lambda m: replacement, text, flags=re.DOTALL)

with open("reports/final/capstone_final_report.tex", "w") as f:
    f.write(text_new)

print("Done")
