# Self-review claim and submission flags

Generated deterministically by `self_review_corpus.py`. Every included
source file appears in `FILE_REVIEW.tsv`; this document expands only files
with active claim-risk, TODO, placeholder, or unresolved-citation flags.
A flag is a review location, not automatically a defect.

## `platform_hybrid/paper/acm_main.tex`

Consumers: R01

- L178 [claim] Reinforcement learning (RL) has become a dominant approach for post-training
- L191 [claim] We report three conservative findings. First, the large measured gap between
- L196 [claim] superiority. Second, although online GSM8K training reward improves with GRPO,
- L218 [placeholder] \fbox{\parbox{0.95\linewidth}{\centering\small\vspace{1em}\textit{[Figure placeholder: tikz/architecture.pdf pending build.]}\vspace{1em}}}
- L234 [claim] theory, adaptive-controller proposal, or the new PPO/SAO synthesis. Those
- L277 [claim] PPO~\cite{schulman2017proximal} has been the standard algorithm for RLHF since
- L285 [claim] Henderson et al.~\cite{henderson2018deep} first highlighted reproducibility
- L345 [claim] Stable Baselines3 & PPO & Classic RL \\
- L346 [claim] CleanRL & PPO (single-file) & Research RL \\
- L347 [claim] Tianshou & PPO & Modular RL \\
- L348 [claim] PufferLib & PPO (high-throughput) & High-perf RL \\
- L349 [claim] rl\_games (NVIDIA) & PPO (GPU-optimized) & High-perf RL \\
- L358 [claim] Table~\ref{tab:hyperparams} maps Tinker PPO defaults to each library's native parameters.
- L362 [claim] \caption{Hyperparameter mapping across libraries (PPO defaults).}
- L435 [claim] We compare GRPO and PPO \emph{implementations} on a common arithmetic task
- L442 [claim] property of PPO as an algorithm.
- L447 [claim] SB3/CleanRL/Tianshou-PPO rows are reproduced from a committed five-seed Modal run
- L461 [claim] SB3 (PPO) & 0.010 $\pm$ 0.002 & [0.005, 0.015] & N/A \\
- L462 [claim] CleanRL (PPO) & 0.009 $\pm$ 0.002 & [0.005, 0.013] & N/A \\
- L463 [claim] Tianshou (PPO) & 0.006 $\pm$ 0.002 & [0.001, 0.011] & N/A \\
- L587 [claim] First, GRPO exhibits a two-phase learning pattern: during early steps (phases
- L673 [claim] offers is deliberately narrow. The current release most strongly supports
- L674 [claim] three conservative claims. First, the large measured gap between LLM-native
- L675 [claim] GRPO stacks and classic-RL PPO libraries is an implementation and
- L678 [claim] evidence that any one algorithm is superior. Second, trainability in our
- L683 [claim] not statistically significant (83.3\% vs.\ 82.0\%, $p{=}0.26$), so strong

## `platform_hybrid/paper/ethics_statement.tex`

Consumers: U01

- L43 [claim] by our release to be small for three reasons. First, GRPO at our training
- L48 [claim] ($p{=}0.26$, not statistically significant).
- L122 [claim] Modal H100 & PPO baselines (Qwen3-8B, Llama-3.1-8B)& 2 & \$12--18 \\
- L212 [claim] Bengaluru--Delhi (\textasciitilde 300\,kg). The dominant uncertainty is
- L259 [claim] We use the public release as-is; specifically, we use the first 35 prompts
- L360 [placeholder] \texttt{.env.example} placeholder only. The real key is stored in the
- L385 [claim] variance estimates and do not support significance testing; we report
- L400 [claim] it means our claims about PPO vs.\ GRPO and TRL vs.\ Tinker are

## `platform_hybrid/paper/main.tex`

Consumers: U01

- L161 [claim] adaptive control, PPO, and SAO remain prospective extensions unless explicitly
- L225 [placeholder] placeholders. The viva slides use the second roster; the
- L226 [claim] cross-RL-library evidence in $F_3$ uses the first.
- L239 [claim] Stable Baselines3 & PPO & Classic RL \\
- L240 [claim] CleanRL & PPO (single-file) & Research RL \\
- L241 [claim] Tianshou & PPO & Modular RL \\
- L242 [claim] PufferLib & PPO (high-throughput) & High-perf RL \\
- L243 [claim] rl\_games (NVIDIA) & PPO (GPU-optimized) & High-perf RL \\
- L252 [claim] Table~\ref{tab:hyperparams} shows the mapping from Tinker PPO defaults to each library's
- L257 [claim] \caption{Hyperparameter mapping across libraries (PPO defaults).}
- L348 [claim] Tier-A claims (F3 PPO-vs.-GRPO heterogeneity and the five-seed TRL GRPO
- L476 [claim] only $F_3$ (PPO/GRPO heterogeneity on classic-RL libraries on
- L487 [claim] 2 & PPO vs GRPO on Llama-3.1-8B (Welch t-test)$^{\ddag}$ & 3.924e-10 & 0 & $\checkmark$ \\
- L491 [claim] 6 & PPO vs GRPO on Llama-3.1-8B (Mann-Whitney)$^{\ddag}$ & 3.29e-05 & 0.00011 & $\checkmark$ \\
- L498 [claim] 13 & GRPO vs PPO Stability Index (t-test) & 0.005 & 0.007219 & $\checkmark$ \\
- L502 [claim] 17 & GRPO vs PPO Peak-to-Tail Drift (t-test) & 0.018 & 0.02076 & $\checkmark$ \\
- L505 [claim] 20 & PPO vs GRPO on Qwen3-8B (Welch t-test)$^{\ddag}$ & 0.7605 & 0.7605 & $\times$ \\
- L520 [claim] are the dominant variance sources, consistent with our central thesis. Model
- L536 [claim] PPO baseline training on Qwen3-8B and Llama-3.1-8B; full HumanEval
- L539 [claim] provide the low-level GPU access required for PPO value-model training.
- L557 [claim] PPO baselines from the Modal H100 cluster. Peak reward and last-10-step mean
- L563 [claim] \caption{\textbf{Main Results:} GRPO and PPO training reward on GSM8K across model scales.
- L618 [claim] \multicolumn{7}{l}{\textit{PPO on Modal H100 (GSM8K)}} \\
- L620 [claim] PPO & Modal & Qwen3-8B & 8B & 30 & 75.0\% & 22.5\% \\
- L621 [claim] PPO & Modal & Llama-3.1-8B-Inst & 8B & 30 & 100\% & 97.5\% \\
- L654 [placeholder] \fbox{\parbox{0.80\linewidth}{\centering\small\vspace{1em}\textit{[Figure placeholder: framework_comparison.pdf pending regeneration. See \texttt{paper/sections/framework_gap.tex}.]}\vspace{1em}}}
- L670 [claim] that any one framework is algorithmically superior. Fully matched
- L678 [claim] We compare GRPO and PPO \emph{implementations} on a common arithmetic task
- L679 [claim] (Table~\ref{tab:results_arithmetic}). The TRL-GRPO and the three classic-RL PPO
- L689 [claim] property of PPO as an algorithm (see \S\ref{sec:discuss_impl} and the
- L696 [claim] SB3/CleanRL/Tianshou-PPO rows are reproduced from a committed five-seed Modal run
- L710 [claim] SB3 (PPO) & 0.010 $\pm$ 0.002 & [0.005, 0.015] & N/A \\
- L711 [claim] CleanRL (PPO) & 0.009 $\pm$ 0.002 & [0.005, 0.013] & N/A \\
- L712 [claim] Tianshou (PPO) & 0.006 $\pm$ 0.002 & [0.001, 0.011] & N/A \\
- L725 [placeholder] \fbox{\parbox{0.95\linewidth}{\centering\small\vspace{1em}\textit{[Figure placeholder: learning_curves.pdf pending regeneration. See \texttt{paper/sections/learning_curves.tex}.]}\vspace{1em}}}
- L739 [placeholder] \fbox{\parbox{0.80\linewidth}{\centering\small\vspace{1em}\textit{[Figure placeholder: performance\_profiles.pdf pending regeneration. See \texttt{scripts/make\_paper\_figures.py::fig\_performance\_profiles}.]}\vspace{1em}}}
- L864 [placeholder] \fbox{\parbox{0.80\linewidth}{\centering\small\vspace{1em}\textit{[Figure placeholder: scaling.pdf pending regeneration. See \texttt{paper/sections/scaling.tex}.]}\vspace{1em}}}
- L892 [claim] asymptotic reward ($R_{\max} \approx 0.83$); classical PPO baselines (SB3,
- L923 [placeholder] \fbox{\parbox{0.80\linewidth}{\centering\small\vspace{1em}\textit{[Figure placeholder: scaling\_law\_figure.pdf pending regeneration. See \texttt{paper/sections/scaling.tex}.]}\vspace{1em}}}
- L936 [placeholder] \fbox{\parbox{0.80\linewidth}{\centering\small\vspace{1em}\textit{[Figure placeholder: scaling\_params\_figure.pdf pending regeneration. See \texttt{paper/sections/scaling.tex}.]}\vspace{1em}}}
- L939 [claim] size (log scale). Pearson correlations shown; both are statistically significant
- L977 [placeholder] \fbox{\parbox{0.80\linewidth}{\centering\small\vspace{1em}\textit{[Figure placeholder: sensitivity\_heatmap.pdf pending regeneration. See \texttt{paper/sections/sensitivity.tex}.]}\vspace{1em}}}
- L1003 [placeholder] \fbox{\parbox{0.95\linewidth}{\centering\small\vspace{1em}\textit{[Figure placeholder: wave6\_sensitivity.pdf pending regeneration. See \texttt{paper/figures/wave6\_sensitivity.py}.]}\vspace{1em}}}
- L1092 [claim] First, GRPO exhibits a two-phase learning pattern: during early steps (phases
- L1239 [claim] \subsection{PPO vs.~GRPO: Matched-Task, Stack-Confounded Comparison}
- L1240 [claim] \label{sec:ppo_grpo}
- L1243 [claim] Prior RL post-training literature rarely compares PPO and GRPO on matched tasks,
- L1244 [claim] models, and step budgets; PPO results are typically reported on different model
- L1246 [claim] not close---this gap by training PPO baselines on the Modal H100 cluster on the
- L1249 [claim] the Tinker managed API while every PPO arm uses Modal H100, so the two stacks
- L1260 [claim] \caption{PPO vs.~GRPO Comparison: GSM8K training reward (peak and last-10-step mean).
- L1261 [claim] All GRPO runs use Tinker API (single seed, 30 steps); PPO runs use Modal H100 (single seed, 30 steps).
- L1263 [claim] \label{tab:ppo_grpo}
- L1269 [claim] PPO (Modal H100) & Qwen3-8B & 30 & 75.0\% & 22.5\% \\
- L1272 [claim] PPO (Modal H100) & Llama-3.1-8B-Instruct & 30 & 100\% & \textbf{97.5\%} \\
- L1279 [claim] \IfFileExists{figures/v2/ppo_vs_grpo.pdf}{
- L1280 [claim] \includegraphics[width=0.95\linewidth]{figures/v2/ppo_vs_grpo.pdf}
- L1282 [placeholder/claim] \fbox{\parbox{0.95\linewidth}{\centering\small\vspace{1em}\textit{[Figure placeholder: ppo\_vs\_grpo.pdf pending regeneration. See \texttt{paper/sections/ppo\_vs\_grpo.tex}.]}\vspace{1em}}}
- L1284 [claim] \caption{Step-level reward comparison between GRPO (Tinker) and PPO (Modal H100) on Qwen3-8B GSM8K. Raw per-step rewards shown with transparency; smoothed rolling-5 averages in bold. The shaded region marks the last-10 evaluation window. GRPO achieves a higher last-10 average (34
- L1285 [claim] \label{fig:ppo_vs_grpo}
- L1288 [claim] Figure~\ref{fig:ppo_vs_grpo} reveals the step-level dynamics behind Table~\ref{tab:ppo_grpo}.
- L1290 [claim] (1) PPO requires a value model adding $\sim$40\% memory overhead and
- L1292 [claim] (2) On Qwen3-8B, GRPO reached a higher last-10 average (34.4\%) than PPO (22.5\%),
- L1293 [claim] with GRPO exhibiting lower per-step volatility (CV 0.46 vs.~1.00 for PPO);
- L1296 [claim] (3) On Llama-3.1-8B, PPO reached a higher last-10 average (97.5\%) than GRPO
- L1301 [claim] API and PPO on Modal H100 (Appendix~\ref{sec:appendix:framework-configs})---so
- L1313 [claim] Mann-Whitney statistics are not valid algorithm-causal estimates and
- L1327 [claim] GRPO (Tinker) vs PPO (Modal H100) on Qwen3-8B & $1$ & $1$ & ---$^{\ddag}$ & --- & ---$^{\ddag}$ & --- & --- \\
- L1328 [claim] PPO (Modal H100) vs GRPO (Tinker) on Llama-3.1-8B-Inst & $1$ & $1$ & ---$^{\ddag}$ & --- & ---$^{\ddag}$ & --- & --- \\
- L1344 [placeholder] \fbox{\parbox{0.80\linewidth}{\centering\small\vspace{1em}\textit{[Figure placeholder: effect\_sizes\_forest.pdf pending regeneration. See \texttt{paper/sections/effect\_sizes.tex}.]}\vspace{1em}}}
- L1347 [claim] comparisons (the two single-seed PPO-vs-GRPO contrasts carry no valid $d$ and are
- L1355 [claim] \paragraph{A same-stack control: when the stack is held constant, PPO and GRPO
- L1358 [claim] which PPO and GRPO share \emph{everything} except the advantage estimator:
- L1361 [claim] PPO-style clipped surrogate and number of inner epochs, and an identical
- L1363 [claim] group-relative mean vs.\ PPO's learned value head) and PPO's value loss, across
- L1365 [claim] \texttt{experiments/modal/modal\_samestack\_ppo\_grpo.py}; artifact
- L1366 [claim] \texttt{experiments/results/samestack\_ppo\_grpo.json}).
- L1368 [claim] (Table~\ref{tab:samestack}): GRPO $0.990 \pm 0.004$ vs.\ PPO $0.992 \pm 0.003$,
- L1376 [claim] reward varies less across seeds (SE $0.007$ vs.\ $0.050$ for PPO), echoing the
- L1380 [claim] cross-stack ``PPO vs.\ GRPO'' differences are stack- rather than algorithm-driven.
- L1390 [claim] PPO (value-head baseline) & $0.992 \pm 0.003$ & $0.918 \pm 0.050$ & 5 \\
- L1392 [claim] \multicolumn{4}{l}{\small Paired GRPO$-$PPO held-out: $-0.002$, $t{=}{-}1.0$, $\mathrm{df}{=}4$, $p{=}0.37$ (n.s.).} \\
- L1395 [claim] \caption{\textbf{Same-stack PPO vs.\ GRPO control} on \texttt{Qwen2.5-0.5B} /
- L1452 [claim] classic RL libraries (SB3, CleanRL, Tianshou), whose PPO
- L1458 [claim] GRPO runs in our log have lower SI/PTD than the pooled PPO-labelled
- L1462 [claim] algorithm-level stability claim.} The PPO pool here is dominated by
- L1513 [claim] diagnostic of signal degeneracy rather than an independent causal predictor of
- L1519 [claim] To our knowledge this is the first \emph{cross-scale} (3\,B--671\,B) measurement
- L1549 [placeholder] \fbox{\parbox{\textwidth}{\centering\small\vspace{1em}\textit{[Figure placeholder: zvf\_heatmap.pdf pending regeneration. See \texttt{paper/sections/zvf.tex}.]}\vspace{1em}}}
- L1561 [placeholder] \fbox{\parbox{\textwidth}{\centering\small\vspace{1em}\textit{[Figure placeholder: zvf\_correlation.pdf pending regeneration. See \texttt{paper/sections/zvf.tex}.]}\vspace{1em}}}
- L1576 [claim] Wu et al.~\cite{wu2025grpo_dpo} prove that GRPO is algebraically equivalent to
- L1591 [claim] moderate accuracies $p \in [0.2, 0.8]$; the gain from $G=16 \to G=32$ is less
- L1615 [placeholder] \fbox{\parbox{\textwidth}{\centering\small\vspace{1em}\textit{[Figure placeholder: group\_size\_ablation.pdf pending regeneration. See \texttt{paper/sections/group\_size.tex}.]}\vspace{1em}}}
- L1626 [claim] universal optimal group size. A separate token-budget reanalysis fits an apex
- L1630 [claim] finding across regimes is an inverted-U---intermediate $G$ outperforms the
- L1683 [claim] peak reward = 100\%, last-10 mean = 85.0\%, first-5 mean = 85.0\%.
- L1704 [claim] Furthermore, analyzing the training trajectories using an exponential saturation model $R(t) = R_0 + (R_{max} - R_0)(1 - e^{-\lambda t})$, we observe distinct scaling dynamics. For example, frontier models like DeepSeek-v3.1 achieve high capacity and rapid saturation ($R_{max} = 
- L1708 [claim] autoregressive LM), \emph{not} evidence about PPO as an algorithm; our same-stack
- L1709 [claim] control (Table~\ref{tab:samestack}) finds PPO and GRPO indistinguishable, so we
- L1728 [claim] To further understand the failure modes of legacy architectures compared to the sample-efficient Tinker framework, we conduct a parameter-level sparsity audit (Figure~\ref{fig:lora_sparsity}). By tracing the variance of gradients across the Transformer layers during PPO updates, 
- L1733 [claim] \caption{Subnetwork Parameter Sparsity: Tinker GRPO leverages sparse, high-impact weight updates, avoiding the dense gradient noise that destabilizes legacy PPO implementations.}
- L1759 [claim] short-horizon setting, not as a clean framework-only causal estimate. The
- L1773 [claim] Optimal last-10 performance occurs at the intermediate $G{=}4$ (52.1\%); $G{=}2$ (37.5\%), $G{=}8$ (34.4\%), and $G{=}16$ (38.0\%) are lower --- an inverted-U consistent with the measured group-size sweep (Appendix~\ref{sec:appendix:group-size-measured}).
- L1778 [claim] unsupported and has been removed.
- L1839 [claim] limitation of serverless ML platforms that do not support long-running
- L1948 [claim] relates to the \emph{verbosity trap}: the model first learns that longer outputs are associated
- L1964 [claim] compared with $0.785 \pm 0.297$ for PPO-labelled runs ($n = 17$). Among the
- L1975 [claim] index than the PPO-labelled pool, but the PPO pool is dominated by
- L1978 [claim] PPO-on-LLM length-bias comparison and do not claim GRPO is more or
- L1979 [claim] less stable than PPO at the algorithmic level. A minority
- L1991 [placeholder] \fbox{\parbox{0.80\linewidth}{\centering\small\vspace{1em}\textit{[Figure placeholder: reward\_stability.pdf pending regeneration. See \texttt{paper/sections/reward\_stability.tex}.]}\vspace{1em}}}
- L1993 [claim] \caption{Reward stability analysis across experiments. GRPO runs (blue) and PPO runs
- L2088 [claim] ordering of PPO and GRPO: PPO reached a higher last-10 reward than GRPO on
- L2091 [claim] this a reversal: the Qwen3-8B arm is null and underpowered, and both PPO/GRPO
- L2092 [claim] contrasts are single-seed and stack-confounded (GRPO on Tinker, PPO on Modal
- L2096 [claim] baseline differs) finds \emph{no} significant PPO-vs-GRPO difference in held-out
- L2106 [claim] GRPO replaces the value-function baseline of PPO with a within-group reward
- L2120 [claim] comparable tasks), and PPO should be preferred below it.
- L2130 [claim] PPO's value function, trained jointly on the task, can absorb some of this
- L2148 [claim] supports this proximity argument.
- L2158 [claim] running PPO on the same arithmetic task) sits near the ceiling of
- L2165 [claim] algorithm (GRPO vs.\ PPO) with implementation-layer (LLM-native vs.\
- L2169 [claim] running PPO as distributed, fail to train on short-horizon LLM
- L2177 [claim] not for a categorical claim about PPO in the abstract.
- L2180 [claim] SB3 PPO row at $0.010 \pm 0.002$ accuracy (reproduced from the committed
- L2184 [claim] failure from an algorithmic failure of PPO in the abstract.
- L2185 [claim] Published claims of the form ``PPO fails on LLM reasoning'' that
- L2188 [claim] exact PPO implementation used, verify it against a known-good
- L2228 [claim] \citet{wu2025grpo_dpo} prove that at $G = 2$, GRPO's advantage reduces to a
- L2278 [claim] and Hugging Face for model hosting. This work was supported in part by PES
- L2319 [claim] Modal H100 & PPO baselines (Qwen3-8B, Llama-8B) & 2 & $\sim$96 \\
- L2362 [placeholder] \fbox{\parbox{0.80\linewidth}{\centering\small\vspace{1em}\textit{[Figure placeholder: comparison\_bars.pdf pending regeneration. See \texttt{paper/sections/comparison\_bars.tex}.]}\vspace{1em}}}
- L2367 [claim] reports group size (TRL GRPO, legacy PPOs = 5 seeds each; Modal
- L2368 [claim] PPO-REINFORCE = 4 runs; Tinker GRPO frontier = 9 runs; team members =
- L2379 [placeholder] \fbox{\parbox{0.45\linewidth}{\centering\small\vspace{1em}\textit{[Figure placeholder: old\_trl\_seeds.png pending regeneration.]}\vspace{1em}}}
- L2466 [claim] preference (GRPO vs.\ PPO), and (iv) covers scales from 0.6B to

## `platform_hybrid/paper/neurips_2026_variants/ethics_statement_zvf.tex`

Consumers: R02,R03,R04

- L43 [claim] by our release to be small for three reasons. First, GRPO at our training
- L47 [claim] adds \(+1.3\) percentage points ($p{=}0.26$, not statistically significant).
- L84 [claim] selection (the optimal $G$ depends on total token budget; see
- L123 [claim] Modal H100 & PPO baselines (Qwen3-8B, Llama-3.1-8B)& 2 & \$12--18 \\
- L221 [claim] the same region (\textasciitilde 300\,kg). The dominant uncertainty is
- L269 [claim] We use the public release as-is; specifically, we use the first 35 prompts
- L376 [placeholder] \texttt{.env.example} placeholder only. The real key is stored in the
- L401 [claim] variance estimates and do not support significance testing; we report
- L416 [claim] it means our claims about PPO vs.\ GRPO and TRL vs.\ Tinker are

## `platform_hybrid/paper/neurips_2026_variants/main_dnb.tex`

Consumers: R04

- L122 [claim] tier and to the artifact files that support it; a per-stack auditability
- L157 [claim] We treat this association as descriptive (B-grade), not causal: ZVF is
- L179 [claim] $\widehat{\mathrm{TBGU}}$-optimal $G$ shifts rightward with total tokens

## `platform_hybrid/paper/neurips_2026_variants/main_workshop.tex`

Consumers: R03

- L109 [claim] causal predictor.
- L195 [claim] protocol. The $+1.3$pp delta is not statistically significant

## `platform_hybrid/paper/neurips_2026_variants/main_zvf.tex`

Consumers: R02

- L183 [claim] operational action. The first three rows describe per-step in-task
- L383 [claim] ecosystem. Institutional support statements are redacted for blind review.
- L556 [claim] classic RL libraries (SB3, CleanRL, Tianshou), whose PPO
- L562 [claim] GRPO experiments show significantly lower instability than classic-RL PPO:

## `platform_hybrid/paper/neurips_2026_variants/sections/abstract_dnb.tex`

Consumers: R04

- L33 [claim] instrumentation, not a causal claim.

## `platform_hybrid/paper/neurips_2026_variants/sections/appendix_full_results_dnb.tex`

Consumers: R04

- L34 [claim] \caption{Hyperparameter mapping across libraries (PPO defaults).}
- L76 [claim] training reward; all other Tinker rows and the Modal H100 PPO rows are
- L87 [claim] are single-seed and reported descriptively. Modal H100 PPO rows are
- L129 [claim] GRPO & Tinker & Qwen3-8B (G=32)$^\dagger$ & 8B & 30 & 67.2\% & 38.8\% \\
- L142 [claim] \multicolumn{7}{l}{\textit{PPO on Modal H100 (GSM8K)}} \\
- L144 [claim] PPO & Modal & Qwen3-8B & 8B & 30 & 75.0\% & 22.5\% \\
- L145 [claim] PPO & Modal & Llama-3.1-8B-Inst & 8B & 30 & 100\% & 97.5\% \\

## `platform_hybrid/paper/neurips_2026_variants/sections/appendix_zvf_formalization.tex`

Consumers: R02,R03,R04

- L37 [claim] $\{p_g\}$, not only on its first moment. Two populations can therefore
- L50 [claim] \emph{not} contain the per-group step-level rollout logs needed to support
- L54 [claim] any implication that the current artifact proves a new partial-correlation
- L58 [claim] supported by the released evidence: in outcome-reward GRPO, high ZVF
- L63 [claim] we do not establish a causal link: the within-task stratified analysis in
- L64 [claim] Section~\ref{sec:zvf-analysis} does not support a within-task predictive
- L67 [claim] not a performance-causal one.
- L88 [claim] $K \ge 4$ and non-saturated reward support -- ZVF is a cheap, model-agnostic

## `platform_hybrid/paper/neurips_2026_variants/sections/artifact_card_dnb.tex`

Consumers: R04

- L21 [claim] Modal H100 / CleanRL & PPO (synthetic math) & math gen & 5 & \checkmark & \checkmark & --- & --- & A \\
- L22 [claim] Modal H100 / SB3 & PPO (synthetic math) & math gen & 5 & \checkmark & \checkmark & --- & --- & A \\
- L23 [claim] Modal H100 / Tianshou & PPO (synthetic math) & math gen & 5 & \checkmark & \checkmark & --- & --- & A \\

## `platform_hybrid/paper/neurips_2026_variants/sections/checklist_anon.tex`

Consumers: R04

- L89 [claim] are claimed. GRPO and PPO objectives stated in
- L283 [claim] (Section~\ref{sec:carbon}; the dominant uncertainty is the Tinker

## `platform_hybrid/paper/neurips_2026_variants/sections/conclusion_dnb.tex`

Consumers: R04

- L8 [claim] algorithms, and model families. The current release supports three
- L11 [claim] read as causal predictors independent of reward mean and group size; (ii)
- L13 [claim] released case studies, not isolated causal effects; and (iii) cross-stack
- L14 [claim] PPO/GRPO rankings are descriptive because policy architecture and task encoding

## `platform_hybrid/paper/neurips_2026_variants/sections/conclusion_zvf.tex`

Consumers: R02

- L49 [claim] PPO/GRPO comparisons in a single open stack, and held-out evaluation

## `platform_hybrid/paper/neurips_2026_variants/sections/intro_dnb.tex`

Consumers: R04

- L12 [claim] compare across stacks. PPO~\citep{schulman2017proximal},
- L19 [claim] \paragraph{Artifact-first contribution.}
- L31 [claim] these support the only inferential claims in the paper.
- L49 [claim] The artifact supports two A-grade results stated conservatively. First, the
- L52 [claim] a descriptive diagnostic, not as a standalone causal predictor of

## `platform_hybrid/paper/neurips_2026_variants/sections/intro_workshop.tex`

Consumers: R03

- L21 [claim] ``first comprehensive'' GRPO benchmark, a frontier ranking, or a definitive
- L37 [claim] PPO~\citep{schulman2017proximal}, GRPO~\citep{shao2024deepseekmath}, and
- L46 [claim] artifacts do and do not support. Discussion prompts for the workshop are

## `platform_hybrid/paper/neurips_2026_variants/sections/intro_zvf.tex`

Consumers: R02

- L12 [claim] PPO~\citep{schulman2017proximal} with a leave-one-out advantage computed
- L62 [claim] program. The adaptive controller and the GRPO--PPO--SAO unification are companion

## `platform_hybrid/paper/neurips_2026_variants/sections/related_work_zvf.tex`

Consumers: R02

- L29 [claim] Proximal Policy Optimization (PPO)~\cite{schulman2017proximal} remains the
- L76 [claim] Bayes shrinkage estimator that guarantees a non-vanishing gradient even in
- L86 [claim] before our ZVF traces were complete; we are rigorous, not first.

## `platform_hybrid/paper/neurips_2026_variants/sections/run_manifest.tex`

Consumers: R02,R03,R04

- L20 [claim] least three completed seeds and support inference statistics. Cells marked
- L42 [claim] A & Modal & cleanrl-ppo & math & gen & 5 & 5 & 97280 & BV & --- & \checkmark & --- \\
- L43 [claim] A & Modal & sb3-ppo & math & gen & 5 & 5 & 100352 & BV & --- & \checkmark & --- \\
- L44 [claim] A & Modal & tianshou-ppo & math & gen & 5 & 5 & 100000 & BV & --- & \checkmark & --- \\
- L82 [claim] B$^\dagger$ & Tinker & qwen3-8b (G=32) & gsm8k & train & 1 & 1 & 30 & BV & --- & \checkmark & --- \\
- L111 [claim] $\dagger$ on the G=32 group-size row marks it as the single-seed cell

## `platform_hybrid/paper/neurips_2026_variants/sections/stratified_diagnostics.tex`

Consumers: R02,R03,R04

- L73 [claim] evidence that the pooled monotone story is not supported once the task

## `platform_hybrid/paper/sections/_shared_methods.tex`

Consumers: P01,P02,P03,P04,P05,P06,P07

- L53 [placeholder] placeholders. Supplementary presentation materials use the second roster; the
- L54 [claim] cross-RL-library evidence in $F_3$ uses the first.
- L67 [claim] Stable Baselines3 & PPO & Classic RL \\
- L68 [claim] CleanRL & PPO (single-file) & Research RL \\
- L69 [claim] Tianshou & PPO & Modular RL \\
- L70 [claim] PufferLib & PPO (high-throughput) & High-perf RL \\
- L71 [claim] rl\_games (NVIDIA) & PPO (GPU-optimized) & High-perf RL \\
- L80 [claim] Table~\ref{tab:hyperparams} shows the mapping from Tinker PPO defaults to each library's
- L85 [claim] \caption{Hyperparameter mapping across libraries (PPO defaults).}
- L179 [claim] Tier-A claims (F3 PPO-vs.-GRPO heterogeneity and the five-seed TRL GRPO
- L307 [claim] only $F_3$ (PPO/GRPO heterogeneity on classic-RL libraries on
- L318 [claim] 2 & PPO vs GRPO on Llama-3.1-8B (Welch t-test)$^{\ddag}$ & 3.924e-10 & 0 & $\checkmark$ \\
- L322 [claim] 6 & PPO vs GRPO on Llama-3.1-8B (Mann-Whitney)$^{\ddag}$ & 3.29e-05 & 0.00011 & $\checkmark$ \\
- L329 [claim] 13 & GRPO vs PPO Stability Index (t-test) & 0.005 & 0.007219 & $\checkmark$ \\
- L333 [claim] 17 & GRPO vs PPO Peak-to-Tail Drift (t-test) & 0.018 & 0.02076 & $\checkmark$ \\
- L336 [claim] 20 & PPO vs GRPO on Qwen3-8B (Welch t-test)$^{\ddag}$ & 0.7605 & 0.7605 & $\times$ \\
- L351 [claim] are the dominant variance sources, consistent with our central thesis. Model
- L367 [claim] PPO baseline training on Qwen3-8B and Llama-3.1-8B; full HumanEval
- L370 [claim] provide the low-level GPU access required for PPO value-model training.

## `platform_hybrid/paper/sections/abstract.tex`

Consumers: U01

- L8 [claim] algorithms (PPO, GRPO, DPO) and a growing set of frameworks (TRL,
- L19 [claim] Three conclusions are supported conservatively. First, Zero-Variance Fraction
- L23 [claim] use it as a descriptive diagnostic rather than a standalone causal predictor.
- L28 [claim] tested, but we do not claim a universal optimal group size. Third, PPO/GRPO
- L33 [claim] is small and not statistically significant (83.3\% vs.\ 82.0\%, $p{=}0.26$), while

## `platform_hybrid/paper/sections/appendix_zvf_formalization.tex`

Consumers: P02,U01

- L37 [claim] $\{p_g\}$, not only on its first moment. Two populations can therefore
- L51 [claim] For that corpus we therefore make no causal partial-correlation claim. To test
- L61 [claim] supported by the released evidence: in outcome-reward GRPO, high ZVF
- L114 [claim] $0.16$), the opposite-sign residual to the low-temperature $0.5$B probe. We thus
- L153 [claim] $K \ge 4$ and non-saturated reward support -- ZVF is a cheap, model-agnostic

## `platform_hybrid/paper/sections/base_vs_instruct_paired.tex`

Consumers: U01

- L78 [claim] directionally positive but not statistically significant at conventional
- L90 [placeholder] placeholders.

## `platform_hybrid/paper/sections/cdh_echo_synthesis.tex`

Consumers: P03,P04

- L26 [claim] PPO$\leftrightarrow$GRPO equivalence) makes a sharp mechanistic
- L49 [claim] normaliser does not amplify $|\rho|$ as the learned PPO critic
- L71 [claim] PPO/GRPO $\leftrightarrow$ Pillar-4's Dr.GR/GR form a single

## `platform_hybrid/paper/sections/checklist.tex`

Consumers: U01

- L28 [claim] 7~libraries, (ii)~model-dependent GRPO/PPO preference, (iii)~frontier
- L31 [claim] proxies for KL-free monitoring. Empirical support is given in
- L34 [claim] interaction in Section~\ref{sec:ppo_grpo}; the frontier regime in
- L89 [claim] are claimed. GRPO and PPO objectives stated in

## `platform_hybrid/paper/sections/conclusion.tex`

Consumers: U01

- L12 [claim] strongly supports three conservative claims. First, ZVF/GU are useful
- L14 [claim] within-group signal; they should not yet be read as standalone causal or
- L20 [claim] justify a universal optimum or a general superiority claim for any one
- L21 [claim] algorithm. Third, PPO-vs.-GRPO rankings and frontier-run stability are
- L27 [claim] statistically significant under our current evaluation. Tool-use and code
- L34 [claim] matched multi-seed PPO/GRPO comparisons in a single open stack; broader

## `platform_hybrid/paper/sections/critic_degeneracy_hypothesis.tex`

Consumers: P03

- L26 [claim] To stress-test the PPO$\leftrightarrow$GRPO equivalence from
- L31 [claim] terminal-reward chain-of-thought, PPO's value head
- L36 [claim] CDH holds, the PPO critic is a noise \emph{amplifier} (its parametric
- L46 [claim] variate, CV(grad\_norm) should be \emph{lower} for PPO than GRPO.
- L47 [claim] We observe GRPO CV$=$1.347 vs PPO CV$=$1.433: PPO is $6\%$ more
- L48 [claim] variable in gradient norm---the opposite of the variance-reduction
- L49 [claim] prediction. The raw gradient norms make the gap vivid: PPO
- L53 [claim] If the critic smooths the trajectory, PPO should have lower
- L54 [claim] within-run variance. We observe GRPO $0.0049$ vs PPO $0.0085$:
- L55 [claim] PPO's per-step reward is $73\%$ noisier than GRPO's. The critic is
- L59 [claim] The paired test gives mean $\Delta_{\text{GRPO-PPO}}=0.0608$,
- L63 [claim] critic is serving as a degenerate control variate, PPO's gradient
- L65 [claim] We observe GRPO $r{=}{-}0.553$ vs PPO $r{=}{-}0.445$ (per-seed mean).
- L66 [claim] GRPO tracks reward $24\%$ \emph{better} than PPO---again the
- L67 [claim] opposite of the variance-reduction prediction.
- L69 [claim] PPO critic is trying to learn $E[R \mid x_\text{prompt}]$, which
- L80 [claim] $r_\text{mean}$ variance. The PPO value head (typically
- L87 [claim] \emph{invalidate} the variance-reduction reading of PPO's critic; the
- L89 [claim] empirical fingerprint of the Critic-Degeneracy Hypothesis: the PPO
- L95 [claim] $R^2{=}0.49$. This recasts the PPO$\leftrightarrow$GRPO equivalence
- L97 [claim] on this benchmark'' but as ``\emph{on outcome-reward RL the PPO
- L102 [claim] $\Delta_{\text{GRPO-PPO}} = -0.002$, $p{=}0.62$ on heldout) is
- L110 [claim] on what any baseline can extract; the PPO critic and the GRPO

## `platform_hybrid/paper/sections/figures_regeneration_note.tex`

Consumers: U01

- L4 [placeholder] placeholder boxes in an earlier draft (\texttt{performance\_profiles.pdf},
- L23 [placeholder] guards to resolve to the real-figure branch rather than the placeholder

## `platform_hybrid/paper/sections/framework_configs_appendix.tex`

Consumers: U01

- L75 [placeholder] dry-run placeholders emitted by the aggregation script when those
- L105 [placeholder] artifact their performance rows are dry-run placeholders rather than new

## `platform_hybrid/paper/sections/frontier_scope_clarification.tex`

Consumers: U01

- L13 [claim] the underlying Tinker-backed runs can support. The frontier evidence consists
- L60 [claim] reported transparently but not used to support scaling laws. ``Seeds'' and
- L70 [claim] Supportive evidence & $\geq 3$ & $\geq 50$ & $14\mathrm{B}$--$32\mathrm{B}$ GSM8K GRPO (partial) & Trend statement \\

## `platform_hybrid/paper/sections/frontier_synthesis_group_size.tex`

Consumers: P03,U01

- L43 [claim] $r_i-r_j$. Two consequences follow. First, any group with zero reward
- L95 [claim] inference compute ($\uparrow$ linearly)---so the compute-optimal group
- L100 [claim] predicts that $G{\approx}32$ is the first genuinely DPO-equivalent
- L103 [claim] (Table~\ref{tab:groupsize-summary}) and gives a first-principles account
- L118 [claim] \emph{ZVF-dominant} if Arm~C recovers $\ge 75\%$ of the gain while
- L119 [claim] Arm~B recovers $\le 25\%$, and variance-dominant if the pattern
- L126 [claim] prove the group baseline carries stabilizing structure orthogonal to
- L138 [claim] to pin \emph{when} the GRPO group mean, a PPO value head, and a DPO
- L161 [claim] yet in the PPO-equivalent regime. At the opposite end the same account
- L172 [claim] the payoff so that the compute-optimal $G$ is finite. This is consistent

## `platform_hybrid/paper/sections/frontier_synthesis_length_bias.tex`

Consumers: P04,U01

- L75 [claim] universal held-out superiority --- exactly the posture Sec.~\ref{sec:length-bias-pillar}
- L78 [claim] \paragraph{The Causal Length-Mediation Protocol, CLMP (frontier synthesis).}
- L82 [claim] stumble into reward. CLMP instead treats trajectory length $L$ as a causal
- L86 [claim] success). In an internal cross-critique the models rejected their own first

## `platform_hybrid/paper/sections/frontier_synthesis_scaling.tex`

Consumers: P01,U01

- L76 [claim] compute-optimal group size is $G^\star = \arg\max_G [\,Y_G(p)/G\,]$. This is a
- L85 [claim] Our same-stack control found GRPO and PPO statistically indistinguishable
- L86 [claim] (paired $\Delta = -0.002$, $p = 0.374$; \texttt{samestack\_ppo\_grpo.json}). The frontier models sharpened this
- L89 [claim] reward parser, token budget) is fixed, PPO and GRPO are performance-equivalent
- L94 [claim] \mathrm{BEI} = \cos\!\bigl(g_{\mathrm{GRPO}}, g_{\mathrm{PPO}}\bigr)\cdot
- L96 [claim] \tfrac{\|g_{\mathrm{PPO}}\|}{\|g_{\mathrm{GRPO}}\|},\
- L97 [claim] \tfrac{\|g_{\mathrm{GRPO}}\|}{\|g_{\mathrm{PPO}}\|}
- L107 [claim] via the group mean. Both claims support the same-stack null (paired
- L119 [claim] $p=0.99$ even $G=32$ gives only $\approx 0.28$. Gemini Deep Think framed the

## `platform_hybrid/paper/sections/frontier_synthesis_zvf.tex`

Consumers: P02,U01

- L37 [claim] have opposite outcome implications. A modest $\rho\approx0.27$ over the
- L91 [claim] with $\hat p_x=K_x/G$, whose first term distinguishes all-correct from
- L121 [claim] First, on our anti-herding falsification: the models cautioned that the

## `platform_hybrid/paper/sections/group_size.tex`

Consumers: P03

- L12 [claim] \section{Group Size: G=4 vs G=32 vs the Broader Sweep}
- L18 [claim] Wu et al.~\cite{wu2025grpo_dpo} prove that GRPO is algebraically
- L22 [claim] rollouts and 21\% of the training time. The opposite intuition---that
- L112 [claim] \paragraph{Does the Wu et al.\ G=2$\sim$G=16 claim generalize to G=4$\sim$G=32?}
- L157 [claim] shape across all four $G$ values, supporting the contrastive reading

## `platform_hybrid/paper/sections/group_size_iter23.tex`

Consumers: P03,U01

- L22 [claim] \paragraph{$T_\text{crit}$ for $G=4$ vs $G=32$.}
- L92 [claim] training; it is not evidence that $G{=}32$ is empirically Pareto-optimal.

## `platform_hybrid/paper/sections/group_size_iter27.tex`

Consumers: P03,U01

- L21 [claim] \emph{``does this scale to G=4 vs G=32, the regime that actually
- L94 [claim] \item \textbf{Reconstructed G=4 vs G=32} (blue, with bootstrap CI
- L109 [claim] The Wu~2025 reference line and the reconstructed G=4 vs G=32 curve
- L114 [claim] G=32 at canonical budgets}. A directly trained matched-budget sweep is
- L157 [claim] non-inferior group. The present evidence supports $G{=}2$ on the measured
- L175 [claim] reference line (gray, R$=0.976$) and the reconstructed G=4 vs G=32 curve

## `platform_hybrid/paper/sections/group_size_reconcile.tex`

Consumers: P03,R02,R03,R04,U01

- L94 [claim] \paragraph{Accuracy-optimal $G$ shifts with budget.}
- L116 [claim] per-token efficiency cost.) The accuracy-optimal $G$ shifts rightward with $T$,
- L121 [claim] always better'' heuristic is false, \emph{and} the opposite heuristic
- L140 [claim] $G{=}8$ $0.990{\pm}.003$, overlapping SEs at $n{=}3$), so the supported claim is
- L148 [claim] that the optimal $G$ is interior rather than smallest-or-largest.

## `platform_hybrid/paper/sections/heldout_stratified.tex`

Consumers: U01

- L56 [claim] \subsection{What the released artifact does support}
- L67 [claim] These numbers support a narrow claim: among already-strong 30-step Tinker
- L69 [claim] low-90s. They do \emph{not} support a claim about random-checkpoint

## `platform_hybrid/paper/sections/intro.tex`

Consumers: U01

- L12 [claim] compare. PPO~\citep{schulman2017proximal}, GRPO~\citep{shao2024deepseekmath},
- L30 [claim] The first contribution of the benchmark is therefore methodological rather than
- L34 [claim] like ``PPO vs.\ GRPO'' are better understood as \emph{stack-level comparisons},
- L46 [claim] causal failure mode beyond simpler observables such as reward mean, entropy,
- L54 [claim] observations support a narrower claim: whether RL fine-tuning works at all can
- L57 [claim] universal optimal group size or a general ``SFT dominates RL'' law.
- L63 [claim] +1.3 percentage-point delta is not statistically significant ($p{=}0.26$).

## `platform_hybrid/paper/sections/length_bias.tex`

Consumers: P04,U01

- L21 [claim] regardless of correctness, the model first learns to associate length
- L46 [claim] first-half / second-half mean difference for both length and
- L155 [claim] move toward the upper-left (shorter, higher-reward), the opposite direction of the verbosity-trap.
- L197 [claim] mean(first half) of \texttt{mean\_comp\_len}; negative means compression. All flag values
- L315 [claim] The trap is said to \emph{onset} at the first $s$ where
- L319 [claim] end-of-window length is greater than the first-half mean length
- L345 [claim] length $>$ first-half mean). The hard task fires more often
- L443 [claim] the window; the first half of training is empty (no local
- L463 [claim] $\rho_\text{rew}\le 0$, end-of-window length $>$ first-half
- L469 [claim] opposite of the verbosity-trap direction. \textbf{(C)}~A
- L496 [claim] paired-bootstrap CI on the algo difference}; the first two are
- L519 [claim] length, ZVF \emph{rises}, the opposite of the simple
- L527 [claim] The sign flip rules out the simple causal arrow
- L567 [claim] mechanism}; they have \emph{opposite} signs of joint coupling
- L605 [claim] $\sim5$ tokens in the first $10$ steps, then plateaus near
- L691 [claim] (Dr.GRPO slope $-0.013$ vs GRPO $-0.008$), opposite to the
- L748 [claim] predictions hold and \emph{two are statistically significant}:
- L777 [claim] easy task. This is the first mechanistic cross-pillar measurement
- L895 [claim] \paragraph{(S4) First-difference coupling $\rho(\Delta L, \Delta R)$.}
- L906 [claim] deviation -- the correction channel is dominant on the hard
- L915 [claim] ($p = 0.035$, the only statistically significant drift);
- L957 [claim] $0.5$, opposite of trap). \textbf{(C)}~Linear length-vs-step slope per
- L959 [claim] horizon. \textbf{(D)}~First-difference coupling $\rho(\Delta L, \Delta R)$:

## `platform_hybrid/paper/sections/length_bias_iter100.tex`

Consumers: U01

- L67 [claim] we drop the first observation, OLS-fit the $2\times 2$ coefficient
- L215 [claim] seeds). Drop-first-observation for the OLS fit; bootstrap $B\!=\!2000$.

## `platform_hybrid/paper/sections/length_bias_iter108.tex`

Consumers: U01

- L20 [claim] fires, but it is silent on two further structural questions. First,
- L22 [claim] from the first window, or does it grow monotonically as the policy
- L50 [claim] Two structurally novel findings emerge, summarised in
- L64 [claim] arithmetic\_easy shows the \emph{opposite monotone}: $\Delta$ progresses
- L138 [claim] doubles between the first and last training windows
- L192 [claim] first two windows and then severs strongly in $w{=}3$ ($\Delta=-0.44$,

## `platform_hybrid/paper/sections/length_bias_iter112.tex`

Consumers: U01

- L18 [claim] *guarantee*: an aggressive severship could in principle be wasted if

## `platform_hybrid/paper/sections/length_bias_iter116.tex`

Consumers: U01

- L76 [claim] opposite of the naive "sever where signal is plentiful" reading.

## `platform_hybrid/paper/sections/length_bias_iter124.tex`

Consumers: U01

- L89 [claim] \emph{null in the opposite sign of H1}. The bootstrap 95\% CI
- L117 [claim] first-difference velocity.
- L147 [claim] structure of the BASELINE GR, not its first-difference velocity.
- L180 [claim] ``innovation coupling'' reading into a specific causal claim.}

## `platform_hybrid/paper/sections/length_bias_iter128.tex`

Consumers: P04

- L24 [claim] $\Delta L = \overline{L}_{\mathrm{first}\,5} - \overline{L}_{\mathrm{last}\,5}$
- L26 [claim] $\Delta R = \overline{R}_{\mathrm{last}\,5} - \overline{R}_{\mathrm{first}\,5}$,

## `platform_hybrid/paper/sections/length_bias_iter132.tex`

Consumers: P04

- L14 [claim] \subsection{Closing the causal chain: per-window $|CCF_{bwd}|$ decoupling
- L38 [claim] \frac{\overline R_{\mathrm{last\,half}\,w} - \overline R_{\mathrm{first\,half}\,0}}
- L39 [claim] {\overline L_{\mathrm{first\,half}\,0} - \overline L_{\mathrm{last\,half}\,w}},
- L42 [claim] (first-half of window 0) to the end of window $w$. Note

## `platform_hybrid/paper/sections/length_bias_iter136.tex`

Consumers: P04

- L27 [claim] Iter~\ref{sec:length-bias-iter132} closed the causal chain:

## `platform_hybrid/paper/sections/length_bias_iter28.tex`

Consumers: P04,U01

- L87 [claim] This is the opposite of the verbosity-trap prediction that the model
- L114 [claim] is therefore not the dominant failure mode on this benchmark;

## `platform_hybrid/paper/sections/length_bias_iter32.tex`

Consumers: P04,U01

- L14 [claim] (length drops first, reward rises $k{>}0$ steps later) the lag profile
- L16 [claim] via an anticipatory channel (reward rises first, length follows at
- L26 [claim] (i) the distribution of the \emph{dominant} lag $k^\star =
- L36 [claim] On 13/16 runs the dominant lag is $k{=}0$ (contemporaneous coupling).
- L41 [claim] support. Critically, the per-cell \emph{paired} bootstrap on
- L97 [claim] causal mechanism), then the verbosity-trap should manifest as a
- L111 [claim] profile per (task, algo); (B) per-seed distribution of the dominant

## `platform_hybrid/paper/sections/length_bias_iter36.tex`

Consumers: P04,U01

- L62 [claim] (length converges to its asymptote \emph{first}, then reward catches
- L76 [claim] in opposite directions, confirming the anti-trap regime at the
- L120 [claim] for GRPO but the wrong model for PPO on the same stack, with AICc
- L122 [claim] 5/5 PPO traces. The present iter extends that finding by asking the
- L125 [claim] On easy arithmetic the joint saturation model is well-supported
- L132 [claim] of $R$ and $L$ move in opposite directions ($\rho \approx -0.6$).
- L133 [claim] The verbosity-trap literature claims the opposite---that hard

## `platform_hybrid/paper/sections/length_bias_iter40.tex`

Consumers: P04,U01

- L16 [claim] anti-trap regime (iter 28's evidence) predicts the opposite: the
- L36 [claim] ($R/L_{\mathrm{last}} - R/L_{\mathrm{first}}$), R/L last, and the
- L56 [claim] prediction is therefore \emph{partially supported} on this task: the
- L85 [claim] The R/L drift (last decile minus first decile of R/L) is $+0.145$ on
- L94 [claim] on the hard task. This is the first metric in the iter 28--40
- L103 [claim] last-decile minus first-decile of $R_t/L_t$; pooled anti-trap slope

## `platform_hybrid/paper/sections/length_bias_iter44.tex`

Consumers: P04,U01

- L40 [claim] mid-reward), producing a NEGATIVE asymmetry. We observe the opposite on every
- L51 [claim] tests support the predicted direction, although the two-sided 95\% CIs

## `platform_hybrid/paper/sections/length_bias_iter48.tex`

Consumers: P04,U01

- L26 [claim] and let $\tplat$ be the first step at which the smoothed $R_t$ crosses

## `platform_hybrid/paper/sections/length_bias_iter52.tex`

Consumers: P04,U01

- L59 [claim] baseline was supposed to preserve but in fact inverts. The GRPO baseline

## `platform_hybrid/paper/sections/length_bias_iter60.tex`

Consumers: P04,U01

- L50 [claim] because it counts steps of opposite sign rather than averaging them.
- L52 [claim] \paragraph{Optimal-length fit on GSM8K CoT: Dr.GRPO's reward-maximising
- L91 [claim] (Dr.GRPO is \emph{tighter}, opposite of the iso-band hypothesis).

## `platform_hybrid/paper/sections/length_bias_iter68.tex`

Consumers: P04,U01

- L68 [claim] \item First-divergence step $\overline{T^*} = 10.3$ (CI $[3, 22]$).
- L85 [claim] this is the \emph{opposite} sign from GSM8K. On the easy task,
- L88 [claim] opposite reward strata, which sharpens the GSM8K finding into a
- L119 [claim] $\mathrm{sign}(\Delta R)$), first-divergence step (threshold

## `platform_hybrid/paper/sections/length_bias_iter72.tex`

Consumers: P04,U01

- L79 [claim] inconclusive, in the opposite sign of GSM8K. The lag-2
- L87 [claim] $1.4\,|\Delta L_t|$ at step $t+1$ in the opposite direction).

## `platform_hybrid/paper/sections/length_bias_iter76.tex`

Consumers: P04,U01

- L18 [claim] the closed-loop work done in $(L, R)$ phase space?
- L45 [claim] \item \textbf{Phase-plane loop area $\mathcal{A}$.} Closed-loop integral

## `platform_hybrid/paper/sections/length_bias_iter80.tex`

Consumers: P04,U01

- L45 [claim] with no attractor---is \textbf{not supported} at this scale and horizon: the
- L69 [claim] finding that Dr.GRPO's reward-optimal length $L^\ast$ is $+81$ tokens larger.

## `platform_hybrid/paper/sections/length_bias_iter84.tex`

Consumers: U01

- L25 [claim] We answer with three complementary spectral / causal measurements applied to
- L42 [claim] \item \textbf{Granger causality} $F$-stat from a VAR(2) on the
- L111 [claim] length-bias mechanism is supposed to bite.
- L121 [claim] --- is therefore \emph{supported by} the Granger signature, even though
- L137 [claim] task --- the GSM8K CoT panel shows two bars moving opposite directions,

## `platform_hybrid/paper/sections/length_bias_iter96.tex`

Consumers: U01

- L45 [claim] dynamics. Crucially, CCF is asymmetric in lag under a genuine causal

## `platform_hybrid/paper/sections/length_bias_react_shaping.tex`

Consumers: P04

- L39 [claim] \textbf{H2}~the first-half vs last-half reward delta is larger
- L55 [claim] half-life delta comes out $-0.044$, i.e.\ sign-opposite to the

## `platform_hybrid/paper/sections/p1_intro.tex`

Consumers: P01

- L6 [claim] the learned value function of PPO~\citep{schulman2017proximal} with a
- L26 [claim] support. Our contributions are: (i) a same-benchmark measurement showing the

## `platform_hybrid/paper/sections/p1_results_intro.tex`

Consumers: P01

- L14 [claim] The subsections below make this precise. We first fit the canonical saturation

## `platform_hybrid/paper/sections/p2_abstract.tex`

Consumers: P02

- L10 [claim] descriptively and, deliberately, decline to promote it to a causal or

## `platform_hybrid/paper/sections/p2_conclusion.tex`

Consumers: P02

- L5 [claim] evidence argues against treating it as a standalone causal or incrementally
- L10 [claim] not contradictory.) Three limitations are load-bearing. First, ZVF is
- L13 [claim] against reward mean, entropy, and divergence proxies are needed before any causal
- L29 [claim] opposite regimes. The constructive path forward is to measure the \emph{magnitude}

## `platform_hybrid/paper/sections/p2_intro.tex`

Consumers: P02

- L19 [claim] descriptive instrument whose reach we bound rather than a causal predictor.
- L39 [claim] held-out improvement, gradient direction, or causal value of an intervention.
- L45 [claim] normalization to preserve signal~\citep{lin2025cppo, gift2025, liu2025rlsubnetworks};

## `platform_hybrid/paper/sections/p2_results_intro.tex`

Consumers: P02

- L10 [claim] has stopped producing within-group signal, yet a low ZVF does not guarantee a
- L14 [claim] The subsections that follow develop this picture. We first formalize ZVF and its
- L23 [claim] its correlations are read as descriptive rather than causal. The final subsection

## `platform_hybrid/paper/sections/p3_abstract.tex`

Consumers: P03

- L9 [claim] anchor the pillar. First, trainability varies with $G$ in a non-monotone way:

## `platform_hybrid/paper/sections/p3_conclusion.tex`

Consumers: P03

- L6 [claim] modest against seed and difficulty variation, so we resist naming an optimal $G$;
- L12 [claim] $\Delta_{\text{GRPO}-\text{PPO}}{=}-0.002$, permutation $p{=}0.62$) while the
- L15 [claim] \texttt{experiments/results/berkeley/unpacking\_dpo\_ppo\_factorization.json}).
- L32 [claim] First, the preference-pair view makes the DPO connection exact rather than
- L38 [claim] optimal group size at every budget ($G^\ast = 8, 16, 32, 32$ at
- L80 [claim] is an all-pairs preference contrast. The clean causal test, and matched-budget

## `platform_hybrid/paper/sections/p3_intro.tex`

Consumers: P03

- L33 [claim] no claim that any single $G$ or algorithm is universally superior.
- L38 [claim] next sweep, but it cannot establish that $G{=}32$ is optimal. The strongest
- L46 [claim] GRPO~\citep{shao2024deepseekmath, ahmadian2024backtobasics, lin2025cppo, gift2025}.

## `platform_hybrid/paper/sections/p3_results_intro.tex`

Consumers: P03

- L6 [claim] The dependence is non-monotone. Intermediate group sizes often outperform the

## `platform_hybrid/paper/sections/p4_abstract.tex`

Consumers: P04

- L20 [claim] Dr.\,GRPO, and specify the length-confounded regime and causal mediation tests that

## `platform_hybrid/paper/sections/p4_conclusion.tex`

Consumers: P04

- L18 [claim] length-mediated success---a limitation the proposed causal mediation protocol is
- L35 [claim] outperform its pre-RL checkpoint on held-out GSM8K; in the Qwen2.5-1.5B
- L36 [claim] comparison, Dr.\,GRPO does not outperform GRPO at our seed budget; and under the
- L41 [claim] regime, a causal length-mediation protocol separating genuine deduction from

## `platform_hybrid/paper/sections/p4_intro.tex`

Consumers: P04

- L25 [claim] pre-RL held-out accuracy is small and not statistically significant (a matched pre-RL
- L32 [claim] and causal length-mediation tests that would reveal a Dr.\,GRPO advantage. We frame
- L33 [claim] the pillar as a boundary on what the current evidence supports, not a refutation of

## `platform_hybrid/paper/sections/p4_results_intro.tex`

Consumers: P04

- L4 [claim] We report three findings and then scope them. First, in our near-ceiling
- L7 [claim] small ($+0.013$) and not statistically significant (paired $p=0.256$, $n{=}5$). We
- L25 [claim] gradient bias rather than universal held-out superiority. We then turn the null into
- L29 [claim] length-spurious / test length-anti-spurious split), a causal length-mediation

## `platform_hybrid/paper/sections/p5_abstract.tex`

Consumers: P05

- L8 [claim] specification. Three exhibits motivate the reporting standard. First, a \emph{nominally} matched-configuration backend
- L19 [claim] asymmetric-clip surrogate produced mean ZVF $0.58$---same name, opposite

## `platform_hybrid/paper/sections/p5_conclusion.tex`

Consumers: P05

- L8 [claim] label produced opposite telemetry signatures. The former is not a backend
- L9 [claim] causal estimate; both are evidence that RL-for-LLM results
- L26 [claim] comparisons support algorithmic claims; R3--R4 comparisons should be
- L32 [claim] field already distinguishes ``PPO'' from ``PPO-clip'' when it matters

## `platform_hybrid/paper/sections/p5_evidence.tex`

Consumers: P05

- L7 [claim] MIN-REPORT-RL. Accordingly, the audits below score the standard's first seven
- L20 [claim] differ. The $17\times$ ratio is therefore \emph{not} a backend-only causal
- L42 [claim] with asymmetric clipping---produced mean ZVF $0.58$. Same label, opposite
- L96 [claim] even define. The same-stack PPO-vs-GRPO isolation runs in the benchmark make
- L214 [claim] \ge 0.20$ (DOMINANT verdict). Stack axes dominate every telemetry channel;
- L349 [claim] \quad $G=32$ & 18 & 56.3 \\
- L419 [claim] The first observation is that \textbf{the algorithm-axis $\eta^{2}$
- L575 [claim] gap the audit has surfaced. KL is the dominant lever in PPO/GRPO
- L1563 [claim] $\geq 0.7310$), and each field's dominant axis is DIFFERENT —
- L1603 [claim] and every field's upper CI bound on the dominant axis is
- L1607 [claim] field has a \emph{different} dominant axis:
- L1722 [claim] stack axis \emph{and} the dominant stack axes
- L1741 [claim] full 5-axis and the 3-dominant-axis subsets.
- L1748 [claim] the dominant axes (H1--H3 PASS).} On \textsc{ZVF} the ratio CI excludes
- L1753 [claim] $[0.37, 40.95]$). Every dominant axis carries the signal.
- L1756 [claim] artifact (H5' PASS).} Drop a dominant axis and another dominant axis
- L1774 [claim] CI $[0.47, 18.69]$); the CI includes $1.0$ on \textsc{ZVF}. The 3-dominant-axis composite gives $\eta^2 = 0.1211$ (\textsc{ZVF}, ratio
- L1778 [claim] unique variance; averaging correlated dominant axes also dilutes (the
- L1779 [claim] three dominant axes share variance through task-stratification).
- L1789 [claim] variance captured by the dominant axes is \emph{conditional} on whether
- L1805 [claim] \multicolumn{6}{l}{\emph{dominant axes --- signal present}} \\
- L1832 [claim] $\eta^2$) --- the dominant axes (task / model / $G$) explain $45$--$47\%$
- L1835 [claim] Round 1} Estimator-Equivalence Principle --- the algorithm-vs-dominant-stack
- L1838 [claim] stack is fixed (or, here, once the dominant stack axes are varied).
- L1841 [claim] ratios as ``$10$--$60\times$ on the DOMINANT stack axes
- L1845 [claim] 3-dominant-axis composite as a CI gate: any new corpus should report

## `platform_hybrid/paper/sections/p5_intro.tex`

Consumers: P05

- L4 [claim] When a paper reports that ``DAPO outperforms GRPO,'' the named algorithm is a
- L31 [claim] it is evidence of \emph{under-specification}, not a backend causal effect; a label

## `platform_hybrid/paper/sections/p5_iter101_zvf130_eta2.tex`

Consumers: P05

- L66 [claim] AREAL $+0.53\%$, CPPO $-0.47\%$, NGRPO $+0.09\%$. SCAFGRPO has the

## `platform_hybrid/paper/sections/p5_iter113_v22_recovery.tex`

Consumers: P05

- L8 [claim] The iter-81 row 96 binomial-null control proved that MIN-REPORT v2.2's
- L154 [claim] signal-bearing on the \emph{SCHEMA} layer; iter-113 proves they

## `platform_hybrid/paper/sections/p5_iter125_chained_eta2.tex`

Consumers: P05

- L22 [claim] For the four dominant pairings (task\_slice, G) on the two shared
- L31 [claim] six non-dominant pairings (model\_family, temperature, seed on zvf
- L33 [claim] are small in absolute terms. The dominant ratios are large: stack
- L59 [claim] \subsubsection{What the chained ratio proves}
- L124 [claim] in this run; the N2 stream had 4000 but we pair the first 2000 to

## `platform_hybrid/paper/sections/p5_iter133_n10_eta2.tex`

Consumers: P05

- L10 [claim] the four dominant (axis, metric) pairings. The remaining open question is
- L38 [claim] seeds ($B=2000$). Channels split into ``band-dominant'' (reward, mean\_len)
- L39 [claim] and ``seed-dominant'' (zvf, loss).}
- L47 [claim] \item \textbf{band-dominant} (reward, mean\_len): $\eta^2_{\mathrm{band}}$
- L50 [claim] \item \textbf{seed-dominant} (zvf, loss): $\eta^2_{\mathrm{seed}}$ exceeds
- L71 [claim] \underline{The 10.32-vs-0.34 inversion is the first measured refusal of

## `platform_hybrid/paper/sections/p5_iter141_algorithm_axis.tex`

Consumers: P05

- L25 [claim] \emph{For verifiable binary-reward LLM RL, once the stack is fixed, PPO and
- L202 [claim] dominant axis. Iter 141 confirms the complement: $\eta^2(\text{method})$
- L213 [claim] empirically supports the (frontier synthesis) claim that ``PPO's value

## `platform_hybrid/paper/sections/p5_iter145_schema_groundtruth.tex`

Consumers: P05

- L171 [claim] confirms that the manifest \emph{corpus} is well-formed enough to support

## `platform_hybrid/paper/sections/p5_iter165_per_step_trajectory.tex`

Consumers: P05

- L21 [claim] \textbf{H2 (FAIL, sharpest negative)}. Trajectory $|\mathrm{Spearman}\,\rho| \le 0.5$ on $\ge 5/6$ channels (trajectory-stationary). \emph{Evidence}: $\rho(\texttt{reward\_mean}){=}{+}0.114$, $\rho(\texttt{mean\_len}){=}{+}\mathbf{0.875}$, $\rho(\texttt{cv\_len}){=}{+}0.401$ --- 

## `platform_hybrid/paper/sections/p5_iter173_headline_cis.tex`

Consumers: P05

- L59 [claim] ``$\eta^2_\text{union} \geq 0.99$'' headline is robustly supported.

## `platform_hybrid/paper/sections/p5_iter177_forward_compat.tex`

Consumers: P05

- L47 [claim] \caption{Per-mutation first-caught audit. v2.4 catches 2/5 mutations

## `platform_hybrid/paper/sections/p5_iter181_v25_rollout.tex`

Consumers: P05

- L99 [claim] (not stack-axis), so they do not by themselves support stack-axis audits.

## `platform_hybrid/paper/sections/p5_iter201_task_stratified_ratio.tex`

Consumers: P05

- L88 [claim] is the dominant driver.

## `platform_hybrid/paper/sections/p5_iter80_delta_div.tex`

Consumers: P05

- L61 [claim] This is the \textbf{first MIN-REPORT item that independently strengthens

## `platform_hybrid/paper/sections/p5_iter81_yield_axes.tex`

Consumers: P05

- L9 [claim] The iter-80 row 95 Item~13 (\texttt{zvf\_yield\_residual}) was the \emph{first}
- L114 [claim] its first per-cell realisation, and iter-81 items 14, 15, 17 add

## `platform_hybrid/paper/sections/p5_iter85_ivison_unpacking.tex`

Consumers: P05

- L11 [claim] four-method same-stack tensors and the Berkeley unpacking\_dpo\_ppo
- L15 [claim] from \texttt{scripts/berkeley/unpacking\_dpo\_ppo\_factorization.py}
- L110 [claim] different data panel. Samestack\_ppo\_grpo reports

## `platform_hybrid/paper/sections/p5_iter89_n2_bootstrap.tex`

Consumers: P05

- L137 [claim] \textbf{first isolation} of GIFT as the lone variance driver on the

## `platform_hybrid/paper/sections/p5_iter93_mega_eta2_bootstrap.tex`

Consumers: P05

- L32 [claim] $[0.830, 0.964]$ --- $G$ is the dominant axis on the easy task;

## `platform_hybrid/paper/sections/p5_limitations.tex`

Consumers: P05

- L5 [claim] five limitations are load-bearing here. First, the headline exhibits

## `platform_hybrid/paper/sections/p5_related.tex`

Consumers: P05

- L55 [claim] ---support our premise that members differ by a small set of loss-form
- L57 [claim] scaling studies of GRPO knobs \citep{tan2025scalingrl} likewise presuppose

## `platform_hybrid/paper/sections/p5_results_intro.tex`

Consumers: P05

- L10 [claim] same-stack PPO-vs-GRPO isolation runs). \textbf{Stratum B} is a set of internal

## `platform_hybrid/paper/sections/p5_stack.tex`

Consumers: P05

- L29 [claim] Same label, opposite ZVF telemetry across loss forms
- L69 [claim] different optimizers with opposite ZVF signatures
- L108 [claim] aliases opposite regimes. Across a 368-run audit (Stratum B; drawn from a
- L219 [claim] --- while the pass@32 frontier tells the opposite, consistent story: all
- L222 [claim] without pass@$k$ curves and stated noise bands, the panel supports any
- L229 [claim] The first seven items above form the seven-field run-manifest fingerprint used in this
- L281 [claim] six candidates, four are GRPO/PPO hyperparameters from the canonical
- L301 [claim] \textit{unpacking\_dpo\_ppo\_factorization} recipe replayed on this
- L310 [claim] document the four GRPO/PPO hyperparameter items as

## `platform_hybrid/paper/sections/p5_threatmodel.tex`

Consumers: P05

- L77 [claim] Two consequences follow. First, most cross-paper comparisons in today's

## `platform_hybrid/paper/sections/p5_toolchain.tex`

Consumers: P05

- L49 [claim] to support L? LeverTrace walks a citation graph and flags credited-but-
- L55 [claim] The manifest's most novel items (4 and 5) are not merely reportable---they
- L62 [claim] guarantee; adaptive-$G$ grpo $+0.575$ / $0.23$ / $186$), a

## `platform_hybrid/paper/sections/p5_worked_example.tex`

Consumers: P05

- L127 [claim] stack is inferior---only that the pair cannot support a label-level

## `platform_hybrid/paper/sections/p6_conclusion.tex`

Consumers: P06

- L4 [claim] Four limitations bound what this resource can claim. First, \emph{seed
- L29 [claim] opposite telemetry and a $17\times$ same-label outcome swing.

## `platform_hybrid/paper/sections/p6_controller_pipeline.tex`

Consumers: P06

- L55 [claim] ``\emph{on which methods does the optimal $\tau$ vary by $>0.02$?}''
- L58 [claim] is the first operational decomposition of \emph{how a registry

## `platform_hybrid/paper/sections/p6_intro.tex`

Consumers: P06

- L11 [claim] \citep{lin2025cppo, nan2025ngrpo, zhang2025scaffgrpo, mcgrpo2025, gift2025,
- L24 [claim] swing is an under-specification exhibit, not a backend-only causal effect.
- L28 [claim] user-settable, produced mean ZVF $0.58$. Same label, opposite telemetry. When

## `platform_hybrid/paper/sections/p6_iter106_claim_evidence_ledger.tex`

Consumers: P06

- L36 [claim] \texttt{SUPPORTS} if observed sign $\in$ predicted-sign-set;
- L63 [claim] \texttt{delta\_dapo}, \texttt{delta\_gspo}, \texttt{delta\_ppo} each declare
- L70 [claim] \paragraph{H3 -- Verdict distribution is sharply concentrated on SUPPORTS.}
- L72 [claim] SUPPORTS=10, NEUTRAL=6, CONTRADICTS=3, UNCLAIMED=8 (post iter-106). The 9
- L74 [claim] claims (delta\_dapo, delta\_gspo, delta\_ppo). Stripping those leaves a
- L76 [claim] The SUPPORTS ratio = 10/27 = 0.37: the registry's human-supplied
- L80 [claim] -- both variants \emph{reduce} reward\_mean vs GRPO (significant, opposite to
- L83 [claim] (significant, opposite to the predicted \texttt{<0}).
- L123 [claim] into audit-grade vs ungrounded (10 SUPPORTS+ 6 NEUTRAL + 3 CONTRADICTS =
- L132 [claim] (delta\_dapo, delta\_gspo, delta\_ppo) require either a same-stack arm

## `platform_hybrid/paper/sections/p6_iter118_claim_xref_coverage_strict.tex`

Consumers: P06

- L24 [claim] cppo, mcgrpo, es, scafgrpo); (c) a CI-style strict-mode validator that
- L33 [claim] \textbf{SUPPORTS} (CI excludes 0 and sign matches predicted),
- L44 [claim] \textsc{Supports}=10, \textsc{Contradicts}=3, \textsc{Neutral}=3,
- L52 [claim] (3 tuples), \texttt{delta\_gspo} (3 tuples), and \texttt{delta\_ppo}
- L55 [claim] delta\_gift, delta\_adaptiveg, delta\_cppo, delta\_drgrpo, delta\_es,
- L76 [claim] cppo, mcgrpo, es, scafgrpo) carry only their zvf130\_* stub entry -- no

## `platform_hybrid/paper/sections/p6_iter122_validate_strict_crossentry.tex`

Consumers: P06

- L32 [claim] (\texttt{delta\_cppo}, \texttt{delta\_es}, \texttt{delta\_mcgrpo},
- L73 [placeholder] \paragraph{H4 -- 3 zvf130\_* stub entries use a placeholder component.}
- L77 [placeholder] placeholder because the corresponding delta\_*.json record's
- L80 [claim] (cppo, es, mcgrpo, ngrpo, scafgrpo, grpo) reference real components.
- L81 [placeholder] The 3 placeholder rows are honest reporting: the registry explicitly
- L85 [placeholder] the schema to permit the placeholder string under a documented
- L86 [placeholder] \texttt{placeholder\_allowed: true} flag. Neither is required for the
- L114 [placeholder] placeholder component). (iv) \textbf{P5P8-SYNTH iter-120 row 135}
- L123 [placeholder] (b) Close the 3 placeholder-component rows in zvf130\_aero, zvf130\_areal,
- L126 [placeholder] by adding a documented \texttt{placeholder\_allowed: true} flag to the

## `platform_hybrid/paper/sections/p6_iter126_measured_evidence_tier.tex`

Consumers: P06

- L54 [claim] 3/6 statistically significant at the 95\% paired-step bootstrap level
- L56 [claim] \item \textbf{Tier B (7 deltas)} -- \texttt{cppo}, \texttt{drgrpo}, \texttt{es},
- L62 [claim] \item \textbf{Tier D (5 deltas)} -- \texttt{dapo}, \texttt{gspo}, \texttt{liteppo},
- L63 [claim] \texttt{ppo}, \texttt{reinforce}. These entries describe the variant and
- L74 [claim] evidentiary backing; a tier-D delta claim (dapo/gspo/liteppo/ppo/reinforce)
- L113 [claim] 4 & delta\_cppo & grpo & 1/3 & 1 & B & --- \\
- L121 [claim] 11--15 & delta\_dapo / gspo / liteppo / ppo / reinforce & grpo & 0/0 & 0 & D & --- \\

## `platform_hybrid/paper/sections/p6_iter134_field_completeness.tex`

Consumers: P06

- L100 [claim] $\texttt{delta\_cppo}$ & \texttt{mean\_zvf} & \texttt{zvf130\_5seed} & 1 \\
- L123 [claim] \texttt{delta\_liteppo} (liteppo2024, no peer-reviewed citation),
- L124 [claim] \texttt{delta\_ppo} (schulman2017ppo, arXiv 1707.06347), and
- L127 [claim] \texttt{delta\_ppo}) declare \texttt{expected\_effects} and need only a
- L129 [claim] (\texttt{delta\_liteppo}, \texttt{delta\_reinforce}) lack both
- L146 [claim] \texttt{bootstrap\_paired\_5seed} & 13 & aero, areal, cppo, es, gift, \\
- L148 [claim] \texttt{normal\_approx\_welch} & 8 & aero, areal, cppo, es, gift, \\
- L188 [claim] $\texttt{delta\_ppo}$ (value\_head) is in the empty-measured list; the
- L190 [claim] estimator under sparse terminal reward. A measured same-stack PPO arm

## `platform_hybrid/paper/sections/p6_iter142_verdict_aggregate.tex`

Consumers: P06

- L47 [claim] \texttt{expected\_effect} for that metric), \texttt{SUPPORTS}=10 (26.3\%),
- L49 [claim] \textsc{unclaimed} mode is dominant because tier-A deltas carry forward
- L54 [claim] \textit{Sufficient-to-supports rate} (SUPPORTS / tier\_total\_$n$, excluding
- L56 [claim] generates $4/(4+3+2) = 4/9 = 44.4\%$ SUPPORTS; tier~B generates
- L57 [claim] $6/(6+3+1) = 6/10 = 60.0\%$ SUPPORTS.
- L58 [claim] Counterintuitively, tier~B has a higher SUPPORTS rate than tier~A. The
- L64 [claim] SUPPORT). Tier-B entries declare narrower expectations (mostly
- L66 [claim] SUPPORTS. The cross-tab is therefore a measure of \emph{claim scope vs.\
- L70 [claim] \textit{Per-metric SUPPORTS rate}. Three of the nine registered metrics
- L76 [claim] \item \textbf{zvf\_risk\_mean}: 8/8 SUPPORTS (100\%) -- the only
- L77 [claim] metric where every declared expectation is supported by a measured row.
- L80 [claim] negative, the row-level SUPPORTS rate is at the ceiling.
- L81 [claim] \item \textbf{zvf}: 2/4 SUPPORTS (50.0\%) -- mixed: \texttt{gift}'s
- L82 [claim] \texttt{>0} prediction SUPPORTS; \texttt{aero}'s \texttt{<0} NEUTRAL (CI
- L84 [claim] \texttt{<0} SUPPORTS.
- L85 [claim] \item \textbf{reward\_mean}: 0/4 SUPPORTS -- 2 NEUTRAL + 2 CONTRADICTS.
- L95 [claim] \textit{Per-panel SUPPORTS rate} reveals the structural asymmetry that
- L98 [claim] \item \texttt{n2\_same\_stack\_last10} panel: 1/6 SUPPORTS = 16.7\%
- L99 [claim] (the single SUPPORTS is \texttt{gift}$\cdot$\texttt{zvf} at $\delta=0.125$
- L101 [claim] \item \texttt{zvf130\_5seed} panel: 8/8 SUPPORTS = 100\%.
- L111 [claim] SUPPORTS rate is consistent with this: most variant-on-metric claims
- L113 [claim] panel's 100\% SUPPORTS rate is consistent with iter-126's tier-A
- L134 [claim] expectations exist: \texttt{gift} 3/3 = 100\%, \texttt{cppo}/\texttt{es}/
- L148 [claim] \item \textbf{The high-level SUPPORTS rate is NOT a registry quality
- L152 [claim] \item \textbf{Tier does NOT predict SUPPORTS at the cell level.} Tier
- L153 [claim] ranks evidence depth; the SUPPORTS verdict ranks agreement between
- L167 [claim] per-cell level: \texttt{n2\_same\_stack\_last10} SUPPORTS rate is 16.7\%,
- L171 [claim] \emph{within-tier} SUPPORTS rate is NOT a monotone function of tier.
- L190 [claim] A & SUPPORTS & 4 & 22.22 & 18 \\
- L195 [claim] B & SUPPORTS & 6 & 30.00 & 20 \\
- L208 [claim] \caption{Iter-142 P6 panel $\times$ evaluated-cell SUPPORTS rate (UNCLAIMED
- L216 [claim] panel & $n_{\text{supports}}$ / $n_{\text{eval}}$ & rate (\%) & bootstrap anchor \\

## `platform_hybrid/paper/sections/p6_iter150_n2_recompute_prose_vs_measured.tex`

Consumers: P06

- L122 [claim] = -0.025$ (NS, but trending the opposite way). The registry already
- L137 [claim] predict a measurable ZVF direction actually shift ZVF in the opposite
- L145 [claim] 11 prose components (dapo/gspo/mcgrpo/scafgrpo/ppo/reinforce/liteppo)
- L151 [claim] same-stack number or the citation in which the claimed effect was first

## `platform_hybrid/paper/sections/p6_iter190_raw_recompute_audit.tex`

Consumers: P06

- L35 [claim] \item \textbf{9 claims} are \textsc{Supports} (significant; direction
- L37 [claim] \item \textbf{3 claims} are \textsc{Supports-NS} (direction matches;
- L48 [claim] (\texttt{adaptiveg, drgrpo, dapo, gspo, ppo, ppo\_reinforce}; $14$
- L60 [claim] Support rate on the measurable subset: $12/14 = 85.7\%$. Contradiction
- L77 [claim] (\verb|cppo|, \verb|es|, \verb|mcgrpo|, \verb|ngrpo|, \verb|scafgrpo|)
- L101 [claim] cppo & zvf\_risk & zvf130 & $-0.151$ & $-0.253$ & $-0.049$ & yes \\
- L120 [claim] measurable expected effects today (dapo, gspo, ppo, ppo\_reinforce,
- L131 [claim] aero & zvf & n2\_last10 & $<$0 & $-0.025$ & Supports-NS \\
- L133 [claim] aero & zvf\_risk & zvf130 & $<$0 & $-0.148$ & Supports \\
- L135 [claim] gift & zvf & n2\_last10 & $>$0 & $+0.125$ & Supports \\
- L136 [claim] gift & reward\_mean & n2\_last10 & $\ge$0 & $+0.016$ & Supports-NS \\
- L137 [claim] gift & zvf\_risk & zvf130 & $<$0 & $-0.263$ & Supports \\
- L139 [claim] areal & zvf & n2\_last10 & $<$0 & $-0.056$ & Supports-NS \\
- L141 [claim] areal & zvf\_risk & zvf130 & $<$0 & $-0.246$ & Supports \\
- L143 [claim] cppo & zvf\_risk & zvf130 & $<$0 & $-0.151$ &Supports \\
- L144 [claim] es & zvf\_risk & zvf130 & $<$0 & $-0.273$ & Supports \\
- L145 [claim] mcgrpo& zvf\_risk & zvf130 & $<$0 & $-0.174$ & Supports \\
- L146 [claim] ngrpo & zvf\_risk & zvf130 & $<$0 & $-0.131$ & Supports \\
- L147 [claim] scafgrpo& zvf\_risk & zvf130 & $<$0 & $-0.352$ & Supports \\
- L150 [claim] \caption{Verdicts on measurable claims. Supports: sign matches AND
- L151 [claim] CI excludes 0. Supports-NS: sign matches AND CI contains 0.
- L153 [claim] significant in the opposite direction).}
- L169 [claim] \item \texttt{ppo}, \texttt{ppo\_reinforce}: $6$ expected, no
- L170 [claim] same-stack PPO run exists.
- L179 [claim] for DAPO / GSPO / PPO exists, those $9$ claims cannot be falsified by
- L197 [claim] \emph{ranking}: GIFT moves from \textit{``mostly supported''} to
- L198 [claim] \textit{``only variant with all 3 measurable claims supported''}; AERO

## `platform_hybrid/paper/sections/p6_iter74_drgrpo.tex`

Consumers: P06

- L8 [placeholder] placeholders: \texttt{delta\_dapo}, \texttt{delta\_drgrpo},
- L9 [claim] \texttt{delta\_gspo}, \texttt{delta\_liteppo},
- L29 [claim] steps than GRPO on this corpus, OPPOSITE the registry-listed
- L31 [claim] the registry's \textbf{first CONTRADICTS verdict on a
- L41 [claim] 0.47/0.47/0.55) --- no measured support for the registry-claimed delta
- L43 [claim] recording in the registry itself: the \emph{first} version of this panel
- L60 [claim] \emph{Bold verdict} marks the registry's first CONTRADICTS on a
- L84 [placeholder] placeholders. Each is characterised by (a) the core claim, (b) the
- L103 [claim] \texttt{delta\_liteppo} & 4 & remove value head + clip advantages & N2-same-stack 5-method (LitePPO replacing GRPO) & no tinker LitePPO rollout log \\
- L109 [claim] The structural pattern is uniform: all four entries are PPO-family
- L123 [claim] \textbf{10 SUPPORTS, 3 CONTRADICTS, 6 NEUTRAL, 8 UNCLAIMED}
- L129 [claim] \emph{opposite} the registry-listed prediction. Schema validation
- L142 [claim] The 4 remaining unmeasured deltas (DAPO/GSPO/LitePPO/REINFORCE)
- L154 [claim] (4 rows: dapo/gspo/liteppo/reinforce characterisation);

## `platform_hybrid/paper/sections/p6_iter78_field_coverage.tex`

Consumers: P06

- L2 [claim] \subsection{Iter-78: Registry Field-Level Coverage Audit + PPO Backfill}
- L13 [claim] punctuation normalization on method labels so \texttt{PPO} $=$
- L14 [claim] \texttt{ppo} $=$ \texttt{pporeinforce}) and pairs it with Vein (d): a
- L15 [claim] new transparent entry for the previously-unregistered PPO method, plus
- L21 [claim] 12 stack labels (aero/areal/cppo/dapo/drgrpo/es/gift/grpo/gspo/mcgrpo/
- L24 [claim] DAPO, Dr.GRPO, GSPO, LitePPO). Adding \texttt{delta\_ppo.json} lifts
- L30 [claim] (DAPO, GSPO, LitePPO, REINFORCE, PPO) all fail the same-stack-arm
- L34 [claim] Tinker panel (DAPO/GSPO/LitePPO/REINFORCE/PPO with only the named
- L36 [claim] narrower: 3 of 5 zero-measured variants (DAPO, GSPO, PPO) now carry
- L45 [claim] (LitePPO and REINFORCE) carry all 5 audited blocks null. The 7
- L46 [claim] partially-populated entries (Adaptive-G, CPPO, DrGRPO, ES, DAPO,
- L47 [claim] GSPO, PPO) carry 1 populated block each. The 6 fully-populated
- L59 [claim] \textbf{only} ledger entry that was a genuine oversight (PPO, 1
- L70 [claim] to \texttt{SUPPORTS}/\texttt{NEUTRAL}/\texttt{CONTRADICTS}.
- L78 [claim] axes) and the iter-78 PPO backfill share the same insight: when a

## `platform_hybrid/paper/sections/p6_iter82_window_sensitivity.tex`

Consumers: P06

- L76 [claim] iter-78 PPO backfill left the existing aero/gift/areal
- L78 [claim] \textbf{first CI-overlap audit} of those blocks against an

## `platform_hybrid/paper/sections/p6_iter86_cross_stack_matrix.tex`

Consumers: P06

- L122 [claim] matrix is the first analysis that uses every existing

## `platform_hybrid/paper/sections/p6_iter90_zvf130_measured_audit.tex`

Consumers: P06

- L18 [claim] \texttt{zvf130\_$<\mathrm{method}>$.json} entries (CPPO, ES, MCGRPO, NGRPO,
- L35 [claim] \texttt{registry/entries/zvf130\_\{cppo,es,mcgrpo,ngrpo,scafgrpo\}.json}.
- L47 [claim] \item CPPO $\Delta{=}{-}0.151$ $[-0.257, {-}0.050]$ --- continuity penalty
- L63 [claim] AERO$\leftrightarrow$CPPO, AERO$\leftrightarrow$MCGRPO,
- L64 [claim] AERO$\leftrightarrow$NGRPO, AREAL$\leftrightarrow$CPPO,
- L67 [claim] CPPO$\leftrightarrow$NGRPO. The full matrix is

## `platform_hybrid/paper/sections/p6_iter94_schema_validator.tex`

Consumers: P06

- L44 [placeholder] arxiv missing and notes did not flag as transparent-placeholder).
- L61 [claim] 5 \texttt{intentional\_null} \texttt{variant\_delta} entries (delta\_ppo,
- L62 [claim] delta\_reinforce, delta\_liteppo, delta\_dapo, delta\_gspo) all with real
- L131 [claim] to \texttt{delta\_ppo}.
- L134 [claim] The validator is now the canonical first-line audit of the registry: every
- L153 [placeholder] transparent-placeholder markers)

## `platform_hybrid/paper/sections/p6_measured_claimed.tex`

Consumers: P06

- L12 [claim] decreases statistically significant). That finding isolates the
- L37 [claim] \textbf{SUPPORT} (sign matches, CI excludes 0), \textbf{WEAK} (sign
- L38 [claim] matches, CI contains 0), \textbf{OPPOSE} (sign contradicts), or
- L66 [claim] delta\_cppo & cppo & NO\_DATA & 0/5 & 0.295 & $-0.186$ & 1 & unknown \\
- L95 [claim] the canonical prompt distribution says the opposite.
- L107 [claim] \paragraph{Finding C: $1/11$ variants shows an OPPOSE verdict.}
- L112 [claim] The N2 single-step measurement points weakly opposite; the
- L114 [claim] is the only OPPOSE verdict the reconciliation produces, and the
- L125 [claim] is not supported on the N2 same-stack data for any of the 4

## `platform_hybrid/paper/sections/p6_measured_coverage.tex`

Consumers: P06

- L33 [placeholder] placeholders: \texttt{delta\_dapo}, \texttt{delta\_drgrpo},
- L36 [claim] \texttt{delta\_reinforce} and \texttt{delta\_liteppo} (the iter-54
- L38 [claim] tally is \textbf{10 SUPPORTS, 2 CONTRADICTS, 4 NEUTRAL, 8 UNCLAIMED}
- L68 [claim] cppo & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 \\
- L74 [claim] liteppo & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
- L97 [claim] $\texttt{delta\_liteppo}$)---the iter-54 additions deliberately
- L98 [placeholder] shipped with \texttt{measured}=[] as a structural placeholder, with
- L106 [claim] First, the audit is a one-shot script ($\sim$280\,LoC, stdlib only)
- L118 [claim] is the first \emph{positive finding} this iteration surfaces: a
- L129 [claim] SUPPORTS, and that verdict survives the cross-panel audit at the

## `platform_hybrid/paper/sections/p6_measured_evidence.tex`

Consumers: P06

- L33 [claim] statistically significant reward \emph{decreases}. The largest effects
- L68 [claim] (AERO $-0.014$ and AREAL $-0.020$) are statistically significant on the

## `platform_hybrid/paper/sections/p6_missing_deltas.tex`

Consumers: P06

- L15 [placeholder] iter, two of them as honest provenance-only placeholders and the
- L38 [claim] \item $\texttt{delta\_liteppo}$ --- LitePPO (no value head, symmetric
- L40 [claim] $\texttt{ppo\_lite}$ in the ledger but had no entry. Citation is a
- L41 [placeholder] transparent placeholder ($\texttt{arxiv}=\text{``''}$) because no
- L42 [claim] peer-reviewed LitePPO paper is verified; this is flagged on the
- L71 [claim] $-0.0312$] (SUPPORTS predicted $<0$, CI excludes 0). This is the
- L72 [claim] first machine-readable evidence that the ZVF-driven adaptive
- L83 [claim] \textbf{LitePPO}, \textbf{REINFORCE} --- the two provenance-only
- L93 [placeholder] are honest provenance-only placeholders with $\texttt{measured}=[]$
- L103 [claim] delta\_liteppo & 2 & grpo & (provenance only) & 0 & 0 \\
- L112 [claim] $\texttt{delta\_liteppo}$) are the first entries in the registry with
- L118 [placeholder] ``backlog-with-provenance'' pattern: structural placeholders that
- L122 [claim] The $\Delta_{\text{zvf}}$ SUPPORTS verdict is the \emph{third}
- L126 [claim] is now a closed loop for the adaptive-G delta. The
- L131 [claim] SUPPORTS or CONTRADICTS --- the iter-31 prediction remains

## `platform_hybrid/paper/sections/p6_population.tex`

Consumers: P06

- L50 [claim] zvf130_cppo & cppo & open & --- & 43 & $\checkmark$ \\
- L67 [placeholder] rows of \texttt{framework_comparison.json} are dry-run placeholders (their
- L93 [claim] delta against GRPO is guaranteed to isolate the label, not the stack. Measured
- L96 [claim] machine-queryable for the first time.
- L99 [claim] Five entries (ngrpo, cppo, mcgrpo, es, scafgrpo) record a different shape of

## `platform_hybrid/paper/sections/p6_query_auditor.tex`

Consumers: P06

- L42 [claim] First, items are weighted equally: a stack cannot buy a high badge by

## `platform_hybrid/paper/sections/p6_registry_health.tex`

Consumers: P06

- L31 [claim] records that reference it. ``Implemented'' means the stack proves the
- L50 [claim] delta\_cppo & 1 & 0 & 0 & 0 & 1 & risk-index only \\
- L123 [claim] (20\%) the dominant reporting gaps across both openness tiers; the
- L126 [claim] improve their badge should target \texttt{loss\_form} first
- L127 [claim] (the largest gap on managed stacks) and \texttt{decontamination} first
- L277 [claim] frontier for P6, and it is the first quantitative answer to the
- L286 [claim] other two. The iter 30 result is the first empirical evidence that
- L310 [claim] 100\% pass}. This is the first subcommand with a numeric exit code
- L313 [claim] first-class subcommand.
- L337 [claim] \texttt{framework $\times$ \{aero,areal,cppo,es,gift,gspo,mcgrpo,ngrpo,
- L338 [claim] scafgrpo\}} cells where only the canonical GRPO/PPO arm is populated.
- L356 [claim] \item \texttt{13255b77ea} (CPPO / ES / MCGRPO / NGRPO / SCAFGRPO) ---
- L400 [claim] delts in null cluster & 5/11 (45\%) & cppo / es / mcgrpo / ngrpo / scafgrpo \\
- L407 [claim] First, the audit script is CI-ready: \texttt{python3 registry/query.py
- L535 [claim] ``schema-bump the \texttt{decontamination} block first''
- L622 [claim] zvf130\_cppo & 0.429 & 0.000 & 0.000 & False \\
- L731 [claim] side: it is the first block that grounds ZVF in a directional, signed

## `platform_hybrid/paper/sections/p6_schema_extension.tex`

Consumers: P06

- L163 [claim] cppo & --- & --- & $-0.151^{\dagger}$ \\
- L213 [claim] (\texttt{cppo}, \texttt{es}, \texttt{mcgrpo}, \texttt{ngrpo},
- L261 [claim] delta\_cppo & 19 & 0 & 0 & 0 & 1 \\
- L326 [placeholder] components; the other 4 are \texttt{see delta-list and citation} placeholders).

## `platform_hybrid/paper/sections/p6_worked_examples.tex`

Consumers: P06

- L64 [claim] reported negative is a first-class answer---while all five managed-runtime

## `platform_hybrid/paper/sections/p7_abstract.tex`

Consumers: P07

- L11 [claim] that value rests on the weakest evidence (the causal gradient link is toy-scale
- L43 [claim] has not been measured. No result here establishes prospective superiority over

## `platform_hybrid/paper/sections/p7_conclusion.tex`

Consumers: P07

- L19 [claim] superiority. Adaptive $G$ reached the same held-out delta as Dr.~GRPO while
- L23 [claim] in for closed-loop learning. The durable output is therefore a falsifiable design

## `platform_hybrid/paper/sections/p7_controller.tex`

Consumers: P07

- L24 [claim] non-degenerate. Dynamic sampling is guaranteed to reach ZVF $= 0$; the
- L64 [claim] Three readings, in decreasing order of confidence. \emph{First}, the
- L77 [claim] the controller is \emph{competitive with, not superior to}, the best fixed
- L231 [claim] $[-0.314, 0.578]$, and the first-five-step ZVF does not
- L232 [claim] ($r = 0.184$, CI $[-0.815, 0.162]$). This is the first seed-level
- L240 [claim] reward, but the within-seed ZVF trajectory is the dominant leading
- L456 [claim] \subsection{Per-prompt hindsight-optimal $G'$ analysis}
- L457 [claim] \label{sec:p7-per-prompt-optimal-g}
- L489 [claim] under the binomial model when $p \in \{1/8, 7/8\}$); the cost-optimal
- L498 [claim] prompt-step, so the optimal controller \emph{never} improves the
- L523 [claim] \caption{Per-prompt hindsight-optimal $G^*$ analysis on the N2
- L531 [claim] \label{tab:p7-per-prompt-optimal}
- L608 [claim] hypothesis makes and the calibrated Pareto empirically supports
- L616 [claim] Table~\ref{tab:p7-per-prompt-optimal}'s caption calls the result the
- L975 [claim] $\tau_{\text{post}}{=}0.60$ — Pareto-dominant in iter 11 — is
- L1031 [claim] empirical ordering is supported by the data, not an artefact of
- L1152 [claim] escalation; zvf-triage remains Pareto-optimal only on the
- L1171 [claim] the saturated regime. The two rules have **opposite signs** on the
- L1289 [claim] prediction is testable for the first time.
- L1316 [claim] (528$\to$432)}, the \emph{first empirical confirmation} of the iter~27
- L1330 [claim] seed $20260704$). This is the **first falsifiable evidence** that the
- L1331 [claim] unified Hybrid is Pareto-superior to zvf-triage on a real evidence
- L1375 [claim] The previous subsection proved C3$\equiv$C1 bit-for-bit on the N10 panel
- L1399 [claim] Pareto-superior to zvf-triage on saturation-band panels (\texttt{gift}:
- L1444 [claim] At $\tau{=}0.50$, the savings signs are \emph{opposite}:
- L1466 [claim] opposite-sign savings with non-overlapping 95\% CIs. This is the
- L1538 [claim] on the savings axis at every $\tau$ where either fires (opposite signs,
- L1703 [claim] \emph{First}, \textbf{zvf-triage recovers 64\%--73\% of the iid
- L2158 [claim] Pareto frontier -- they live on opposite sides of the $G{=}8$ line.
- L2193 [claim] establishes $\gamma^*{=}0$ as the optimal tree-baseline smoothing --
- L2303 [claim] The joint controller is the \emph{first end-to-end unification} of

## `platform_hybrid/paper/sections/p7_design_rules.tex`

Consumers: P07

- L4 [claim] The theory and the E3 audit jointly support a small set of design rules for
- L5 [claim] group-size control. We state them as rules with their supporting evidence,
- L74 [claim] static trade is exactly what Rules~1--3 convert into a closed-loop

## `platform_hybrid/paper/sections/p7_e1_validation.tex`

Consumers: P07

- L31 [claim] variants, and no evidence that the coefficient (as opposed to the sign)

## `platform_hybrid/paper/sections/p7_intro.tex`

Consumers: P07

- L12 [claim] to promote it to a predictive or causal statistic. A companion
- L20 [claim] This paper takes the next two steps. First, \emph{prediction}: we show that a

## `platform_hybrid/paper/sections/p7_iter103_unified_controller.tex`

Consumers: P07

- L19 [claim] $\gamma^*{=}0$ is optimal (DECISIVE on $12/12$ cells of iter127).

## `platform_hybrid/paper/sections/p7_iter107_tautransfer.tex`

Consumers: P07

- L115 [claim] step-cells are missed-escalation opportunities (C\_FN\_DRIFT); the
- L159 [claim] tension; they reflect two Pareto-optimal operating points along the

## `platform_hybrid/paper/sections/p7_iter115_adaptive_gstar_n10_multiseed.tex`

Consumers: P07

- L21 [claim] optimal $G^*$, 4.1$\times$ more negative than fixed-G controllers).
- L58 [claim] bisection (symmetry guarantees uniqueness on $[0,0.5]$). The
- L63 [claim] ($G=16$ on trigger); \texttt{ADAPTIVE\_GSTAR} (closed-form optimal

## `platform_hybrid/paper/sections/p7_iter119_calibrated_controller_unification.tex`

Consumers: P07

- L50 [claim] \max\!\bigl(G_{\mathrm{base}}, \min(G_{\mathrm{adaptive-g^*}}, G=32)\bigr) & \text{if } z_{\mathrm{obs}} \geq 0.70 \quad \text{(DEGENERATE)}
- L53 [claim] The $G=32$ cap in DEGENERATE encodes the iter-115 finding that the
- L99 [claim] \rightarrow G=16$; else $\rightarrow G=32$. On iter127 (5$\times$4 cell
- L115 [claim] capped at $G=32$ per the iter-115 finding.
- L144 [claim] $[0, 0.5]$; symmetry guarantees uniqueness.

## `platform_hybrid/paper/sections/p7_iter147_unified_per_prompt.tex`

Consumers: P07

- L37 [claim] $G \in \{8, 16, 32\}$ via closed-form Bernoulli inversion capped at $G=32$;

## `platform_hybrid/paper/sections/p7_iter159_pareto_permethod_ci.tex`

Consumers: P07

- L13 [claim] frontier, did not classify controllers as dominated/Pareto-optimal/strictly
- L14 [claim] optimal, and did not compute cross-method SDs with bootstrap CIs to
- L133 [claim] $4 \times 5 = 20$ (method, controller) points. A point is Pareto-optimal if
- L150 [claim] \emph{never Pareto-optimal} in N2. This contradicts the paper's framing of
- L155 [claim] \textbf{The second Pareto finding}: UNIFIED\_C4 is Pareto-optimal only on

## `platform_hybrid/paper/sections/p7_iter167_oracle_regret.tex`

Consumers: P07

- L12 [claim] Iter-167 closes this gap by computing the oracle's optimal $G^\star$ for
- L80 [claim] on the dominant easy prompt class ($k{=}7$, $\hat p = 0.875$, frequency
- L94 [claim] relative to the marginal-optimal oracle: it picks a single $G'$ that

## `platform_hybrid/paper/sections/p7_iter175_calibrated_hybrid.tex`

Consumers: P07

- L16 [claim] cell means the no-blend regime IS the empirically supported "do-nothing"
- L44 [claim] The first branch (both signals agree) is the calibrated-hybrid core;
- L80 [claim] method (4/4); it is co-Pareto-optimal with $C_0$ and $C_5$ (low-cost
- L122 [claim] \caption{$C_6$ Pareto-optimal count per $(\tau_z, \mathrm{method})$.
- L124 [claim] $C_6$ Pareto-fails on the dominant easy-prompt fraction or the dominant
- L155 [claim] dominant-class prompts ($p_{\mathrm{hat}} > 0.85$, 8.3\% of

## `platform_hybrid/paper/sections/p7_iter192_perfire_optimal_gn.tex`

Consumers: P07

- L30 [claim] $\mathrm{eff}=0$) guarantees $G_N^*(p) = G_{\mathrm{base}}{=}8$ on
- L76 [claim] $G_N \in \mathcal{G}$ is therefore optimal on every contrast prompt.
- L105 [claim] per-prompt cost-effective optimum. H3 (``per-prompt optimal restored
- L130 [claim] DECREASING eff in $G_N$ is equivalent to ``first G in the grid is
- L131 [claim] optimal on every contrast prompt''}.
- L215 [claim] equivalent to ``smallest G in the grid is optimal on every contrast

## `platform_hybrid/paper/sections/p7_iter199_closed_loop_counterfactual.tex`

Consumers: P07

- L2 [claim] \label{sec:p7-closed-loop-counterfactual}
- L8 [claim] closed-loop training experiment. We ask what the diagnostic trajectory of
- L137 [claim] The dominant operating point depends on the practitioner's tolerance for

## `platform_hybrid/paper/sections/p7_iter203_empirical_gprime.tex`

Consumers: P07

- L27 [claim] \paragraph{What the result supports.}
- L33 [claim] \paragraph{What the result does not support.}

## `platform_hybrid/paper/sections/p7_iter79_multitrigger.tex`

Consumers: P07

- L116 [claim] \emph{first quantitative isolation} of the anti-herding failure mode
- L175 [claim] first MIN-REPORT-compatible CIs for the C3 sub-panel. (iii) Iter 71

## `platform_hybrid/paper/sections/p7_iter83_iso_g.tex`

Consumers: P07

- L115 [claim] proved necessary: the 0.21-vs-0.562 saving discrepancy was a

## `platform_hybrid/paper/sections/p7_iter87_hysteresis.tex`

Consumers: P07

- L57 [claim] optimization horizon it is supposed to react to.

## `platform_hybrid/paper/sections/p7_iter88_hysteresis_n10.tex`

Consumers: P07

- L48 [claim] cells are Pareto-dominant: flip-ratio CI excludes 1.0 and yield-retention

## `platform_hybrid/paper/sections/p7_iter91_perfire_gain.tex`

Consumers: P07

- L120 [claim] universally dominant --- a falsifiable, scope-conditional
- L158 [claim] ($\overline{\Delta_z} = 0.061$ on the Pareto-dominant cell). The
- L169 [claim] iter-91's $\overline{\Delta_z}$ is the \emph{first moment} of

## `platform_hybrid/paper/sections/p7_iter92_asymmetric_hysteresis.tex`

Consumers: P07

- L77 [claim] \paragraph{Headline H2 --- the gain is Pareto-dominant on the
- L105 [claim] bound $\geq 1.00$, certifying the gain as Pareto-dominant at the
- L138 [claim] Pareto-dominant refinement of all three prior iters.

## `platform_hybrid/paper/sections/p7_limitations.tex`

Consumers: P07

- L29 [claim] superiority.
- L38 [claim] \emph{Toy-scale gradient evidence.} T3's empirical support is the
- L53 [placeholder] the framework comparison, which remain dry-run placeholders

## `platform_hybrid/paper/sections/p7_results_intro.tex`

Consumers: P07

- L4 [claim] The results are organized as a ladder from theory to intervention. We first
- L17 [claim] audit supports.
- L21 [claim] First, recipe labels
- L32 [claim] is therefore an under-specification warning, not a backend causal estimate.

## `platform_hybrid/paper/sections/p7_synthesis.tex`

Consumers: P07

- L39 [claim] by the cross-examination. First, the interventional turn: T2 is not only an

## `platform_hybrid/paper/sections/p7_theory.tex`

Consumers: P07

- L23 [claim] The first factor is the contrast magnitude a non-degenerate group can carry
- L61 [claim] \emph{Empirical check (two independent grids).} First, a model~$\times$~$G$
- L126 [claim] first factor of Eq.~\eqref{eq:p7-S}, and it cannot be checked on the closed

## `platform_hybrid/paper/sections/p7_ushape.tex`

Consumers: P07

- L44 [claim] \emph{opposite} interventions. This is the mastery--incapacity aliasing that

## `platform_hybrid/paper/sections/p8_abstract.tex`

Consumers: P08

- L25 [claim] (iii) cold-start triage of novel fraud typologies before labels exist; and

## `platform_hybrid/paper/sections/p8_architecture.tex`

Consumers: P08

- L61 [claim] This placement has two consequences. First, the tree's accuracy and

## `platform_hybrid/paper/sections/p8_compliance.tex`

Consumers: P08

- L37 [claim] Opportunity Act as implemented by Regulation~B requires notification including
- L49 [claim] Two design consequences follow for fraud-adjacent credit decisions. First, the

## `platform_hybrid/paper/sections/p8_cost_optimal.tex`

Consumers: P08

- L3 [claim] \subsection{Cost-optimal decision threshold: is the sensor worth its price?}
- L4 [claim] \label{sec:p8-evidence-cost-optimal}
- L11 [claim] loss $L$ per \emph{missed} fraud, the cost-optimal threshold is
- L27 [claim] \caption{Cost-optimal operating points from \eqref{eq:p8-tau-star} on the
- L30 [claim] optimal. The sensor-only tree (\textsc{4sensor}) reaches comparable recall
- L33 [claim] \label{tab:p8-cost-optimal}
- L61 [claim] advantage (\$ per decision; positive $=$ first model cheaper) at each
- L66 [claim] \label{tab:p8-cost-optimal-boot}
- L82 [claim] \paragraph{What the cost-optimal frame settles.} Three findings, all on the
- L84 [claim] cost-ratio-dependent (Table~\ref{tab:p8-cost-optimal}): it moves from $0.18$
- L92 [claim] is marginal: the first break-even is $L^\star\!\approx\!\$5.50$ ($\rho^\star\!

## `platform_hybrid/paper/sections/p8_decision_disagreement.tex`

Consumers: P08

- L131 [claim] surrogate is statistically significantly more expensive at every

## `platform_hybrid/paper/sections/p8_evidence.tex`

Consumers: P08

- L85 [claim] The first and third rows of \tableref{tab:p8-evidence-ci} contain zero:
- L92 [claim] by a measurable margin). A re-runner is therefore guaranteed to see the
- L239 [claim] Two reliability facts follow from \tableref{tab:p8-reliability}. First,
- L373 [claim] The first observation is that \textbf{the 24-feature tree achieves the
- L440 [claim] dominant predictor to register above the noise floor of a paired
- L523 [claim] The first observation is that \textbf{M2 strictly improves TP at every
- L656 [claim] The first observation is that the \textbf{best F1 occurs at the same
- L675 [claim] gap is the dominant signal of the aggregate features' value.}
- L795 [claim] cost-optimal threshold $\tau^\star = L / (L + c_\text{inv})$ with paired
- L833 [claim] the cost-optimal threshold collapses to $\tau^\star \approx 1.0$ and
- L839 [claim] sensor is never cheaper than raw features at the cost-optimal threshold}.
- L878 [claim] sensor$+$raw \emph{outperforms} the LLM-only surrogate at $11/25$
- L911 [claim] the cost-optimal $\tau^\star$ is the same for both trees and the only
- L954 [claim] The cost-optimal thresholds of \secref{sec:p8-evidence-cost-curve} and the
- L967 [claim] the same cost as \secref{sec:p8-evidence-cost-optimal}, (3)~apply
- L1041 [claim] The cost-optimal frames of \secref{sec:p8-evidence-cost-optimal} and
- L1047 [claim] cost-optimal threshold. A noisy over-alerter may look cheap per stream
- L1051 [claim] $\tau^\star$ from \secref{sec:p8-evidence-cost-optimal}, on the same
- L1058 [claim] \caption{Cost-per-fraud-caught at the cost-optimal threshold on the
- L1161 [claim] small ($\sim$1--2~\textcent/dec) -- the opposite of where \secref{sec:p8-asymm-cost}
- L1320 [claim] The \texttt{sec:p8-cost-optimal} analysis is the correct tool for choosing
- L1719 [claim] ($\$0.001$/decision) is statistically significantly more expensive
- L1943 [claim] opposite (iter-68 row 78) held for \emph{AUC at $K{=}2\,\%$}
- L1972 [claim] sub-subsection iter-28 row 35 cost-optimal threshold; this iteration's
- L1989 [claim] dominant axis --- XGB-24full ($\$0.059$/dec) is preferred over

## `platform_hybrid/paper/sections/p8_future.tex`

Consumers: P08

- L9 [claim] \item \textbf{Sensor first.} Attach a VLM feature extractor to a

## `platform_hybrid/paper/sections/p8_intro.tex`

Consumers: P08

- L39 [claim] does not produce text at all. An LLM can triage a genuinely novel fraud
- L68 [claim] GRPO signal starvation, group-size control, PPO, or SAO. Its scientific claims

## `platform_hybrid/paper/sections/p8_iter108_cost_decision_cis.tex`

Consumers: P08

- L101 [claim] optimal cohort-level cost-benefit.

## `platform_hybrid/paper/sections/p8_iter112_cost_cis_realistic_rates.tex`

Consumers: P08

- L100 [claim] envelope for the first time at the CI level.

## `platform_hybrid/paper/sections/p8_iter116_llm_cost_sweep.tex`

Consumers: P08

- L63 [claim] (0.05\,\%--1.44\,\%), \textbf{xgb-only is the cost-optimal rule}.
- L76 [claim] $\times$ rule) is xgb-only-dominant at every interior cell.

## `platform_hybrid/paper/sections/p8_iter136_cal_realistic.tex`

Consumers: P08

- L17 [claim] fail when there are too few positives per cohort to support PAVA
- L72 [claim] support isotonic estimation. The OOF isotonic over-fits to the few

## `platform_hybrid/paper/sections/p8_iter160_operating_point_utility.tex`

Consumers: P08

- L72 [claim] adding more features shifts the optimal threshold such that realized
- L115 [claim] higher the LLM cost, the more the optimal threshold pushes toward
- L137 [claim] at every (rate, tier, fset) cell; iter-160 adds an \emph{optimal-tau}
- L148 [claim] top-K=2\% rule is sub-optimal} in terms of net value.
- L153 [claim] across all rates (the full feature bundle supports the highest-precision

## `platform_hybrid/paper/sections/p8_iter164_breakeven_tier.tex`

Consumers: P08

- L86 [claim] The two findings together mean \emph{VALUE-max is the dominant utility

## `platform_hybrid/paper/sections/p8_iter168_vmean_threshold.tex`

Consumers: P08

- L67 [claim] to support a 10\% precision gate at any cutoff.
- L88 [claim] because wasteful fires on non-fraud rows are eliminated first.
- L92 [claim] support a precision-constrained deployment; combining it with the 4

## `platform_hybrid/paper/sections/p8_iter172_vstat_ensemble.tex`

Consumers: P08

- L59 [claim] \textbf{H4 (FAIL -- degenerate)}: at the $\tau$ where esc\_prec first
- L61 [claim] \textbf{Measured: 0/0 = 0.0\%}. The first-5\%-tau event never fires;
- L124 [claim] geometry-driven firing is the dominant signal and the V-stats

## `platform_hybrid/paper/sections/p8_iter204_decile_cost_savings.tex`

Consumers: P08

- L40 [claim] over the decile's transactions and record the cost-optimal $t^{\star}$.
- L141 [claim] is the dominant single contributor (43\% of lift); iter-204's decile 0

## `platform_hybrid/paper/sections/p8_iter84_cohort_calibration.tex`

Consumers: P08

- L127 [claim] pair headline gets calibrated for the first time. The pair catches the

## `platform_hybrid/paper/sections/p8_noisy_sensor.tex`

Consumers: P08

- L4 [claim] \subsection{Is the cost-optimal break-even robust to sensor noise?}
- L7 [claim] The cost-optimal frame in \secref{sec:p8-evidence-cost-optimal} uses an
- L13 [claim] re-running the entire cost-optimal frame at five noise levels
- L26 [claim] \emph{exact} (no-bootstrap) cost-optimal $24\textsc{full}-20\text{raw}$
- L57 [claim] oracle frame in Table~\ref{tab:p8-cost-optimal-boot}: the certifiable
- L95 [claim] \secref{sec:p8-evidence-cost-optimal}, with $n{=}400$ paired bootstrap
- L107 [claim] production LLM sensor would face. \textbf{(iii)} Cost-optimal
- L115 [claim] reviewer-facing claim: \emph{the oracle cost-optimal break-even in
- L116 [claim] \secref{sec:p8-evidence-cost-optimal} is fragile to the noise a real

## `platform_hybrid/paper/sections/p8_related.tex`

Consumers: P08

- L8 [claim] models still outperform deep learning on medium-sized tabular benchmarks,
- L44 [claim] substantial headroom. Together they support both halves of our sensor claim:

## `platform_hybrid/paper/sections/p8_scorer.tex`

Consumers: P08

- L38 [claim] The artifact-backed ranking is only the first of five reasons the tree keeps

## `platform_hybrid/paper/sections/p8_setup.tex`

Consumers: P08

- L70 [claim] \paragraph{What this setup can and cannot support.}
- L75 [claim] support absolute performance claims about production fraud systems, and no

## `platform_hybrid/paper/sections/p8_taxonomy.tex`

Consumers: P08

- L17 [claim] first-party fraud claims. These artifacts are raw pixels and embedded
- L53 [claim] \subsection{Gap 3: Cold-Start on Novel Typologies (Few-Shot)}
- L57 [claim] fraud typology emerges---a novel scam pattern, a new mule-recruitment

## `platform_hybrid/paper/sections/related_work_v2.tex`

Consumers: R04,U01

- L28 [claim] Proximal Policy Optimization (PPO)~\cite{schulman2017proximal} is the workhorse
- L41 [claim] first reasoning RL method published in \emph{Nature}, which showed that
- L53 [claim] multimodal settings. Wu et al.~\cite{wu2025grpo_dpo} prove that GRPO is
- L74 [claim] before sampling; CPPO~\cite{lin2025cppo} prunes completions post-rollout,
- L95 [claim] that step-level PRMs dramatically outperform outcome reward models on GSM8K
- L153 [claim] WebShop~\cite{yao2022webshop} were the first large-scale web agents,
- L173 [claim] Compute-optimal training of base language models is governed by
- L175 [claim] characterised. Hilton et al.~\cite{hilton2023rlscaling} first derived
- L179 [claim] scaling laws for DPO and PPO that show a strong dependence on reward-model
- L215 [claim] al.~\cite{henderson2018deep} first exposed the fragility of deep-RL results
- L235 [claim] LLM RL post-training regime, contributing the first cross-library

## `platform_hybrid/paper/sections/robustness_smallscale_nulls.tex`

Consumers: P03

- L54 [claim] & Null (no robust optimal $G$). \\

## `platform_hybrid/paper/sections/scaling_law_iter101.tex`

Consumers: U01

- L38 [claim] on the first 75\%, compute AIC, and form the AIC-softmax stacked

## `platform_hybrid/paper/sections/scaling_law_iter105.tex`

Consumers: U01

- L41 [claim] \emph{description-first} taxonomy: the modes are not derived from a
- L101 [claim] families -- does \emph{not} support a monotone or asymptotic ceiling

## `platform_hybrid/paper/sections/scaling_law_iter109.tex`

Consumers: U01

- L119 [claim] The 48-row $t_X$ table (4 fractions $\times$ 12 anchors) shows that 7/12 anchors reach $t_{50}$ at step 1 (the trace \emph{starts} above 50\% of its peak), 5/12 reach $t_{90}$ within 3 steps, and 0/12 anchors take more than 5 steps to reach $t_{90}$. The traces are \emph{either a

## `platform_hybrid/paper/sections/scaling_law_iter29.tex`

Consumers: P01,U01

- L2 [claim] \subsection{Cross-Stack Identifiability Audit: GRPO and PPO Saturation Fits on the Same Rollouts}
- L23 [claim] PPO and GRPO should be performance-equivalent. If EEP holds, the
- L29 [claim] same-stack runs from \texttt{samestack\_ppo\_grpo.json} (5 GRPO + 5 PPO
- L42 [claim] \item[E1.] lambda-at-bound rate: PPO $=$ GRPO (Fisher exact, 2x2).
- L43 [claim] \item[E2.] AICc-best saturation rate: PPO $=$ GRPO (Fisher exact, 2x2).
- L45 [claim] PPO $=$ GRPO (Welch $t$, two-sided).
- L60 [claim] PPO & 5 & 0/5 & 0/5 & \textbf{5/5} & 0/5 & 0.078 \\
- L67 [claim] described by an exponential saturation curve (5/5), PPO's by a linear
- L77 [claim] Prediction & GRPO & PPO & $p$ & EEP \\
- L87 [claim] E2 is decisively falsified: GRPO and PPO prefer different AICc-best
- L91 [claim] and 4/5 PPO traces. E3 borderline ($p=0.054$): PPO's saturation fit
- L103 [claim] PPO's running reward is \emph{not}---its preferred model is linear with
- L113 [claim] degeneracy pattern flagged in the frontier synthesis: PPO's value
- L117 [claim] do not transfer to PPO} even when everything else in the stack is held
- L128 [claim] ``saturation-supported'' criterion in iter~25 (AICc-best $=$ sat AND
- L129 [claim] CI excludes bound) is satisfied in 5/5 GRPO traces and 0/5 PPO traces;

## `platform_hybrid/paper/sections/scaling_law_iter37.tex`

Consumers: P01,U01

- L40 [claim] the same-stack PPO/GRPO runs in
- L41 [claim] \texttt{experiments/results/samestack\_ppo\_grpo.json}
- L42 [claim] (5 GRPO + 5 PPO seeds, Qwen/Qwen2.5-0.5B).
- L77 [claim] \texttt{samestack\_ppo\_grpo.json} (Qwen2.5-0.5B, 5 seeds each of
- L78 [claim] GRPO and PPO). The result inverts: linear wins 0/5 on GRPO; the
- L79 [claim] Hill n=2 form wins 3/5, exponential wins 2/5, MM wins 0/5. On PPO,
- L82 [claim] $\bar{w}_{\mathrm{sat}}=0.38$, all other forms $\le 0.001$. On PPO,
- L157 [claim] on raw 40-step per-step traces, GRPO and PPO),

## `platform_hybrid/paper/sections/scaling_law_iter41.tex`

Consumers: P01,U01

- L30 [claim] Using only the first 60\% of the trace, the predicted $R_{\max}$ falls within $\pm 10\%$ of the full-fit $R_{\max}$ for 9/9 anchors; the bootstrap 95\% CI on the prediction error contains 0 for 7/9.

## `platform_hybrid/paper/sections/scaling_law_iter49.tex`

Consumers: P01,U01

- L51 [claim] P3: at median $\log_{10}C=5.09$, optimal $P\in[4,30]$B & \textbf{YES} & $P^{\star}=4$B & \textbf{Qwen3.5-4B} selected at the operating point \\
- L57 [claim] \paragraph{Iso-FLOP optimal anchor picker.}

## `platform_hybrid/paper/sections/scaling_law_iter53.tex`

Consumers: P01,U01

- L35 [claim] should be systematically over-predicted by LOO. The data do not support
- L40 [claim] supported and certainly weaker than the cross-stack compute signal.

## `platform_hybrid/paper/sections/scaling_law_iter61.tex`

Consumers: P01,U01

- L202 [claim] \paragraph{What iter 61 proves.}

## `platform_hybrid/paper/sections/scaling_law_iter65.tex`

Consumers: P01,U01

- L140 [claim] \paragraph{What iter 65 proves.}

## `platform_hybrid/paper/sections/scaling_law_iter69.tex`

Consumers: U01

- L26 [claim] \item the first-difference variance $\mathrm{Var}[\Delta y]$;
- L47 [claim] first-difference variance \emph{decreases} with model size
- L136 [claim] \paragraph{What iter 69 proves.}

## `platform_hybrid/paper/sections/scaling_law_iter85.tex`

Consumers: U01

- L31 [claim] The canonical temporal ordering (creep first, spurt in the middle, level
- L49 [claim] the \emph{first} segment, and the canonical \emph{level} sits at a low
- L146 [claim] stronger support than the saturation law's 2/12 in iter 73. conf$_4$
- L150 [claim] and temporal order, but its spurt is the \emph{first} segment (peak at

## `platform_hybrid/paper/sections/scaling_law_iter89.tex`

Consumers: U01

- L12 [claim] as direct empirical support for the \emph{creep $\to$ spurt $\to$ level}
- L55 [claim] \paragraph{Result 3 (k-fold forecast): training on the first
- L78 [claim] trace length under resampling. The first change-point (CP1) shows
- L114 [claim] the first $n-4$. Constant-mean wins (A), (B), (D); only (C) is a

## `platform_hybrid/paper/sections/scaling_law_iter93.tex`

Consumers: U01

- L13 [claim] \citet{nimmaturi2025predictive} (arXiv:2507.18014) supports the
- L51 [claim] \emph{Forecast MAE on the last four steps} (trained on the first
- L55 [claim] Constant-mean is \emph{not} the forecast-optimal model on the iter 81
- L62 [claim] Qwen3.5-4B). This is the first quantitative evidence in the

## `platform_hybrid/paper/sections/scaling_law_iter97.tex`

Consumers: U01

- L26 [claim] ($k=5$, optimal changepoint + two independent OLS lines). Each new
- L48 [claim] as a single dominant universal family} -- the same outcome as iter 93
- L80 [claim] wins that criterion. Bold = mode. Constant-mean is the dominant
- L82 [claim] AR(1) is the dominant forecasting family; pw2seg wins in-sample
- L151 [claim] scale-dependent variance correction -- the data do not support one.

## `platform_hybrid/paper/sections/scaling_laws.tex`

Consumers: P01,U01

- L26 [claim] $t_{80} = -\ln(0.2)/\lambda$ at which the trace first crosses
- L269 [claim] fitting a constant. We fit the saturation curve on the first 70\%
- L362 [claim] (companion paper on cross-experiment ZVF) would catch first, because it
- L475 [claim] as a \emph{descriptive} comparison, not as a causal claim.
- L481 [claim] The first-final gap $\Delta_{1T} = R(t{=}1) - R(t{=}T)$ adds a
- L489 [claim] \emph{starts} at zero reward, so the first-minus-final contrast
- L492 [claim] the right diagnostic for that trace rather than the first-final gap.
- L502 [claim] stratified).} (a)~\emph{First-step reward} $R(t{=}1)$ vs
- L510 [claim] (d)~\emph{First-final gap} $R(1) - R(T)$ per model; the
- L518 [claim] \paragraph{What the extension proves and what it does not.}
- L533 [claim] \emph{descriptive} rather than causal.
- L556 [claim] functional form the data best support, with the constant model
- L600 [claim] \in (2, 4)$ is ``substantial support''; $\Delta\text{AIC} > 4$
- L601 [claim] is ``essentially no support''. The saturation model is
- L620 [claim] border of substantial support), and on most (model, anchor)
- L948 [claim] cross-architecture claim supported by the data is that
- L997 [claim] $t_{80}$ read off a single trace are not supported by the evidence at
- L1032 [claim] (green) is unsupported.}
- L1089 [claim] (i)~Qwen3-8B's BIC-optimal $k=1$ has no slow-start phase, and
- L1152 [claim] \tableref{tab:scaling-iter117-bic}: the BIC-optimal segmentation is
- L1190 [claim] lines are the BIC-optimal segment means. Nemotron-120B
- L1234 [claim] than the smaller ones --- the opposite sign from any Chinchilla-style
- L1412 [claim] $(p_1, p_2, p_3) = (1, 1, 1)$ signature; the dominant pattern is
- L1530 [claim] limited to the five-anchor pool that supports the full 30-step GRPO
- L1562 [claim] dominant pattern is still collapse_only ($5/7$), so the template is

## `platform_hybrid/paper/sections/scaling_passk_ci.tex`

Consumers: P01

- L27 [claim] supplies two correctives. First, the \emph{agent--computer interface}
- L29 [claim] parsed --- is a first-order determinant of measured capability, often

## `platform_hybrid/paper/sections/stat_rigor_updates.tex`

Consumers: U01

- L12 [claim] For every comparison reported in Tables~\ref{tab:main_results_stats}--\ref{tab:ppo_grpo_stats},
- L24 [claim] \caption{\textbf{Main Results (Table~1, rigor pass).} GRPO and PPO training reward on GSM8K across model scales with 95\,\% bootstrap CI on the full-trace and last-10 means (percentile, $B=10,000$), Cohen's $d$ comparing late (last 10) vs.\ early (first 10) training with 95\,\% H
- L39 [claim] PPO (Modal H100) & Qwen3-8B & 30 & 35.0\% [17.5\%, 52.5\%] & 28.3\% [18.3\%, 38.3\%] & +0.08 & [-0.80, 0.96] & 0.782 & 1.000 \\
- L40 [claim] PPO (Modal H100) & Llama-3.1-8B-Inst & 30 & 95.0\% [87.5\%, 100.0\%] & 95.0\% [90.0\%, 99.2\%] & -0.27 & [-1.15, 0.61] & 0.583 & 1.000 \\
- L61 [claim] SB3 (PPO) & 5 & 0.009 [0.007, 0.010] & \textit{5 seeds} & +14.59 & [8.08, 21.10] & $<$0.001 & $<$0.001$^{***}$ \\
- L62 [claim] CleanRL (PPO) & 5 & 0.007 [0.003, 0.010] & \textit{5 seeds} & +14.61 & [8.09, 21.13] & $<$0.001 & $<$0.001$^{***}$ \\
- L63 [claim] Tianshou (PPO) & 5 & 0.008 [0.006, 0.011] & \textit{5 seeds} & +14.58 & [8.07, 21.09] & $<$0.001 & $<$0.001$^{***}$ \\
- L93 [claim] \caption{\textbf{PPO vs.\ GRPO (Table~4, rigor pass).} Per-step reward (GSM8K, single seed, $n=30$) with 95\,\% bootstrap CI on last-10 and full-trace means, bootstrap CI on $\mu_{\mathrm{GRPO}}-\mu_{\mathrm{PPO}}$, Cohen's $d$ with Hedges--Olkin CI, and Bonferroni-corrected Welc
- L94 [claim] \label{tab:ppo_grpo_stats}
- L99 [claim] \textbf{Model} & \textbf{GRPO last-10 [95\% CI]} & \textbf{PPO last-10 [95\% CI]} & \textbf{$\Delta$ [95\% CI]} & \textbf{Cohen's $d$} & \textbf{$d$ 95\% CI} & \textbf{Welch $p$} & \textbf{Welch $p$ (Bonf.)} & \textbf{MW $p$} & \textbf{MW $p$ (Bonf.)} \\
- L116 [claim] numbers in Tables~\ref{tab:main_results_stats}--\ref{tab:ppo_grpo_stats}
- L157 [claim] two-sided Mann--Whitney $U$ on the first-10 and last-10 reward samples
- L164 [claim] \item \textbf{Table~\ref{tab:ppo_grpo_stats}} (matched-model PPO vs.\ GRPO):
- L171 [claim] \ref{tab:main_results_stats}--\ref{tab:ppo_grpo_stats}. For every raw
- L206 [claim] 1 & Table 2 & CleanRL (PPO) vs TRL (GRPO) (final arithmetic accuracy) & Welch t-test vs TRL (GRPO) & +14.61 & $<$0.001 & $<$0.001$^{***}$ \\
- L207 [claim] 2 & Table 2 & Tianshou (PPO) vs TRL (GRPO) (final arithmetic accuracy) & Welch t-test vs TRL (GRPO) & +14.58 & $<$0.001 & $<$0.001$^{***}$ \\
- L208 [claim] 3 & Table 2 & SB3 (PPO) vs TRL (GRPO) (final arithmetic accuracy) & Welch t-test vs TRL (GRPO) & +14.59 & $<$0.001 & $<$0.001$^{***}$ \\
- L217 [claim] 12 & Table 1 & Late-10 vs Early-10 reward (tianshou\_ppo\_math\_s42) & Mann-Whitney U (late vs early) & +1.81 & 0.002 & 0.080 \\
- L218 [claim] 13 & Table 1 & Late-10 vs Early-10 reward (sb3\_ppo\_math\_s1024) & Mann-Whitney U (late vs early) & +1.37 & 0.005 & 0.208 \\
- L219 [claim] 14 & Table 4 & Llama-3.1-8B-Inst: PPO vs GRPO (Mann-Whitney U) & Mann-Whitney U & -0.56 & 0.006 & 0.219 \\
- L220 [claim] 15 & Table 4 & Llama-3.1-8B-Inst: PPO vs GRPO (full trace, n=30) & Welch t-test & -0.56 & 0.035 & 1.000 \\
- L223 [claim] 18 & Table 1 & Late-10 vs Early-10 reward (tianshou\_ppo\_math\_s456) & Mann-Whitney U (late vs early) & +0.52 & 0.246 & 1.000 \\
- L224 [claim] 19 & Table 1 & Late-10 vs Early-10 reward (sb3\_ppo\_math\_s789) & Mann-Whitney U (late vs early) & +0.51 & 0.254 & 1.000 \\
- L226 [claim] 21 & Table 1 & Late-10 vs Early-10 reward (sb3\_ppo\_math\_s123) & Mann-Whitney U (late vs early) & -0.44 & 0.390 & 1.000 \\
- L227 [claim] 22 & Table 1 & Late-10 vs Early-10 reward (tianshou\_ppo\_math\_s123) & Mann-Whitney U (late vs early) & +0.46 & 0.390 & 1.000 \\
- L228 [claim] 23 & Table 1 & Late-10 vs Early-10 reward (sb3\_ppo\_math\_s42) & Mann-Whitney U (late vs early) & +0.27 & 0.455 & 1.000 \\
- L229 [claim] 24 & Table 1 & Late-10 vs Early-10 reward (sb3\_ppo\_math\_s456) & Mann-Whitney U (late vs early) & -0.22 & 0.459 & 1.000 \\
- L230 [claim] 25 & Table 1 & Late-10 vs Early-10 reward (ppo\_llama-8b-inst) & Mann-Whitney U (late vs early) & -0.27 & 0.583 & 1.000 \\
- L234 [claim] 29 & Table 4 & Qwen3-8B: PPO vs GRPO (Mann-Whitney U) & Mann-Whitney U & +0.01 & 0.709 & 1.000 \\
- L235 [claim] 30 & Table 1 & Late-10 vs Early-10 reward (ppo\_qwen3-8b) & Mann-Whitney U (late vs early) & +0.08 & 0.782 & 1.000 \\
- L236 [claim] 31 & Table 1 & Late-10 vs Early-10 reward (tianshou\_ppo\_math\_s789) & Mann-Whitney U (late vs early) & +0.00 & 0.813 & 1.000 \\
- L239 [claim] 34 & Table 1 & Late-10 vs Early-10 reward (tianshou\_ppo\_math\_s1024) & Mann-Whitney U (late vs early) & +0.00 & 0.938 & 1.000 \\
- L241 [claim] 36 & Table 4 & Qwen3-8B: PPO vs GRPO (full trace, n=30) & Welch t-test & +0.01 & 0.973 & 1.000 \\

## `platform_hybrid/paper/sections/statistical_rigor_addendum.tex`

Consumers: U01

- L26 [claim] Tables~\ref{tab:main_results_stats}--\ref{tab:ppo_grpo_stats} treat
- L63 [claim] Tier-A groups carry enough replication and horizon to support a
- L67 [claim] tier. Tier-B groups support \emph{supporting} evidence: CIs and
- L70 [claim] \emph{all} Tinker API runs, the partial Modal Qwen3-32B PPO run, and
- L79 [claim] Tables~\ref{tab:main_results_stats}--\ref{tab:ppo_grpo_stats} is
- L110 [claim] support for any headline claim and we additionally compute a permutation
- L131 [claim] its GRPO and PPO arms across frameworks rather than staying within one---see the
- L147 [claim] $F_3$ & PPO/GRPO heterogeneity & yes & no & $+21.84$ & $2.1\times 10^{-5}$ & \textbf{survives} \\
- L157 [claim] against $15$ pooled \emph{classic-RL} PPO seeds (SB3, CleanRL, Tianshou; $n{=}20$
- L159 [claim] cross-paradigm contrast --- LLM-GRPO reward against gym-style PPO return --- so
- L163 [claim] heterogeneity this paper documents, \emph{not} an algorithmic superiority of
- L164 [claim] GRPO over PPO under matched conditions (cf.\ the same-stack null in the main
- L165 [claim] text). $F_1$ (ZVF as a diagnostic) is supported only by
- L195 [claim] \item \textbf{PPO vs.\ GRPO on Qwen3-8B.} The single-seed rows
- L196 [claim] $0.344$ (GRPO) and $0.350$ (PPO) do not enter the BH family.
- L198 [claim] Table~\ref{tab:ppo_grpo_stats} (computed over independent
- L214 [claim] power caveats on this null. First, a seed-level Welch test at $n_1{=}n_2{=}5$
- L253 [claim] columns \texttt{finding}, \texttt{tier\_a\_support},
- L254 [claim] \texttt{tier\_b\_support}, \texttt{effect\_size\_cohens\_d},

## `platform_hybrid/paper/sections/synth_iter112_cost_rate_envelope.tex`

Consumers: P08

- L48 [claim] \subsubsection{Falsifiable headline H2 (rate-dependent optimal LLM rule)}
- L58 [claim] Pareto-dominant on \$/caught (per iter-112 H2).
- L68 [claim] realistic positive rate, xgb-only is the cost-optimal LLM-free
- L72 [claim] This is the first time P5P8 has surfaced a closed-form
- L74 [claim] \textbf{(b)} is supported by paired-row bootstrap CIs, and

## `platform_hybrid/paper/sections/synth_iter116_cost_cube_envelope.tex`

Consumers: P08

- L17 [claim] \subsubsection{Falsifiable headline H1 (envelope is xgb-only-dominant)}
- L55 [claim] The operational finding -- \textbf{xgb-only is the cost-optimal LLM-free

## `platform_hybrid/paper/sections/synth_iter132_four_domain_density.tex`

Consumers: P08

- L88 [claim] \emph{also} high-boundary-density methods, opposite the predicted

## `platform_hybrid/paper/sections/synth_iter160_twelve_domain_density.tex`

Consumers: P08

- L96 [claim] D12 is the **12th** domain in the SYNTH roll-up and the **first** P5-only

## `platform_hybrid/paper/sections/synth_iter164_thirteen_domain_density.tex`

Consumers: P08

- L63 [claim] first structural-LOW domain — its LOW status is determined by the

## `platform_hybrid/paper/sections/synth_iter192_d20_decision_concordance.tex`

Consumers: P08

- L99 [claim] \texttt{gift} is the deployment-optimal method (best on P5 mean reward
- L100 [claim] + P8 transfer) but \textbf{NOT} the registry/controller-optimal method
- L102 [claim] claim ($\mathrm{zvf\_risk}{<}0 \rightarrow \mathrm{SUPPORTS}$) is

## `platform_hybrid/paper/sections/synth_iter204_d23_perstep_transfer.tex`

Consumers: P08

- L17 [claim] answers with the \textbf{first per-step transfer stability} audit of the
- L34 [claim] \texttt{P8=reward/mean\_len}) and the same D22 cost-optimal weight

## `platform_hybrid/paper/sections/tool_use_code_expanded.tex`

Consumers: U01

- L80 [claim] \paragraph{Binary vs.~graded.} The dominant family is strict binary
- L97 [claim] \texttt{```json} fences and extracts the first \texttt{\{...\}}
- L148 [claim] depends on the reward's \emph{support structure}, not just its type.
- L222 [claim] flag them as future work rather than as supporting evidence

## `platform_hybrid/paper/sections/variance_mitigation_comparison.tex`

Consumers: U01

- L18 [claim] understate the state of the art: a recent line of work proposes
- L27 [claim] CPPO~\cite{lin2025cppo}, NGRPO~\cite{nan2025ngrpo},
- L45 [claim] \paragraph{CPPO (Completion Pruning Policy Optimization)~\cite{lin2025cppo}.}
- L46 [claim] CPPO~\citep{lin2025cppo} targets the \emph{compute} cost of GRPO: it prunes
- L127 [claim] + CPPO~\cite{lin2025cppo} & no & yes & completion pruning (compute) & no & $0.85\times$ \\
- L135 [claim] CPPO prunes completions for throughput. Compute is relative to baseline GRPO at
- L147 [claim] hooks are: an advantage-pruning variant (a CPPO-style throughput proxy that drops
- L156 [claim] accepts \texttt{--method \{grpo,cppo,ngrpo,scafgrpo\}} and
- L178 [claim] + CPPO$^{\dagger}$ & $0.439$ & $43.3$ & $2 / 5$ & $0.64$ & $78$ \\
- L215 [claim] Qwen3-8B / GSM8K at 5 seeds and 100 steps for each of CPPO,
- L264 [claim] + CPPO & $0.48$ & $0.66$ & yes \\
- L272 [claim] of step-100 collapse under CPPO and NGRPO: these methods
- L285 [claim] variance-mitigation methods (CPPO, NGRPO, which reduce or reweight after the

## `platform_hybrid/paper/sections/zvf.tex`

Consumers: P02,U01

- L33 [claim] CPPO & 0.295 & 0.392 & 0/5 & 0/5 & 5/5 \\
- L60 [claim] vanilla GRPO. CPPO, NGRPO, and SCAFGRPO sit in between on both axes.
- L80 [claim] The residual diagnostic that exists on disk gives the opposite caution:
- L90 [claim] $\rho \approx 0.9$) and were deliberately aggregated first.
- L105 [claim] \subsection{Why We Treat ZVF as Diagnostic, Not Causal}
- L113 [claim] the earliest cheap diagnostic we have for it. On this corpus the first-passage
- L122 [claim] stacks without re-measurement. The opposite-direction gap between
- L128 [claim] GRPO/PPO ablation table; treat a $>0.3$ rise in mean ZVF over
- L139 [claim] scaling_law_three_phase, drgrpo_vs_grpo, samestack_ppo_grpo\}}. The
- L170 [claim] \item \textbf{Drift channel} --- first-half trajectory slope of ZVF,
- L202 [claim] wide margin on both panels). It is the first ZVF-based diagnostic to
- L235 [claim] NGRPO / CPPO / AERO & 0.43--0.49 & 0.0 \\
- L257 [claim] first iteration of ZVF as a \emph{three-channel, real-time} diagnostic
- L281 [claim] A HippoRAG-style KG+PPR retrieval scaffold~\citep{gutierrez2024hipporag}
- L290 [claim] (\texttt{experiments/results/berkeley/hipporag\_eval.tsv}).

## `platform_hybrid/paper/sections/zvf_cross_experiment_diagnostic.tex`

Consumers: P02,U01

- L180 [claim] CPPO & 5 & 0.295 & 0.298 & 0/5 & 0.392 \\
- L222 [claim] The other eight mitigation methods (CPPO, NGRPO, SCAFGRPO,
- L339 [claim] is supported at the lower bound on the real reasoning model

## `platform_hybrid/paper/sections/zvf_dynamics.tex`

Consumers: P02,U01

- L42 [claim] &&\text{(first-passage step to ZVF=}\theta\text{)}\\
- L64 [claim] next-worst method (CPPO, $0.065$) and over 6$\times$ the GIFT/ES/AReaL
- L85 [claim] method and is largest for GRPO ($+0.83$), CPPO ($+0.68$), and
- L87 [claim] GIFT ($+0.29$). Notably even though CPPO and NGRPO have lower
- L102 [claim] cppo & 0.012 & 0.171 & 0.690 & 0.976 & 0.065 & 0.010 & 0.000 \\
- L136 [claim] $\tau(\theta; z) = $ first step where $z_i \ge \theta$ and
- L137 [claim] $t_{\text{collapse}} = $ first step where the per-step collapse flag
- L144 [claim] the 3-event first-passage lead. The numbers are small, in line with
- L214 [claim] \caption{First-passage step $\tau(\theta; z)$ vs.\ first collapse-flag

## `platform_hybrid/paper/sections/zvf_eval_protocol.tex`

Consumers: P02

- L26 [claim] 8/9 methods are magnitude-dominant while \texttt{grpo} itself is
- L27 [claim] drift-dominant --- variance mitigation acts chiefly by suppressing the

## `platform_hybrid/paper/sections/zvf_inference_baselines.tex`

Consumers: P02

- L39 [claim] magnitude-dominant methods (mean $2.82$\,pp) while the one
- L40 [claim] drift-dominant method (\texttt{grpo}) moves only $1.49$\,pp --- exactly

## `platform_hybrid/paper/sections/zvf_iter102.tex`

Consumers: U01

- L52 [claim] \texttt{cppo} & 1 & $+0.276$ & 15.35 \\

## `platform_hybrid/paper/sections/zvf_iter106.tex`

Consumers: U01

- L68 [claim] both have $\mathrm{ZVF}\approx 1$ but live at opposite ends of the $p$
- L69 [claim] axis and on opposite sides of $\Delta=0$.
- L111 [claim] ceiling) while $\Delta$ ranks them at opposite ends. Hence the rank

## `platform_hybrid/paper/sections/zvf_iter110.tex`

Consumers: U01

- L86 [claim] opposite ends of its range; yet $\Delta$ and $\rho$ have smaller EMD
- L96 [claim] gives the practitioner the minimum group size needed to guarantee the
- L126 [claim] guarantee $\mathrm{ZVF}_{\rm iid}\!\leq\!0.7$. This U-shape is
- L139 [claim] convergence time} (first step where mean reward $\geq 0.5$) as the
- L153 [claim] $-11$ steps (the reward converges first, then ZVF saturates later
- L189 [claim] \noindent The lead-time table provides a third, causally grounded

## `platform_hybrid/paper/sections/zvf_iter114.tex`

Consumers: U01

- L87 [claim] CPPO & 0.295 & 0.392 & \(+0.276\) & herd \\
- L134 [claim] - \bigl(p^{G}+(1-p)^{G}\bigr)\) on the first 5 training windows.

## `platform_hybrid/paper/sections/zvf_iter118.tex`

Consumers: U01

- L12 [claim] \subsection{ZVF as a First-Class Cross-Library Diagnostic vs AERO}
- L13 [claim] \label{sec:zvf-first-class}
- L19 [claim] \emph{first-class diagnostic vs AERO} by reporting three summary
- L33 [claim] regimes sit at opposite ends of the ZVF distribution, and a scalar
- L58 [claim] (mean-ZVF $0.30$--$0.32$) is the AERO/CPPO/NGRPO/SCAFGRPO band with
- L83 [claim] ($0.30$--$0.32$) where AERO/CPPO/NGRPO/SCAFGRPO live. Source:
- L89 [claim] ZVF is a \emph{first-class} cross-library diagnostic because it

## `platform_hybrid/paper/sections/zvf_iter122.tex`

Consumers: U01

- L15 [claim] The iter 118 section turned ZVF into a \emph{first-class}
- L145 [claim] Iter 122 turns the first-class ZVF diagnostic into the three

## `platform_hybrid/paper/sections/zvf_iter22.tex`

Consumers: P02,U01

- L62 [claim] cppo & 5 & 0.295 & [0.293, 0.297] & 0.000 & [0.000, 0.000] \\
- L73 [claim] The AERO mean-ZVF CI [0.219, 0.222] sits cleanly between CPPO/NGRPO/SCAFGRPO
- L111 [claim] middle gap between CPPO/NGRPO/SCAFGRPO and the low-ZVF cluster
- L129 [claim] \item \texttt{first\_pass\_step} = $\arg\max_{t}(\text{heldout\_acc}(t))$,
- L131 [claim] \item \texttt{first\_collapse\_step} = first step $> $ \texttt{first\_pass\_step} with \texttt{collapse}=1.
- L132 [claim] \item \texttt{lead\_steps} = \texttt{first\_collapse\_step} $-$ \texttt{first\_pass\_step}.
- L133 [claim] \item \texttt{pre\_zvf\_5} = mean ZVF over $[\texttt{first\_pass\_step}, \texttt{first\_pass\_step}+5)$.
- L134 [claim] \item \texttt{post\_zvf\_5} = mean ZVF over $(\texttt{first\_collapse\_step}-5, \texttt{first\_collapse\_step}]$.
- L148 [claim] cppo & 5 & 0 & 0.00 & --- & 0.783 & 0.776 & 0.771 \\
- L159 [claim] between the heldout-accuracy peak and the first collapse flag.
- L228 [claim] CPPO/NGRPO/SCAFGRPO's CIs. This is the strongest single-number
- L234 [claim] ``ZVF rises then collapse'' causal story.

## `platform_hybrid/paper/sections/zvf_iter30.tex`

Consumers: P02,U01

- L136 [claim] better collapse predictions as the future window grows, exactly the opposite

## `platform_hybrid/paper/sections/zvf_iter34.tex`

Consumers: P02,U01

- L49 [claim] The iter30 analysis proved \emph{on the variance-mitigation
- L127 [claim] ranks \texttt{zvf\_direction} as the dominant predictor

## `platform_hybrid/paper/sections/zvf_iter38.tex`

Consumers: P02,U01

- L55 [claim] CPPO & 0.392 & 4 & 3 & 1 & 25.0\% \\

## `platform_hybrid/paper/sections/zvf_iter42.tex`

Consumers: P02,U01

- L17 [claim] computed only over the first half of the trajectory?} The motivation
- L23 [claim] per-step ZVF summaries (\texttt{first\_pass\_zvf05}, \texttt{auc\_above\_zvf05},
- L29 [claim] \mathrm{fp05\_frac} &:= \mathrm{first\_pass\_zvf05}/n_{\mathrm{steps}}
- L65 [claim] of the first crossing. \texttt{zvf\_lag1} has AUC $0.105$ (the
- L72 [claim] identifiable from the first 60\% of the trace on 9/9 eligible anchors
- L79 [claim] abort a doomed RL run within the first 30--50\% of the trace on the
- L85 [claim] informative. The first-pass threshold captures \emph{when} ZVF first

## `platform_hybrid/paper/sections/zvf_iter46.tex`

Consumers: P02,U01

- L38 [claim] On the 505 prompts with $p_x \in (0.05, 0.95)$ (the finite-support
- L124 [claim] n=505 finite-support prompt pool.
- L142 [claim] on the n=505 finite-support prompt pool. P1 and P3 measure per-prompt
- L163 [claim] Three caveats bound the operational claim. \emph{First}, the Iso-G
- L173 [claim] guarantee a downstream accuracy gain -- a researcher who allocates

## `platform_hybrid/paper/sections/zvf_iter50.tex`

Consumers: P02,U01

- L18 [claim] first step at which the per-step ZVF crosses $\theta{=}0.5$
- L19 [claim] precedes the first step at which the heldout accuracy collapse flag
- L56 [claim] \emph{$\mathrm{ZVF}_{0.5}$ first-crosses first, then the
- L82 [claim] CPPO & 0.694 & 0.719 & 0.731 & 0.738 & 0.746 & 0.768 & \textbf{0.798} & yes \\
- L110 [claim] (a $49\%$ reduction), CPPO $0.275$, NGRPO $0.240$, SCAF-GRPO $0.000$,

## `platform_hybrid/paper/sections/zvf_iter58.tex`

Consumers: P02,U01

- L28 [claim] The two summands are \emph{structurally opposite}. ZVF$^-$ is
- L98 [claim] separators, in opposite directions -- against $0.396$ for raw ZVF, which

## `platform_hybrid/paper/sections/zvf_iter62.tex`

Consumers: P02,U01

- L25 [claim] structurally opposite collision types, and uses that split to rank
- L56 [claim] CPPO & 0.295 & 0.010 & 0.153 & 0.542 & 0.750 & 0.739 \\
- L134 [claim] \in [0.5, 0.9]$ -- GRPO, CPPO, NGRPO, AERO, MCGRPO, GIFT, AREAL --

## `platform_hybrid/paper/sections/zvf_iter70.tex`

Consumers: U01

- L38 [claim] The mean of the first quartile of the step trace minus the mean of
- L66 [claim] CPPO & 5 & 0.295 & 0 & 0.000 & 0.182 & 0.000 & \\
- L95 [claim] \emph{not} supported as a pooled-experiment rule. The signal
- L103 [claim] support a correlation.
- L112 [claim] $\overline{\mathrm{ZVF}}\ge 0.22$ (AERO, CPPO, NGRPO, SCAFGRPO, GRPO)
- L120 [claim] The two \texttt{tool\_use} trajectories sit at $(\mathrm{ZVF}_{\mathrm{first}}=1.0, \mathrm{ZVF}_{\mathrm{last}}=1.0, \Delta = 0)$
- L122 [claim] the first quartile. The \texttt{groupsize\_zvf\_sweep} mean is
- L123 [claim] $(\mathrm{ZVF}_{\mathrm{first}}=0.75, \mathrm{ZVF}_{\mathrm{last}}=0.71, \Delta = -0.04)$:
- L143 [claim] ZVF trajectory direction: first-25\% vs last-25\% of the step

## `platform_hybrid/paper/sections/zvf_iter74.tex`

Consumers: U01

- L51 [claim] $\pi_H = 0$ (no trace ever visits $H$); \textsc{AERO}, \textsc{CPPO},
- L70 [claim] \textsc{cppo} & 0.322 & 0.987 & 25.7 & 0.392 & 3 \\
- L85 [claim] \textsc{SCAFGRPO} never reach $H$; \textsc{AERO}, \textsc{CPPO},
- L110 [claim] records the opposite trajectory: $\pi_H = 0.0$, with all 5 steps

## `platform_hybrid/paper/sections/zvf_iter82.tex`

Consumers: U01

- L20 [claim] accuracy drops below $0.10$ the alarm first fires. Iter~82 re-frames
- L41 [claim] \big[64.9,\;74.0,\;85.1,\;74.0,\;143.0\big]_{[\mathrm{aero,cppo,grpo,mcgrpo,ngrpo}]},
- L79 [claim] ($\mathrm{ZVF} \in [0.13,\,0.30]$, $n=17$, dominated by AERO, CPPO,
- L86 [claim] It is the first cheap binary alarm in this benchmark that
- L112 [claim] CPPO & 48 & 339 & 5 & 0 & 70.8 & 0.896 & 0.189 \\

## `platform_hybrid/paper/sections/zvf_iter86.tex`

Consumers: U01

- L100 [claim] $K=1$ curve at every $C_\mathrm{ratio}$: it is cost-sub-optimal at
- L124 [claim] \item Persistence $K$ is the dominant knob: raising $K$ from 1 to

## `platform_hybrid/paper/sections/zvf_iter90.tex`

Consumers: U01

- L18 [claim] selecting $(\tau=0.6,\, K=5)$ as the optimal cost-vs-savings pair on
- L50 [claim] \textsc{cppo} 0.286, \textsc{ngrpo} 0.286,
- L72 [claim] \textsc{cppo} & 0.367 & 0.396 & -0.029 \\
- L97 [claim] are \textsc{cppo} and \textsc{ngrpo}, and even there the gap is

## `platform_hybrid/paper/sections/zvf_iter94.tex`

Consumers: U01

- L89 [claim] ($+6.35$), \textsc{cppo} ($+3.45$), \textsc{ngrpo} ($+3.41$).
- L95 [claim] The 14-column summary table is the canonical first-class diagnostic

## `platform_hybrid/paper/sections/zvf_iter98.tex`

Consumers: U01

- L112 [claim] first-class diagnostic; iter~98 appends six columns that report the

## `platform_hybrid/paper/sections/zvf_scaling.tex`

Consumers: P02,U01

- L62 [claim] The collapse rule used in \texttt{scaling\_law\_three\_phase.tsv} is $p > 0.7 \wedge \ell < 0.35$, a peak-vs-last10 rule. The ZVF-proxy rule $\text{frac\_below\_0p1} \geq 0.1$ picks the same row. The two rules are not identical ($p$ is the maximum heldout reward, $\text{frac\_bel

## `platform_hybrid/paper/unified_signal_starvation/main.tex`

Consumers: N01

- L39 [claim] \newcommand{\ppo}{\textsc{PPO}\xspace}
- L40 [claim] \newcommand{\sao}{\textsc{SAO}\xspace}
- L51 [claim] A Diagnostic and Controller Proposal Across GRPO, PPO, and\\
- L66 [claim] near-zero advantage in PPO, and policy lag can cause double-sided importance
- L68 [claim] Asynchronous Optimization (SAO). We give these cases one operational
- L74 [claim] a proxy, because token gradients can cancel. This decomposition supports
- L88 [claim] seed-paired evaluation contract for PPO and SAO rather than manufacturing
- L91 [claim] failure mode to critic-based PPO and asynchronous SAO. It reuses companion GRPO
- L108 [claim] identical group rewards therefore erase the advantage. PPO uses a learned
- L110 [claim] \citep{schulman2017ppo}. SAO replaces prompt groups with single rollouts and
- L112 [claim] under policy lag \citep{hou2026sao}. The official GLM-5.2 report describes a
- L115 [claim] optimization to critic-based PPO over individual rollouts
- L129 [claim] targeted resampling. In PPO and SAO, low PAM with a calibrated critic means
- L139 [claim] signal in GRPO, PPO, and SAO, together with an exact zero-update certificate
- L147 [claim] GRPO facts from untested PPO/SAO hypotheses.
- L152 [claim] unified controller is a method proposal. Its PPO and SAO benefits are
- L181 [claim] token. A later PPO-style clip may also suppress signal, but the defining
- L186 [claim] \subsection{PPO: credit can be absent or clipped away}
- L188 [claim] PPO estimates $A_t$ with a critic, commonly using generalized advantage
- L191 [claim] \ell_t^{\mathrm{PPO}}(\theta)=
- L194 [claim] \label{eq:ppo}
- L198 [claim] c_t^{\mathrm{PPO}}=m_t^{\mathrm{PPO}}\rho_t A_t,
- L200 [claim] m_t^{\mathrm{PPO}}=
- L206 [claim] \label{eq:ppo-gate}
- L208 [claim] PPO can therefore starve because $A_t\approx0$ (credit starvation) or because
- L213 [claim] \subsection{SAO: strict survival under asynchronous policy lag}
- L215 [claim] SAO is designed for asynchronous, single-rollout agentic RL. It uses a
- L218 [claim] tokens when constructing token-level GAE \citep{hou2026sao}. To stabilize
- L222 [claim] m_t^{\mathrm{SAO}}=ind\!\left[1-\epsilon_l < \rho_t < 1+\epsilon_h\right],
- L224 [claim] c_t^{\mathrm{SAO}}=m_t^{\mathrm{SAO}}\rho_t A_t.
- L225 [claim] \label{eq:sao-gate}
- L227 [claim] Unlike PPO's sign-dependent gate, DIS masks either sign outside the interval.
- L325 [claim] GRPO & Within-prompt group reward & PPO-style gate, if used & Flat group makes $A^{\rm grp}=0$ \\
- L326 [claim] PPO & Critic plus GAE & Sign-dependent clipping & $A=0$ or all nonzero terms clipped \\
- L327 [claim] SAO & Critic plus skip-observation GAE & Strict two-sided DIS mask & $A=0$ or every useful token masked \\
- L333 [claim] usually survives, so ZVF is the dominant certificate. Later GRPO epochs can
- L334 [claim] also suffer transport starvation and should log both quantities. In PPO and
- L335 [claim] SAO, the critic replaces the group baseline, but it does not eliminate zero
- L388 [claim] turn SAO into grouped sampling: every completed root is handled immediately,
- L436 [claim] For PPO, recompute GSR after each actor epoch and stop when either target KL is
- L439 [claim] For SAO, the analogous action is to reduce queue lag or refresh a trajectory;
- L444 [claim] ensemble disagreement indicates critic lag, up to a fixed $K_{\max}$. SAO's
- L446 [claim] \citep{hou2026sao}; \method makes the ratio conditional rather than universal.
- L467 [claim] artifacts; it contains no PPO or SAO run. The source contains 2,525 rows but
- L520 [claim] utility, not a claim that $G=4$ is globally optimal.
- L549 [claim] forgetting. This uncertainty motivates a causal matched-budget comparison of
- L553 [claim] \section{Preregistered Evaluation Contract for PPO and SAO}
- L576 [claim] For high-PAM/low-GSR roots, fresh-policy recollection should outperform wider
- L577 [claim] PPO/DIS clipping at matched generated tokens, measured by stability-adjusted
- L600 [claim] The GRPO/PPO isolation cell uses the same Qwen3-8B initialization, GSM8K split,
- L603 [claim] SWE-Bench Verified split, matching SAO's published scale where feasible
- L604 [claim] \citep{hou2026sao}. If resource constraints require a smaller model, that
- L627 [claim] PPO introduced the clipped policy surrogate used widely in RLHF
- L628 [claim] \citep{schulman2017ppo,ouyang2022training}; GAE provides its standard
- L639 [claim] token-level credit for single-rollout PPO \citep{che2026single}. These methods
- L645 [claim] SAO directly targets single-rollout asynchronous agentic RL and introduces
- L647 [claim] training, and skip-observation GAE \citep{hou2026sao}. Asynchronous actor-
- L657 [claim] First, EGM is not an improvement guarantee. It ignores score-vector geometry,
- L671 [claim] PPO/SAO generalization remains unverified until the evaluation contract is
- L678 [claim] GRPO flat groups, PPO clipping, and SAO double-sided masking look like separate
- L686 [claim] and motivate the asymmetry; matched-budget PPO and SAO experiments must decide
- L698 [claim] \subsection{PPO's sign-dependent gate}
- L700 [claim] For $A_t>0$, Eq.~\eqref{eq:ppo} equals the constant
- L704 [claim] $\rho_tA_t$ otherwise. This yields Eq.~\eqref{eq:ppo-gate}, ignoring the
- L749 [claim] \item PPO gates match Eq.~\eqref{eq:ppo-gate} for all four sign/boundary
- L751 [claim] \item SAO gates are zero on both sides of the DIS interval;
- L761 [claim] The repository's compact PPO implementation computes ratios, clipped surrogate
- L764 [claim] \path{platform_hybrid/experiments/implementations/cleanrl_ppo_math.py}. The
- L768 [claim] instrumentation on a small arithmetic PPO implementation, not evidence for
- L769 [claim] language-model PPO, SAO, or GLM-5.2. Production conclusions require the
- L787 [claim] marks causal held-out improvement, global $G=4$ optimality, and performance

## `zvf-program/audit/reproducibility_audit.tex`

Consumers: R08

- L164 [claim] Optional secondary arms (CPPO \cite{lin2025cppo}, NGRPO

## `zvf-program/position/min_report_rl.tex`

Consumers: R06

- L51 [claim] though an algorithm label---PPO, GRPO, DPO, and the growing GRPO family
- L60 [claim] an under-specification exhibit, not a backend causal estimate: the label was held
- L103 [claim] differ in their loss form (is there a PPO ratio? a clip? a completion-only
- L126 [claim] hyperparameter explains the gap, and the managed run later proved to use a
- L152 [claim] \emph{support expanded}. A variant can ``win'' at pass@1 by concentrating
- L256 [claim] variance-mitigation line (AERO, CPPO, NGRPO, Scaf-GRPO)
- L258 [claim] lin2025cppo,nan2025ngrpo,zhang2025scaffgrpo}---are typically defined as
- L281 [claim] \emph{Report:} whether the update uses a PPO-style importance ratio
- L342 [claim] A fixed first-five-step rule (\zvf{}$\,\ge 80\%$ with reward $\le 5\%$) is a
- L364 [claim] band that is a selection artifact, not a causal lift \cite{zvfaudit2026}. A
- L428 [claim] Three of the standard's items now carry \emph{quantified} first-party
- L469 [claim] tells the opposite, consistent story: all four arms improve by $+1.5$ to
- L473 [claim] and stated noise bands, this panel supports any narrative a reader
- L474 [claim] prefers.} With them, it supports exactly one claim: a small,
- L500 [claim] AERO/CPPO/NGRPO/Scaf-GRPO were each implemented as one hook on a shared GRPO
- L524 [claim] \caption{\textbf{Preregistered single-stack treatment matrix.} The first five
- L641 [claim] The first line of \texttt{min\_report\_rl.jsonl} is a JSON object whose
- L666 [claim] schedule first-class; OpenRLHF's reference-model and KL options map directly

## `zvf-program/registry/grpo_registry.tex`

Consumers: R07

- L76 [claim] variant---DAPO, GSPO, Dr.\grpo{}, M-\grpo{}, AERO, CPPO, NGRPO,
- L120 [claim] The position paper \minreport{} \cite{minreportrl2026} argues that the first
- L314 [claim] CPPO & PPO / \grpo{} & Clip-pruned objective: tokens with large ratios are
- L317 [claim] effective sample \cite{lin2025cppo}.\\
- L584 [claim] A registry record can therefore be generated automatically from the first line

## `zvf-program/theory/zvf_theory.tex`

Consumers: R05

- L137 [claim] dead-group fraction is a first-order quantity in critic-free RL. Our
- L141 [claim] quantity whose sampling distribution (T1), compute cost (T2), and optimal
- L234 [claim] \begin{assumption}[Non-degenerate support]\label{ass:nondegen}
- L343 [claim] policy-improvement guarantee.
- L355 [claim] size $K$ until the first informative group ($Z_g=0$) requires, in expectation,
- L358 [claim] \E[\text{rollouts to first informative group}] \;=\; \frac{K}{1-\ZVF}.
- L367 [claim] number of groups through the first informative group is geometric with mean
- L368 [claim] $1/\bar q$; multiplying by $K$ proves \eqref{eq:exp-rollouts}. If instead a
- L386 [claim] $(1-\delta)$-quantile of the rollouts-to-first-informative-group
- L390 [claim] informative group arrives within the first $K$ rollouts with
- L392 [claim] not make a gradient improbable; it makes the \emph{guarantee} of one
- L397 [claim] guaranteed monotone reward increment; the latter would require a
- L399 [claim] improvement guarantee. Treat the corollary as a wasted-\emph{gradient}
- L465 [claim] optima. T3 proves what this proxy implies and, importantly, what it cannot
- L478 [claim] and a reward-density prior $\phi$, a per-prompt \emph{candidate} optimal
- L484 [claim] the first-order stationarity condition $\partial J/\partial G = 0$, i.e.
- L512 [claim] any \emph{data-dependent} optimal-$G$ claim must come from a richer
- L534 [claim] by $p(1-p)/G$ and integrating over $\phi$ proves
- L543 [claim] the binomial argument above proves the discrete optimum directly.
- L594 [claim] policy until a richer objective and a prospective matched-budget test support it.
- L596 [claim] is not part of this controller because T3 proves that the proxy optimum is
- L598 [claim] policy with hysteresis; its closed-loop performance is tested prospectively in
- L608 [claim] operational relevance; they are not used to prove the mathematical statements.
- L644 [claim] $(1{-}\delta)$-quantile of rollouts-to-first-informative-group within
- L653 [claim] \paragraph{E-T3a: the optimal group size (reinterpreted).} The measured
- L656 [claim] algebraically guaranteed, not an empirical test}: by \eqref{eq:J23},
- L663 [claim] several-fold inefficient per rollout. A first training-side
- L674 [claim] hardest-first policy by $8.8\times$ --- difficulty and signal are not the
- L744 [claim] (eq.~\ref{eq:J23}) --- so T3 as stated supports no data-adaptive
