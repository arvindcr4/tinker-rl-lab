**Segment scope**: platform_hybrid/paper/paper_P1_identifiability_note.tex (Standalone 2-page workshop note: "Failed Identifiability of GRPO Post-Training Scaling: A Short Negative Note on Cross-Scale Anchors")

# Potential Mistakes and Improvements

### [Major] Missing Formal Reference Citations in Standalone Note
- **Observation**: The note discusses several foundational models, benchmarks, and external empirical comparisons by name (e.g., DeepSeek-V3.1, Nemotron-3-Super-120B-A12B, Kimi-K2, GSM8K, and Chinchilla-style scaling laws), but contains no formal bibliography (`\bibliography`), `\bibitem` list, or citation commands (`\cite`).
- **Concern**: Although intended as a concise negative note (portfolio §3.3), publication venues and reviewers require formal bibliographic entries for attributed algorithms (GRPO, PPO), benchmark datasets (GSM8K), and frontier model technical reports (DeepSeek-V3, Nemotron-3 Super, Chinchilla). Without citations, readers cannot trace primary model architecture cards (such as the verification that Nemotron-3-Super-120B-A12B is a LatentMoE architecture).
- **Remediation**: Append a compact 4-to-6 entry bibliography citing:
  1. DeepSeek-Math / GRPO (Shao et al., 2024);
  2. PPO (Schulman et al., 2017);
  3. GSM8K (Cobbe et al., 2021);
  4. Chinchilla scaling laws (Hoffmann et al., 2022);
  5. NVIDIA Nemotron-3 Super technical report / model card establishing LatentMoE specifications.

### [Minor] Explicit Parameterization of the 12-Anchor Panel in Headline Table
- **Observation**: In Section 3 (Table~\ref{tab:neg}), the note reports two-sided permutation tests on the 12-anchor panel: $\lambda$--$N$ ($p=0.74$) and $R_\infty$--$N$ ($p=0.51$).
- **Concern**: The table notes "12-anchor panel, $10^4$ permutations", but does not explicitly state that the underlying step horizons vary from $T=3$ to $T=30$ across these 12 anchors (as recorded in `scaling_law_extended_frontier.tsv` and `scaling_law_iter109_lambdaN.tsv`). A reader might assume all 12 anchors underwent full-budget post-training trajectories.
- **Remediation**: Add a brief parenthetical or footnote in Table~\ref{tab:neg} clarifying that the 12 anchors comprise 5 long-horizon anchors ($T \in [20, 30]$) and 7 short-horizon probe checkpoints ($T \in [3, 5]$).

### [Minor] Specification of LatentMoE Routing Architecture Context
- **Observation**: In Section 3 and the Abstract, the note states that Nemotron-3-Super-120B-A12B is "LatentMoE: 120B total / 12B active".
- **Concern**: For readers unfamiliar with NVIDIA's LatentMoE architecture, the distinction between standard sparse MoE (top-$k$ routing over feed-forward experts) and LatentMoE (where routing occurs in a compressed latent space) is relevant to why it was initially mislabelled as dense.
- **Remediation**: Briefly mention that LatentMoE utilizes latent expert representations, which contributed to its initial erroneous assignment as a dense backbone in early exploratory scripts.

# Minor Corrections and Typos

1. **Table 1 Column Justification and Underfull Box Remediation**:
   - In Table~\ref{tab:neg}, line 67 originally specified `\begin{tabular}{@{}p{0.42\linewidth}p{0.5\linewidth}@{}}`. On short rows, full justification created an underfull horizontal box warning (`Underfull \hbox (badness 1173) in paragraph at lines 78--80`).
   - *Status*: Remediated by adding `\usepackage{array}` and formatting the outcome column as `>{\raggedright\arraybackslash}p{0.54\linewidth}`. Compilation now completes cleanly with 0 warnings.

2. **Empirical Verification of Retracted Numbers**:
   - The headline statistics reported in Table~\ref{tab:neg} were independently verified against repository data:
     - Nemotron-120B collapse: peak $0.875$ at step 3, last-10 mean $0.1625$, zero-reward fraction exact $0.55$ ($11/20$ steps), and $5/20$ steps $>0.25$ verified against `platform_hybrid/experiments/tinker-runs/results/frontier_gsm8k_nemotron-120b.json`.
     - Permutation tests $p=0.74$ and $p=0.51$ verified against `platform_hybrid/experiments/results/scaling_law_iter109_lambdaN.tsv`.
     - Withdrawn MoE-vs-dense gap $+0.338$ ($p=0.023$) and corrected gap $+0.188$ ($p \approx 0.178$) verified against `scaling_law_extended_frontier.tsv`.
