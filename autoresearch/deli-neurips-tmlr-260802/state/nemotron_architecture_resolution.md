# Nemotron-120B architecture resolution — P1 MoE-vs-dense split

Log tag: `w9-nemotron` · resolved 2026-08-02 · read-only investigation, no repo code or `.tex` modified.

## Verdict: **MoE — the paper's classification is wrong**

`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` (the exact HF repo ID recorded in
`platform_hybrid/experiments/tinker-runs/results/frontier_gsm8k_nemotron-120b.json`) is a sparse
Mixture-of-Experts model: 120.6 B total parameters, 12.7 B active per token, 512 experts per MoE
layer with top-22 routing, interleaved with Mamba-2 blocks and sparse attention anchors. The `A12B`
suffix is exactly the standard active-parameter naming convention, used here as intended. There is
no ambiguity in the primary sources — NVIDIA calls it MoE in the model card, the technical report
title, the arXiv abstract, and the launch blog.

Classifying it as `dense` in the 6-vs-6 split is a straightforward labeling error, and it is the
single anchor carrying the headline result: reclassifying it destroys the +0.338 / p=0.023 finding.

## Primary sources (verbatim)

**1. HuggingFace model card — the exact checkpoint used in the run.**
<https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16>
(raw: `/raw/main/README.md`)

> | **Total Parameters** | 120B (12B active) |
> | **Architecture** | LatentMoE - Mamba-2 + MoE + Attention hybrid with Multi-Token Prediction (MTP) |

> The model employs a hybrid **Latent Mixture-of-Experts (LatentMoE)** architecture, utilizing
> interleaved Mamba-2 and MoE layers, along with select Attention layers. […] The model has **12B
> active parameters** and **120B parameters in total**.

> ## Model Architecture
> - **Architecture Type:** Mamba2-Transformer Hybrid Latent Mixture of Experts (LatentMoE) with Multi-Token Prediction (MTP)
> - **Network Architecture:** Nemotron Hybrid LatentMoE

**2. NVIDIA technical report, "Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid
Mamba-Transformer Model for Agentic Reasoning".**
<https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf>
(arXiv HTML: <https://arxiv.org/html/2604.12374>)

> Abstract. We describe the pre-training, post-training, and quantization of Nemotron 3 Super, a 120
> billion (active 12 billion) parameter hybrid Mamba-Attention Mixture-of-Experts model.

> Nemotron 3 Super 120B-A12B Base scales up the hybrid Mamba-Attention Mixture-of-Experts (MoE)
> architecture introduced in Nemotron-3 Nano. We extend this foundation to 120.6B total parameters,
> maintaining a constrained active budget of 12.7B parameters (12.1B excluding embeddings) per
> forward pass.

> Sparse scaling further improves efficiency. Each MoE layer activates only a subset of experts per
> token (top-22 routing), enabling the model to scale to 120.6B total parameters while maintaining a
> 12.7B active parameter budget per forward pass.

> The architecture employs a hybrid Mamba-MoE design, featuring Mixture-of-Experts (MoE) layers with
> 512 total experts and a top-22 routing mechanism (k = 22).

Architecture table (also mirrored at <https://docs.nvidia.com/nemotron/nightly/nemotron/super3/pretrain.html>):
Total Layers 88 · Total Experts per Layer 512 · Top-k (Activated Experts) 22 · MoE Latent Size 1024
· Shared Expert Intermediate Size 5376.

**3. NVIDIA research announcement.** <https://research.nvidia.com/labs/nemotron/Nemotron-3-Super/>

> We are releasing NVIDIA Nemotron 3 Super, a 12B active 120B total parameter Mixture-of-Experts
> hybrid Mamba-Transformer model.

**4. NVIDIA developer blog (launch, 2026-03-11).**
<https://developer.nvidia.com/blog/introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning/>

> Super addresses the "thinking tax" with its hybrid mixture-of-experts (MoE) architecture.
> […] MoE layers scale effective parameter count without the cost of dense computation. Only a
> subset of experts activates per token […]

**5. NVIDIA NIM / build.nvidia.com model card** (same table as HF):
<https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard> — "Architecture | LatentMoE -
Mamba-2 + MoE + Attention hybrid with Multi-Token Prediction (MTP)".

Note: the only sense in which Nemotron 3 Super is not a "plain" MoE is that it is a *LatentMoE*
hybrid (experts computed in a compressed 1024-dim latent space, interleaved with Mamba-2 and a few
global attention anchors). That makes it *more* architecturally exotic than the other MoE anchors,
not less sparse. It does not license a `dense` label under any reading.

## Per-anchor classification audit (all 12)

Anchor definitions come from `platform_modal/scripts/scaling_law_extended.py` (`EXTENDED_MODELS`);
HF repo IDs come from the `model` field of each trace JSON in
`platform_hybrid/experiments/tinker-runs/results/`.

| # | Paper label | HF repo ID in trace JSON | Paper `arch` | Ground truth | Total / active | Correct? |
|---|---|---|---|---|---|---|
| 1 | Qwen3.5-4B | `Qwen/Qwen3.5-4B` | dense | **dense** | 4 B / 4 B (dense FFN, ffn_dim 9216; no expert block) | ✅ |
| 2 | Qwen3-8B | `Qwen/Qwen3-8B` | dense | **dense** | 8.2 B / 8.2 B | ✅ |
| 3 | Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | dense | **dense** | 8 B / 8 B | ✅ |
| 4 | Qwen3-32B | `Qwen/Qwen3-32B` | dense | **dense** | 32.8 B / 32.8 B | ✅ |
| 5 | Qwen3.5-27B | `Qwen/Qwen3.5-27B` | dense | **dense** | 27 B / 27 B (Gated DeltaNet + dense FFN, ffn_dim 17408) | ✅ |
| 6 | gpt-oss-20B | `openai/gpt-oss-20b` | moe | **MoE** | 20.9 B / 3.61 B, 32 experts, top-4 | ✅ |
| 7 | Qwen3-30B-MoE | `Qwen/Qwen3-30B-A3B` | moe | **MoE** | 30.5 B / 3.3 B, 128 experts, top-8 | ✅ |
| 8 | Qwen3-30B-MoE-Inst | `Qwen/Qwen3-30B-A3B-Instruct-2507` | moe | **MoE** | 30.5 B / 3.3 B | ✅ |
| 9 | DeepSeek-V3.1 | `deepseek-ai/DeepSeek-V3.1` | moe | **MoE** | 671 B / 37 B | ✅ |
| 10 | **Nemotron-120B** | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | **dense** | **MoE** | 120.6 B / 12.7 B, 512 experts, top-22 (LatentMoE + Mamba-2) | ❌ **WRONG** |
| 11 | Qwen3-235B-MoE | `Qwen/Qwen3-235B-A22B-Instruct-2507` | moe | **MoE** | 235 B / 22 B | ✅ |
| 12 | Kimi-K2-Thinking | `moonshotai/Kimi-K2-Thinking` | moe | **MoE** | 1 T / 32 B, 384 experts, top-8 + 1 shared | ✅ |

Naming-convention scrutiny applied to every row: the only `<total>B-A<active>B` repo IDs in the
panel are #7, #8, #10, #11. Three of the four are labeled `moe`; #10 is the sole violation. The
three MoE anchors *without* an A-suffix (#6, #9, #12) are all labeled correctly, so the error is not
systematic — it is one bad row.

Sources for the non-Nemotron rows:
- Qwen3.5-27B / Qwen3.5-4B are dense: <https://huggingface.co/Qwen/Qwen3.5-27B>,
  <https://huggingface.co/Qwen/Qwen3.5-4B> (model overview lists a Feed-Forward Network block with a
  single intermediate dimension and no expert count), and the Transformers docs, which state:
  *"This page covers the dense Qwen3.5 and Qwen3.6 variants (Qwen/Qwen3.5-9B, Qwen/Qwen3.5-27B,
  Qwen/Qwen3.6-27B). […] For the sparse mixture-of-experts variants see Qwen3.5 MoE."*
  (<https://huggingface.co/docs/transformers/en/model_doc/qwen3_5>). The Qwen3.5 MoE members are the
  `-A*B` checkpoints (35B-A3B, 122B-A10B, 397B-A17B); none is in this panel.
- gpt-oss-20b is MoE: OpenAI model card — *"The gpt-oss models are autoregressive Mixture-of-Experts
  (MoE) transformers […] gpt-oss-20b with 24 layers (20.9B total and 3.6B active parameters)"*
  (<https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf>).
- Kimi-K2-Thinking is MoE: model card table — *"Architecture | Mixture-of-Experts (MoE) | Total
  Parameters | 1T | Activated Parameters | 32B"* (<https://huggingface.co/moonshotai/Kimi-K2-Thinking>).

### Two secondary defects found in the same code path (not blocking, but fix together)

1. `platform_modal/scripts/scaling_law_extended.py` docstring, line ~12, says the panel includes
   *"the non-MoE frontier (kimi-k2)"*. Kimi-K2-Thinking is a 1 T / 32 B-active MoE. The `arch` field
   in `EXTENDED_MODELS` is correct (`moe`); only the prose is wrong. It is evidence that the
   architecture labels were assigned from recollection rather than from the model cards — the same
   process that produced the Nemotron error.
2. `params_B` for DeepSeek-V3.1 is `685.0` in `scaling_law_extended_frontier.tsv` but `671` in the
   iter-105 anchor table (`platform_hybrid/paper/sections/scaling_law_iter105.tex`). Both are
   defensible (HF sidebar total incl. MTP module vs. reported total) but they should not disagree
   inside one paper. Does not affect the arch split.

## Recomputed statistics

Recomputed from raw `reward_trace` arrays in
`platform_hybrid/experiments/tinker-runs/results/{scale,arch,moe,frontier}_gsm8k_*.json` with
`python3`; $\bar R$ = mean of the full reward trace, matching `scaling_law_extended.py`. Permutation
p-values are **exact** (enumerating all $\binom{12}{k}$ splits) rather than 5 000-replicate Monte
Carlo — the exact values reproduce the published MC numbers to 3 decimals, confirming the pipeline
is faithful and the only defect is the label.

| Scenario | MoE $n$ | MoE mean | dense $n$ | dense mean | gap | one-sided $p$ | two-sided $p$ |
|---|---|---|---|---|---|---|---|
| **As published** (Nemotron = dense) | 6 | 0.8098 | 6 | 0.4721 | **+0.3376** | **0.0238** | 0.0476 |
| **Corrected** (Nemotron = MoE) | 7 | 0.7191 | 5 | 0.5316 | **+0.1875** | **0.1780** | 0.3725 |
| Sensitivity: Nemotron dropped | 6 | 0.8098 | 5 | 0.5316 | +0.2782 | 0.0476 | 0.0931 |

(Published values for cross-check: gap $+0.3376$, MC one-sided $p=0.0230$, two-sided $p=0.0460$ in
`platform_hybrid/experiments/results/scaling_law_moe_vs_dense.tsv` — reproduced exactly.)

**The headline comparison does not survive.** Under the corrected split the gap falls by 44 % to
$+0.19$ and the one-sided permutation $p$ rises from 0.024 to **0.178** — a factor of 7.5, nowhere
near any conventional threshold. The result was entirely load-bearing on the single misclassified
anchor: Nemotron has the lowest $\bar R$ in the whole panel (0.175, a collapse run), so mislabeling
it as dense simultaneously deflated the dense mean and inflated the MoE mean relative to the truth.

Even the "drop Nemotron" sensitivity ($p = 0.0476$ one-sided, $p = 0.093$ two-sided) is a
knife-edge result on $n=11$ and should not be reported as a positive finding.

Note on the paper's existing caveat: `P01.tex` already says *"After collapsing Nemotron the gap
shrinks to $+0.20$ and is no longer significant at $p < 0.05$."* That $+0.20$ does not correspond to
dropping Nemotron ($+0.278$) and is not reproducible from the traces under any documented rule;
the closest match is replacing Nemotron's $\bar R$ with its peak ($+0.221$). The *correct*
reclassification gap is $+0.1875$, i.e. the caveat sentence accidentally lands near the right number
for the wrong reason. The caveat therefore cannot be used as a defense — it attributes the fragility
to one outlier's collapse, when the actual cause is that the anchor is on the wrong side of the
partition.

### Additional threat noticed while recomputing (out of scope, worth flagging)

$\bar R$ is averaged over wildly unequal trace lengths: Qwen3-32B ($n=3$), Qwen3.5-27B ($n=3$),
Qwen3-30B-MoE-Inst ($n=3$), Qwen3-235B-MoE ($n=4$), Qwen3-30B-MoE ($n=5$) vs. six anchors at
$n=20$–$30$. Two of the three anchors at the MoE ceiling ($\bar R = 1.000$) are 3–4-step probes.
A 6-vs-6 permutation test over $\bar R$ treats a 3-step probe and a 30-step run as equally
informative. Even with the arch labels fixed, this comparison is not defensible as evidence about
architecture.

## Recommendation

**Drop the comparison.** Reclassifying Nemotron as MoE takes the result to $+0.19$ at $p = 0.178$,
which is not a finding; the fallback framings (drop-the-outlier at $p = 0.048$ on $n=11$, or the
$\bar R$-over-unequal-trace-lengths construction) are too fragile to restate as a headline. Remove
the "headline *positive* result of the extension is on the architecture axis" claim, the MoE-vs-dense
table (`tab:scaling-moe-vs-dense`), and panel (c) of `fig:scaling-extended`; if any architecture
discussion is retained, report it as a null with the corrected 7-vs-5 numbers above.

Files that must change if the paper is revised (all currently assert the wrong split — **not touched
by this investigation**):
- `platform_modal/scripts/scaling_law_extended.py` — `EXTENDED_MODELS["Nemotron-120B"]["arch"]`, plus
  the stale "non-MoE frontier (kimi-k2)" docstring line
- `platform_hybrid/experiments/results/scaling_law_moe_vs_dense.tsv`,
  `scaling_law_extended_frontier.tsv` (regenerate)
- `platform_hybrid/paper/sections/scaling_law_iter105.tex` (Table `tab:iter105-modes`, Nemotron row =
  `dense`), `scaling_law_iter33.tex` ("6 are dense and 6 are MoE"; the "no MoE model collapses"
  claim is **falsified** by the correction — Nemotron is the collapse anchor and it is MoE)
- `platform_hybrid/experiments/results/scaling_law_iter109b_family.tsv` and any other
  family-stratified fit that partitions on the same labels
- P1 / U01 paper sources carrying `tab:scaling-moe-vs-dense` and `fig:scaling-extended`

⚠️ Highest-severity downstream consequence: `scaling_law_iter33.tex` argues that *"no MoE model
collapses"* and links it to "the broader pillar 3 observation that MoE models are less [brittle]".
Under the corrected labels the panel's only collapse anchor **is** an MoE model, so that qualitative
claim inverts. This propagates further than the +0.338 table and should be checked wherever pillar-3
brittleness claims appear.
