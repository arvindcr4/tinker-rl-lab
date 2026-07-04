### Bridging Informal and Formal Mathematical Reasoning with AI
*Speaker:* Sean Welleck

#### Key claims / techniques
- Formal mathematics treats proofs as source code that compiles iff correct, enabling collaboration, trust, and instant feedback; informal math is flexible but hard to check.
- The informal-formal gap is a core bottleneck: informal intuitions require detailed specification and deep formal-system knowledge to express.
- Lean-STaR trains a 7B model to generate informal "thoughts" before each formal proof step via expert-iteration-style reinforcement learning, improving miniF2F test pass rates over direct tactic generation.
- Increasing search budget is more effective when the model interleaves thoughts with tactics, suggesting thoughts add useful computational capacity and diversify search.
- Draft, Sketch, Prove (DSP) is a three-stage pipeline: (1) LLM drafts an informal proof, (2) LLM translates it into a formal sketch, (3) a low-level prover (Sledgehammer) fills the gaps.
- LeanHammer brings Sledgehammer-style automation to Lean by combining a neural premise selector, external automated theorem provers (e.g., Zipperposition), and the Aesop tree-search tactic.
- LeanHammer's premise selector frames retrieval as contrastive learning over (state, positive premises, negative premises) using a transformer encoder; even sub-100M-parameter retrievers substantially raise proof rates on held-out Mathlib theorems.
- miniCTX benchmarks neural theorem provers on real Lean projects ("future Mathlib", PFR, PrimeNumberTheorem) with long, cross-file context, exposing a gap between competition-problem performance and research-level formalization.
- Premise selection and file-tuning on preceding in-file/cross-file context improve handling of realistic project dependencies; the resulting model is deployed in the LLMLean tool.
- Open-source artifacts are emphasized: models, data, extraction tools, and evaluation code are released to lower the accessibility gap for research-level formalization.

#### Relevance hooks
- Lean-STaR is an instance of RL post-training for reasoning: it uses expert iteration to bootstrap a thought-generating policy, with implications for how reward signals and search can improve theorem-proving agents.
- miniCTX advances agent evaluation methodology by testing provers on real, long-context formal-mathematics projects rather than self-contained competition benchmarks, making it relevant to robust eval design for reasoning agents.
- The talk's emphasis on releasing data, models, and evaluation pipelines connects to RL reproducibility standards for reasoning benchmarks.

#### Cited paper titles (verbatim only)
- Lean-STaR: Learning to Interleave Thinking and Proving
- STaR: Bootstrapping Reasoning with Reasoning
- Draft, Sketch, Prove: Guiding Formal Theorem Provers with Informal Proofs
- Premise Selection for a Lean Hammer
- miniCTX: Neural Theorem Proving with (Long-)Contexts

Index row: sp25 | welleck2025_berkeley_bridging.pdf | Sean Welleck | Bridging informal and formal math via RL-based thought generation (Lean-STaR), sketch-and-fill proving (DSP/LeanHammer), and real-project benchmarking (miniCTX). | ok
