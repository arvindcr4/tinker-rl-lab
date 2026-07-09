# Berkeley Agents Curriculum → TinkerRL-Bench Improvement Mining

## Mission
You are mining the three Berkeley RDI agents courses for CONCRETE, IMPLEMENTABLE
improvements to (A) the TinkerRL-Bench research/benchmark (the 4 pillar papers) and
(B) this repo's autonomous-research pipeline (`minimax_autoresearch/`). Every
iteration must convert course material into a verifiable artifact in this worktree —
a proposal doc WITH a working prototype/analysis, never a reading summary alone.

Course pages (public, re-scrape if needed):
- F24  "LLM Agents"            https://rdi.berkeley.edu/llm-agents/f24
- SP25 "Advanced LLM Agents"   https://rdi.berkeley.edu/adv-llm-agents/sp25
- F25  "Agentic AI"            https://rdi.berkeley.edu/agentic-ai/f25

## Improvement targets (map course ideas onto THESE)
A1. Statistical rigor of the benchmark: error bars / CIs on every headline number
    (Sida Wang F25 "Predictable Noise in LLM" + "Adding Error Bars to Evals").
    Data: experiments/results/*.tsv, reward traces.
A2. Evaluation methodology: agent-eval best practices ("Survey on Evaluation of
    LLM-based Agents" F25; WorkArena/WebArena/OSWorld/τ2-Bench design lessons) →
    stronger eval protocol sections + diagnostics in the 4 papers.
A3. Post-training science: DPO/PPO/GRPO connections (Jason Weston SP25; Tulu 3;
    "Unpacking DPO and PPO"; verifiable rewards — Jiantao Jiao F25 "Post-Training
    Verifiable Agents") → sharpen P1 scaling, P3 group-size≈DPO, ZVF-vs-verifiable-
    reward framing.
A4. Tool-use / agentic RL: ReAct, SWE-agent/OpenHands interfaces, function calling
    (F24; SP25 agentic workflow) → improve the tool_use environment + its 0%-reward
    failure analysis.
A5. Inference-time reasoning: CoT-without-prompting, self-correction limits,
    LLM-as-optimizer, search/planning (F24 Denny Zhou; SP25 Xinyun Chen) →
    inference-time baselines vs RL post-training claims.
B1. Orchestrator improvements: state-driven workflows (StateFlow), compound-system
    prompt optimization (DSPy, F24 Omar Khattab), memory (HippoRAG SP25),
    multi-agent designs (AutoGen F24; Noam Brown / Oriol Vinyals F25),
    Paper2Agent / Virtual Lab (James Zou F25) → write PATCH PROPOSALS as files under
    minimax_autoresearch_improvements/ (never modify the live orchestrator).
B2. Agent safety/security: prompt injection (DataSentinel), memory poisoning
    (AgentPoison), privilege control (Progent, SP25 Dawn Song; F25 Dec 8) → audit
    the orchestrator guardrail, add test cases, propose hardening.

## Syllabus digest (lecture → key readings worth pulling from arXiv)
### F24 LLM Agents (CS294/194-196)
1. LLM Reasoning (Denny Zhou, GDM) — CoT Without Prompting; LLMs Cannot Self-Correct
   Reasoning Yet; Premise Order Matters; CoT Empowers Transformers on Serial Problems.
2. Agent history & overview (Shunyu Yao, OpenAI) — WebShop; ReAct.
3. Agentic frameworks (Chi Wang; Jerry Liu) — AutoGen; StateFlow.
4. Enterprise GenAI (Burak Gokturk, Google) — RAG grounding; Needle-in-Haystack evals.
5. Compound AI & DSPy (Omar Khattab) — MIPRO (Optimizing Instructions & Demos for
   Multi-Stage LM Programs); Fine-Tuning + Prompt Optimization Together.
6. SW-dev agents (Graham Neubig, CMU) — SWE-agent (agent-computer interfaces); OpenHands.
7. Enterprise workflow agents (Nicolas Chapados) — WorkArena(+); TapeAgents.
8. Neural+symbolic decision making (Yuandong Tian) — Beyond A*; Dualformer; SurCo.
9. Robotics GR00T (Jim Fan) — Voyager; Eureka (LLM reward design!); DrEureka.
10. Open science (Percy Liang) — Cybench.
11. Measuring agent capabilities & RSP (Ben Mann, Anthropic) — RSP; computer use.
12. Safe & trustworthy agents (Dawn Song) — DecodingTrust; Representation Engineering;
    training-data extraction.

### SP25 Advanced LLM Agents (CS294/194-280)
1. Inference-time reasoning (Xinyun Chen) — LLMs as Optimizers; Self-Debug; no-self-correct.
2. Learning to reason (Jason Weston, Meta) — DPO; Iterative RPO; Chain-of-Verification.
3. Reasoning/memory/planning of agents (Yu Su) — Grokked Transformers; HippoRAG;
   LLM-as-world-model for web agents.
4. Open training recipes (Hanna Hajishirzi, UW) — Tulu 3; Unpacking DPO & PPO; OpenScholar.
5. Coding agents & vuln detection (Charles Sutton, GDM) — interactive tools for security;
   Naptime→Big Sleep.
6. Multimodal autonomous agents (Salakhutdinov) — Mind2Web; WebArena; VisualWebArena;
   Tree Search for LM Agents.
7. Multimodal perception→action (Caiming Xiong) — OSWorld; AGUVIS.
8. AlphaProof: RL meets formal math (Thomas Hubert, GDM) — IMO silver; AlphaZero.
9. Autoformalization & theorem proving (Kaiyu Yang) — LeanDojo; Autoformalization.
10. Advanced theorem proving (Sean Welleck) — Draft-Sketch-Prove; miniCTX; Lean-STaR; ImProver.
11. Abstraction & discovery (Swarat Chaudhuri) — in-context theorem-proving agent;
    symbolic regression w/ concept library.
12. Safe & secure agentic AI (Dawn Song) — DataSentinel; AgentPoison; Progent.

### F25 Agentic AI (CS294/194-196)
1. Intro (Dawn Song). 2. LLM agents overview (Yann Dubois, OpenAI) — Kimi K2; DeepSeek-V3.
3. AI-engineer system design (Yangqing Jia, NVIDIA).
4. Post-training verifiable agents (Jiantao Jiao, NVIDIA) — SWE-bench Verified; BrowseComp.
5. Agent evaluation (course staff) — Survey on Evaluation of LLM-based Agents.
6. Lessons training agentic models (Weizhu Chen, Microsoft).
7. Multi-agent AI (Noam Brown, OpenAI).
8. Predictable noise in LLMs (Sida Wang, Meta) — Adding Error Bars to Evals.
9. AI agents for scientific discovery (James Zou, Stanford) — Virtual Lab (nanobodies);
   Paper2Agent.
10. Deploying real-world agents (Clay Bavor, Sierra) — τ2-Bench; Voice Sims.
11. Multi-agent systems in the LLM era (Oriol Vinyals, GDM).
12. Embodied autonomous agents (Peter Stone) — GT Sophy; SLAC.
13. Agentic AI safety & security (Dawn Song).

## Deliverable conventions (every iteration)
- Proposal docs → docs/berkeley_improvements/<NN>_<slug>.md  (≤300 lines each):
  the course idea, the verified citation(s), the mapping to a target (A1–B2), the
  prototype you built, measured result, and a go/no-go recommendation.
- Prototypes/analyses → scripts/berkeley/<slug>.py, run on REAL repo data,
  outputs under experiments/results/berkeley/.
- Orchestrator patch proposals → minimax_autoresearch_improvements/<slug>.md
  (+ optional .patch). Do NOT edit minimax_autoresearch/ itself.
- Ranked ledger → BERKELEY_IMPROVEMENTS.md at worktree root: one table, every
  improvement idea ever logged, columns: id | source lecture | target | status
  (proposed/prototyped/validated/rejected) | evidence path. Keep it current.
- Findings → append to ./AUTORESEARCH_FINDINGS.jsonl:
  {"ts","pillar","claim","evidence_path","citation_ok"} with pillar "B-F24"|"B-SP25"|
  "B-F25"|"B-SYNTH".
- Citations: verify EVERY paper (title/authors/year/venue) via the arXiv MCP tools
  before citing. Zero fabrication.

## Research prompt pack (added 2026-07-04)
`./research_prompts/` holds 10 research prompts with output contracts + failure modes
(hypothesis-stress-test, ablation-gap-finder, claim-evidence-linter, minimal-decisive-
experiment, rebuttal-strategy-builder, ...). When VERIFYING a proposal or paper claim,
run the matching prompt's contract instead of ad-hoc review — especially
design/hypothesis-stress-test.md before promoting a ledger row to `validated`.
