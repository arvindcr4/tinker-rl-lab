### Measuring Agent Capabilities and Anthropic’s RSP
*Speaker:* Ben Mann

#### Key claims / techniques
- Anthropic’s Responsible Scaling Policy (RSP) is a public commitment to ensure model capability does not outstrip the ability to build effective guardrails and mitigate harm.
- The RSP is organized around AI Safety Levels (ASL-1 through ASL-4), with Anthropic currently preparing for ASL-3, the “significantly higher risk” tier.
- RSP goals include structuring hard safety decisions, public accountability, learning to iterate on safe decisions, and providing a template for policymakers and industry.
- Standard capability benchmarks are described as not lasting; performance relative to human baselines saturates quickly (Kiela et al., 2023).
- Anthropic emphasizes measuring task-completion time relative to humans, citing a METR August 2024 evaluation finding that Claude 3.5 Sonnet completes work that would take human developers ~30 minutes in seconds.
- Claude 3.5 Sonnet is claimed to outperform OpenAI o1-preview on SWE-bench Verified while being cheaper and faster.
- Similar claims are made on Aider’s code-editing and refactoring benchmarks, including an 18% lead on the refactoring benchmark, which tests producing long code chunks without skipping sections or making mistakes.
- A “computer use” case study is presented as a concrete agentic capability, covering demos, technical implementation, safety considerations, and future implications.
- Practical safety measures include ASL-standard implementation, security measures, deployment safeguards, and year-1 lessons.
- Future priorities include scaling governance, capability measurement, and academic collaboration.

#### Relevance hooks
- Agent evaluation methodology: the lecture discusses SWE-bench Verified, Aider benchmarks, METR task-time evaluation, and the challenge that benchmarks saturate relative to human performance.
- RL reproducibility standards: the RSP framing ties measurement rigor to accountable deployment decisions, though it does not discuss RL training reproducibility per se.
- None directly supported by the extracted text for GRPO/RL post-training benchmarking, ZVF/zero-variance diagnostics, group-size effects, length bias, or evals-with-error-bars.

#### Cited paper titles (verbatim only)
- Mathematical Framework for Transformer Circuits
- Toy Models of Superposition
- Sleeper Agents: Deceptive LLMs

Index row: f24 | antrsp.pdf | Ben Mann | Anthropic’s RSP uses ASL tiers and agentic benchmarks (SWE-bench, Aider, METR) to measure capabilities and guide safety decisions for increasingly autonomous models. | ok
