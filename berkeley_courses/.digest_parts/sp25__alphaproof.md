### AlphaProof: When RL meets Formal Maths
*Speaker:* Thomas Hubert

#### Key claims / techniques
- AlphaProof applies AlphaZero-style search and reinforcement learning inside the Lean proof assistant: the action space is Lean tactics, states are Lean proof states, and the reward signal is exact formal verification of a finished proof.
- The training pipeline has four stages: (1) an auto-formalisation model that translates natural-language problems into Lean; (2) supervised pre-training on Mathlib (≈100k definitions, 200k theorems, 300k lines of proof) to learn a strong tactic prior; (3) AlphaZero RL on generated formal problems with Lean-based verification; and (4) test-time RL on variants of a target problem to produce a specialist checkpoint.
- The core bet is that formal mathematics is the right target for scalable RL because it provides both an in-silico exploration environment and a perfect, verifiable reward signal, even though the corpus is smaller than informal math.
- At IMO 2024 AlphaProof solved problems P1, P2 and P6 in algebra/number theory, while AlphaGeometry solved P4; the combined system reached a silver-medal score, missing the gold threshold by one point.
- For "determine the answer" IMO problems, AlphaProof ran in "hard mode": it generated O(100) candidate answers with Gemini, filtered out easily disprovable ones, and attempted to prove or disprove the rest using test-time RL.
- The system fully solved P6, described as one of the hardest IMO problems in the last ten years (only 5 of 609 human participants solved it fully), including a non-obvious construction praised by Timothy Gowers.
- The talk frames superhuman Alpha systems as arising from scaled-up trial-and-error, a grounded feedback signal, search, and curriculum; AlphaProof satisfies the first three, while curriculum generation for mathematics remains an open question.
- Key remaining challenges include Mathlib gaps (especially geometry and combinatorics), orders-of-magnitude more compute than human contestants, and the difficulty of creative theory building and "interestingness" in proofs.

#### Relevance hooks
- Formal theorem proving in Lean offers a perfect, verifiable reward signal, making it a strong setting for RL post-training and for benchmarking reasoning methods such as GRPO-style training on verified proofs.
- The IMO 2024 protocol (formalised problem inputs, human judging, partial scoring, and compute/time budgets) provides a concrete example of rigorous agent-evaluation methodology for mathematical reasoning.
- The test-time RL / specialist-training step, which generates variants of a hard target problem and re-trains, is a form of post-training adaptation whose transfer effectiveness can be benchmarked.

#### Cited paper titles (verbatim only)
- None explicitly listed in extracted text.

Index row: sp25 | alphaproof.pdf | Thomas Hubert | AlphaProof combines AlphaZero-style search/RL in Lean with auto-formalisation, Mathlib pretraining, and test-time RL to reach silver-medal level at IMO 2024 | ok
