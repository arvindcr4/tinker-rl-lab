### Agentic AI: Post-Training Verifiable Agents
*Speaker:* Jiantao Jiao

#### Key claims / techniques
- Agentic models are "environment feedback aligned" and trained to maximize verifiable rewards (e.g., unit tests, proof checkers, DOM scripts) rather than only human preference, unlike earlier chat-optimized models.
- Training a capable agent requires three core ingredients: good verifiable training data, good evaluation data that defines intelligence, and good training recipes for feeding data to the model.
- Verifier quality is critical for difficult prompts: verifiers should minimize false positives/negatives and reward all valid answer forms unless a specific format is requested.
- Agent evaluations must be holistic, covering many tasks, harnesses, tools, vague instructions, and robustness; benchmark quality can be assessed along hardness, separability, and diversity.
- The training pipeline should first use SFT to imitate correct demonstrations and discourage meaningless attempts, then use RL to explore diverse successful trajectories and reinforce correctness.
- Good RL rests on three pillars: train longer with stable entropy/reward trade-offs, train on relevant but meaningfully difficult prompts, and sample more diverse high-quality responses per prompt.
- Interventions for longer stable training include reducing biased/off-policy updates, balancing update strength (e.g., DAPO-style clipping), and directly encouraging entropy (e.g., Skywork Open Reasoner 1).
- Difficulty-aware training should target prompts where model confidence correlates with reward and reward harder prompts more, rather than blindly adding very easy or extremely hard examples.
- Better response sampling can be achieved by scaling compute per answer (GenSelect), beam/search over reasoning (DeepConf), and confidence-thresholded majority voting to improve exploration quality.
- Future scale-up calls for a crowd-sourced collection of diverse environments, evaluations, and algorithms, analogous to how humans learn across many settings, teachers, and tasks.

#### Relevance hooks
- Directly relevant to RL post-training benchmarking and agent evaluation methodology: discusses holistic benchmark suites, harnesses, verifier quality, and benchmark dimensions (hardness, separability, diversity).
- Relevant to RL reproducibility standards: emphasizes stable training recipes, entropy control, avoiding biased updates, and the SFT-then-RL pipeline.
- None directly supported by the extracted text for GRPO/zero-variance diagnostics, group-size effects, or length bias.

#### Cited paper titles (verbatim only)
- None explicitly listed in extracted text.

Index row: f25 | AgenticAI_Jiantao_Jiao.pdf | Jiantao Jiao | Framework for post-training verifiable agents emphasizing environment/tool/verifier diversity, SFT-then-RL recipes, and stable RL via entropy control, harder prompts, and better sampling. | ok
