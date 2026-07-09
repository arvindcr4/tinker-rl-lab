### Open-Source and Science in the Era of Foundation Models
*Speaker:* Percy Liang

#### Key claims / techniques
- Foundation model capabilities have risen sharply while access has narrowed from full paper/code/data/weights to API-only releases.
- Access level shapes the kind of science possible: API access is like cognitive-science behaviorism, open-weight access like neuroscience probing internals, and open-source access like full systems-level control.
- API-based agents can be composed of tools and verifiers to solve complex problems in ML engineering and cybersecurity, and can simulate social behavior (e.g., realistic interview simulations).
- Open-weight models enable reproducible research on interpretability, fine-tuning, distillation, and model merging, and findings such as adversarial attacks often transfer to API models.
- A permutation-based hypothesis test can assess whether two weight checkpoints were independently trained by permuting hidden units to build a null distribution of weight similarities.
- Empirical lineage checks link models such as Miqu-70B to Llama-2-70B, StripedHyena-Nous-7B to Mistral-7B-v0.1, and Llama-3.1-8B to Llama-3.2-3B.
- Open-source language modeling efforts mentioned include OLMo, OLMoE, RedPajama, DCLM-BASELINE, MAP-Neo, OpenCoder, FineWeb, and SmolLM.
- A workable open-source AI definition can require data information and processing code rather than raw copyrighted data, plus sufficient documentation and compute to retrain.
- Training techniques referenced in the open-source context include distributionally robust optimization, diagonal Hessian with clipping, and precise model editing.
- Strategies to scale open research include building downward-extrapolating scaling laws, pooling idle consumer GPUs for decentralized training, and public funding for shared infrastructure.

#### Relevance hooks
- Agent evaluation methodology: the lecture explicitly contrasts problem-solving agents (ML engineering, cybersecurity) with simulation agents (social-behavior digital twins) and discusses success-rate evaluation over repeated trials.
- RL reproducibility standards: argues that open-weight and open-source access are prerequisites for reproducible mechanistic research, model-derivation studies, and independent lineage verification.

#### Cited paper titles (verbatim only)
- None explicitly listed in extracted text.

Index row: f24 | percyliang.pdf | Percy Liang | Lecture maps foundation-model access tiers (API/open-weight/open-source) to the kinds of agent, mechanistic, and reproducibility research they enable. | ok
