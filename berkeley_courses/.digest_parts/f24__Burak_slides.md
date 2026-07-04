### LLM Agents — Enterprise Trends for Generative AI
*Speaker:* Burak Gokturk

#### Key claims / techniques
- Scale (compute, data, model size) has been the dominant driver of ML capability gains, from ImageNet and LibriSpeech to modern foundation models.
- Foundation models are built on transformers + autoregressive next-token prediction pre-trained on trillions of tokens.
- The standard path to useful agents is Supervised Fine-Tuning (SFT) followed by Reinforcement Learning from Human Feedback (RLHF) with a reward model.
- Gemini was designed as a natively multimodal model from the start; the project began February 2023 and had public releases in December 2023 (1.0) and February 2024 (1.5).
- Gemini 1.5 demonstrates long-context retrieval up to 10M tokens, with the claim that context-window information is "clearer" (less perturbed by gradient descent) and therefore reduces hallucination and enables in-context learning.
- A "needle in a haystack" evaluation across text, audio, and video reports >99.7% recall out to 10M tokens.
- Enterprise trends highlighted: AI development is accelerating; separate task-specific models are giving way to single generalizing models; dense models are moving toward efficient sparse models; single-modality models are moving toward multimodal models; API cost is approaching zero; search and LLMs are converging.
- Key production success factors are broad model choice, a managed production platform, ability to customize with proprietary data, and flexibility/avoidance of vendor lock-in.
- Customization toolkit includes fine-tuning, distillation (teacher/student with soft labels and temperature scaling), grounding, and function calling/extensions.
- Parameter-efficient adaptation methods covered include conventional fine-tuning, prompt tuning (Lester et al.), and LoRA, which decomposes weight updates into low-rank matrices and can be applied to attention layers.
- Grounding mitigates hallucinations and stale knowledge via Retrieve-Augment-Generate (RAG), private-document retrieval, fresh web content, and Natural Language Inference (NLI) based post-hoc corroboration with citations.
- Function calling gives developers structured output and external API/tool integration, enabling real-time data retrieval, database queries, and autonomous agent workflows.

#### Relevance hooks
- Agent evaluation methodology: the lecture explicitly discusses grounding, attribution scoring (supporting/contradicting sources), retrieval-augmented generation, and function calling as core components for building reliable enterprise agents.
- RL post-training benchmarking: only indirectly related—RLHF is mentioned as the standard post-SFT tuning stage, but no benchmarking methodology is discussed.
- None directly supported by the extracted text for GRPO/ZVF diagnostics, group-size effects, length bias, or evals-with-error-bars.

#### Cited paper titles (verbatim only)
- Gemini: A Family of Highly Capable Multimodal Models
- Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context

Index row: f24 | Burak_slides.pdf | Burak Gokturk | Enterprise generative-AI overview covering scaling laws, Gemini long-context multimodality, and production customization via RAG/grounding, distillation, and function calling. | ok
