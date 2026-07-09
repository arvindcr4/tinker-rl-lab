### Building a Multimodal Knowledge Assistant
*Speaker:* Jerry Liu

#### Key claims / techniques
- A production knowledge assistant should accept any task (simple/complex questions, research) and return any output form (short answer, structured output, report).
- Naive RAG suffers from poor data processing, weak query understanding/planning, no function calling, and no memory, limiting time savings and decision-making value.
- A production-ready stack requires four capabilities: high-quality multimodal RAG, complex output generation, agentic reasoning over complex inputs, and a scalable full-stack application.
- Data quality is a prerequisite: “garbage in = garbage out”; ETL for LLMs entails parsing, chunking, and indexing into clean structured data.
- LlamaParse is presented as an LLM-native parser for complex documents (embedded tables, charts, images, irregular layouts), producing text chunks, tables, diagrams, and metadata without requiring per-element bounding boxes or exhaustive JSON.
- A true multimodal RAG pipeline parses documents into interleaved text and image chunks, links them via metadata, embeds/indexes text chunks, and feeds both text and retrieved images into a multimodal LLM during synthesis.
- Complex inputs (summarization, comparison, multi-part, research tasks) require agentic capabilities: tool use, query planning, memory, and reflection.
- There is a reliability–expressiveness trade-off: constrained flows (routers, fixed pipelines) are more reliable, while unconstrained agent/orchestrator flows are more expressive but less reliable.
- LlamaIndex Workflows are proposed as an event-driven, composable, code-first alternative to graph-based pipelines, intended to be more readable, maintainable, observable, and production-deployable.
- Multimodal report generation can be implemented via structured outputs (e.g., interleaving `TextBlock` and `ImageBlock`) with separate researcher and writer agent steps.
- Production agent systems require encapsulation/reusability, standardized communication interfaces, scalability, human-in-the-loop support, and debugging/observability tooling.
- `llama-deploy` is offered as a microservices architecture for agentic workflows, using a central message queue, distributed tool execution, and human-in-the-loop as a service.

#### Relevance hooks
- Maps to **agent evaluation methodology**: the lecture explicitly decomposes agentic systems into tool use, planning, memory, reflection, routing, and orchestration, and discusses reliability–expressiveness trade-offs that are central to agent benchmarks.
- Touches on **RL reproducibility standards** indirectly through the emphasis on debugging, observability, standardized interfaces, and human-in-the-loop guardrails for production agent deployments.

#### Cited paper titles (verbatim only)
- None explicitly listed in extracted text.

Index row: f24 | MKA.pdf | Jerry Liu | Production multimodal knowledge assistants require advanced parsing, multimodal RAG, structured agentic reasoning, and scalable event-driven orchestration. | ok
