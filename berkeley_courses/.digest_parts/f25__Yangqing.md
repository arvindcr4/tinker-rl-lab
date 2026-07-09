### Evolution of System Designs from an AI Engineer Perspective
*Speaker:* Yangqing Jia

#### Key claims / techniques
- The lecture frames recent LLM progress as a sequence of architectural waves analogous to computer-vision history: GPT as a structural innovation like AlexNet, MoE like ensemble learning, test-time scaling like Inception/ResNet/Fully convolutional networks, and reinforcement learning like GANs/multi-instance learning.
- New algorithms are still driving continued model improvement and growing consumption, with usage expanding beyond training to inference and application traffic (citing openrouter.ai).
- Consumer-facing AI apps remain highly fluid because foundation models keep improving, and prosumers’ willingness to pay is currently the dominant revenue driver.
- Enterprise AI applications are described as hopeful but still nascent; enterprises are adopting AI faster than historical enterprise-software cycles.
- AI infrastructure is positioned as the third pillar of enterprise IT strategy, following scientific computing, virtual private servers, web-service clouds, and data clouds.
- AI compute differs fundamentally from conventional cloud and data workloads: compute dominates IO, runs arbitrary user code, and requires tightly coupled distributed systems, weakening the traditional cloud value propositions of workload variety, hardware flexibility, and VM interchangeability.
- Running AI workloads on bare metal or Kubernetes as-is is considered the wrong default; instead, organizations should adopt AI-native platforms that unify development, training, and inference.
- Recommended infrastructure practices include multi-cloud supply-chain management, elasticity and utilization management, building an AI-native platform, and organizing teams around both models and applications.
- GPU hardware failures occur more frequently than many developers expect, so infrastructure design must optimize both developer efficiency and hardware efficiency.

#### Relevance hooks
- None directly supported by the extracted text.

#### Cited paper titles (verbatim only)
- The Chinese Typewriter: a History
- The Bitter Lesson

Index row: f25 | Yangqing.pdf | Yangqing Jia | Lecture traces the LLM-stack evolution (algorithms, apps, AI cloud infra) and argues AI compute needs a new systems paradigm | ok
