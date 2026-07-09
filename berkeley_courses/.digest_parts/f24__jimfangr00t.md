### A Tale of Two Kittens
*Speaker:* Dr. Jim Fan, NVIDIA Research

#### Key claims / techniques
- Frames embodied AI with the 1963 Held & Hein kitten experiment contrasting passive and active experience, updated with a ChatGPT → Embodied AI narrative.
- Argues the timing for humanoid robotics is driven by falling cost/time curves: examples place NASA Robonaut ($1.5M, 2001) → Unitree G1 ($30K, 2024), alongside Tesla Optimus, Boston Dynamics e-Atlas, and Figure F.02.
- Introduces Project GR00T ("Generalist Robot 0 0 Technology") as an AI brain for humanoid robots.
- Proposes three design principles: Data Pyramid, Foundation Agent, and "The Matrix".
- Data Pyramid layers: Internet Data (EB/day) → Simulation Data (TB/GPU-day) → Real Robot Data (24 hours/robot-day), supported by Omniverse Cloud teleoperation and Isaac Lab.
- Core simulation claim: "It's easier to simulate a problem than to solve it"; training robots in large-scale simulation is viable.
- Describes a generative simulation pipeline: Text-to-3D models (Stable Diffusion), USD generation (ChatUSD), and task/code generation with GPT-4o/Claude-3.5, culminating in the RoboCasa framework.
- Demo amplification chain: 1 human demo → N synthetic demos (RoboCasa) → N×M demos (MimicGen), including bimanual strategies (parallel, coordination, sequential) and DexMimicGen for humanoid tasks.
- Cross-embodiment foundation policy: MetaMorph tokenizes a robot's kinematic tree as a graph of joints and trains a single Transformer from observation to action across varied terrains and robot forms.
- Eureka: a coding-LLM approach that writes reward functions from task descriptions and environment code, then runs massively parallel GPU training with reward candidates, automated feedback, and self-reflection.
- DrEureka extends Eureka by using an LLM to generate sim2real domain-randomization configurations; Eureka++ further generalizes the loop to new tasks and new simulations.
- GR00T stack: OVX for synthetic data/token generation, DGX for foundation model training, AGX for edge deployment, plus HOVER and digital-twin evaluation, orchestrated by OSMO.

#### Relevance hooks
- None directly supported by the extracted text.

#### Cited paper titles (verbatim only)
- RoboCasa: Generative Simulation Framework

Index row: f24 | jimfangr00t.pdf | Dr. Jim Fan, NVIDIA Research | Humanoid foundation agents require a data pyramid (internet→sim→real), generative simulation, cross-embodiment policies, and LLM-designed rewards/domain randomization for sim2real. | ok
