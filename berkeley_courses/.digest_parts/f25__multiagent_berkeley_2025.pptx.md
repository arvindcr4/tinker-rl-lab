### Multi-Agent AI
*Speaker:* Noam Brown

#### Key claims / techniques
- Self-play is framed as the third step of an AlphaGo-style recipe: pretrain on human data, scale inference compute, then recursive self-improvement through self-play.
- In two-player zero-sum games, sound self-play provably converges to a minimax equilibrium given sufficient memory and compute; exploitability measures expected loss to a best response.
- Neural-net approximations of minimax remain vulnerable to adversarial exploitation because finding an exploit is easier than defending against one, especially in imperfect-information games.
- Fictitious Play, Regret Matching, and Hedge regularize best-response dynamics; last-iterate algorithms such as Regularized Nash Dynamics and Magnetic Mirror Descent now empirically converge to minimax while performing better in single-agent RL settings.
- In a two-player zero-sum minimax equilibrium, "cheap talk" communication is theoretically useless because any message that helps one player can be ignored by the opponent.
- The central claim for general-sum / cooperative settings is that learning to cooperate with humans without human data is a dead end; population best-response notions require data on the population of human players.
- DORA learned no-press Diplomacy from scratch via AlphaZero-style self-play, achieving an 86.5% ± 6.1% win rate against human experts in 2-player no-press Diplomacy.
- Diplodocus placed first in a 200-game real human no-press Diplomacy tournament; CICERO placed in the top 10% of an online natural-language Diplomacy league and more than doubled the average human score.
- Multi-agent LLM systems face a latency bottleneck because chain-of-thought is serial, whereas parallel test-time scaling techniques such as Best-of-N/consensus trade compute efficiency for lower latency.
- Diversity and routing are already practical multi-agent AI scaffolds, using the best model for each particular query rather than relying on a single reasoning model.

#### Relevance hooks
- Directly relevant to RL post-training benchmarking: compares self-play convergence, exploitability, and last-iterate RL algorithms in both perfect- and imperfect-information games.
- Relevant to agent evaluation methodology: contrasts minimax equilibrium vs. population best response, and emphasizes the need for human population data and statistical significance in human-agent evaluations.
- Relevant to RL reproducibility standards: reports confidence intervals (e.g., 86.5% ± 6.1%) and tournament-scale validation for Diplomacy agents.

#### Cited paper titles (verbatim only)
- "Cooperative AI: machines must learn to find common ground"
- "DORA: No-press Diplomacy from Scratch"

Index row: f25 | multiagent_berkeley_2025.pptx.pdf | Noam Brown | Self-play solves two-player zero-sum games but cooperative multi-agent AI requires human-data-informed population objectives and careful eval design. | ok
