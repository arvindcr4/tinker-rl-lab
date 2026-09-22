# Request: explicit licence and hosted-revision attestation for the Wordle wrapper

Hello EnvCommons / OpenReward maintainers,

I am preparing a reproducible, non-commercial benchmark evaluation. I have
pinned this repository at commit
`92bea32efa102e86275dedd2e0367e86d3754754`, but that tree contains no
`LICENSE`, `COPYING`, or `NOTICE` file. The OpenReward environment card links
to TextArena's MIT licence; that appears to cover the upstream game engine,
not this repository's OpenReward wrapper, split definition, reward mapping, or
deployment files.

Could you please either add an explicit licence file for this wrapper or
confirm in writing the licence that governs evaluation use and publication of
aggregate results?

For reproducibility, could you also confirm whether the hosted
`GeneralReasoning/Wordle` environment is bound to this commit (or expose a
deployed commit SHA), and whether the published `test` seeds are a
provider-defined evaluation split rather than a public convenience split?

I will not claim a model benchmark score from this environment until the
licence and suite/revision boundaries are evidenced. No API key, paid session,
or credentials are requested by this issue.

Thank you.
