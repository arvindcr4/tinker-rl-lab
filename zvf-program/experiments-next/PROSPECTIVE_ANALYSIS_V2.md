# Prospective pool-analysis-v2

This contract applies to new runs only. Existing result JSON, review slides,
progress logs, and manuscripts are historical evidence and are not rewritten.
Their earlier “validated”, Wald-readiness or unique-G-star statements must not
be used as the launch decision for this prospective protocol.

## Frozen identity and completeness
`analysis_contract.py` is the executable specification. A live builder requires
immutable model, checkpoint, dataset and tokenizer identities; dataset/tokenizer
revisions are passed to their loaders. Checkpoint/model identity declarations
remain operator attestations because this API exposes no independent content
hash verifier. They must be audited against provider records before publication;
metadata presence alone is not proof of immutability. Resume always uses the
original resolved sampler path, including base-model snapshots, and rejects
missing/changed requested paths or model/config identities before provider setup.
It then compares the complete prompt manifest before resuming.

Full means every planned prompt, every reward and every token count is present.
Dataset shortfall requires a new explicit sample plan, never silent completion.
Deliberate finalization records truncated/exploratory and preserves the original.
The default theory loaders reject truncated and legacy-unverified pools. The
quality CLI allows explicit `--allow-partial`, propagates completeness, and seed
aggregation rejects anything except full versioned inputs. Legacy analysis is
still possible explicitly but cannot become confirmatory by changing `status`.

Seed aggregation compares the entire contract, including manifest and prompt
selection policy. Thus different prompt manifests cannot silently masquerade as
replicates. A future varying-prompt evaluation-seed design needs its own frozen
selection/population contract rather than disabling this check. Evaluation seeds
are not independent training seeds.

## Statistical decision contract
`analysis_contract.T1_DECISION` is authoritative for future output: Wilson,
shuffled batches, minimum batch size 64, exploratory empirical evidence.
Curriculum global inference and controller readiness remain blocked. T1's exact
truth is conditional on the finite recorded binary pool, not population truth.
Stratified draws use all prompts, actual stratum weights and a weighted Hoeffding
bound; one sample per stratum cannot estimate within-stratum variance. Ordered
Wilson intervals are diagnostic only. Coverage has its own Monte Carlo interval.

T2 uses exact finite-pool ZVF, nearest-rank quantiles, explicit right censoring,
DKW quantile/ratio bounds (null for unidentifiable tails), and binomial coverage
intervals for the event waiting-time <= model budget when observable. A point
quantile below budget is never a validation decision. Conditional simulation does
not remove uncertainty about the original prompt population.

T3 outputs name their objectives. P(mixed)/G is descriptive; variance-weighted
J(G) is a proxy. J(2)=J(3) for all Bernoulli probabilities and their mixtures.
Neither a tie-broken argmax nor empirical agreement establishes a unique optimum
for learning. No new training outcome is claimed.

## ZVF conformance
The pool pipeline uses `binary-exact-v1`: finite rewards in {0,1}, population
variance, epsilon zero, at least two rewards, singleton rejected. GU is the
fraction of non-degenerate reward groups, not measured optimizer-gradient use.
Continuous-reward triage uses a separately declared epsilon/variance contract;
its metrics are not interchangeable with this binary contract. Near-identical
continuous rewards are rejected here rather than silently reclassified. All
comparisons must carry the definition version and reward-parser source digest.
