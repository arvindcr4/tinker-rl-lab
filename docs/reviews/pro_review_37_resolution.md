# Review 37 resolution (partial source)

Only visible review text is addressed. The supplied review ends midway through the contamination discussion; missing ending and five-first list are skipped, not reconstructed.

- P0 self-attested receipts: WebBench v2 requires Ed25519 signatures over the complete envelope including schema, key ID, receipt ID, binding and payload. Ordinary hashes/true booleans cannot authenticate a result. Verilog decontamination records are explicitly assertions; absent an implemented external verifier, readiness remains blocked. Alternate authority files must equal the independently installed contract.
- P0 trust roots: supplied roots must match independently installed fingerprint anchors. Production anchor/key collections intentionally remain empty: no provider key was invented or taken from test fixtures. Test-only keys are scoped with mocks. Rotation/revocation is an installation responsibility; removing an installed anchor revokes it.
- P1 finite costs: both Verilog validators reject NaN, infinities, booleans and negative amounts; zero remains valid.
- P1 checkout provenance: require approved revision, license and roster, matching Git HEAD and clean tracked/untracked state before reading tasks. Synthetic layout tests explicitly mock only checkout provenance; they do not claim official source membership.
- P1 revision override: approved pin is enforced both in builder and metadata validator.
- P1 BALROG equivalence: no authentic official-run golden fixture was supplied. Existing pinned-native aggregation and complete episode gates are preserved. An invented fixture would not prove published-score equivalence. This external evidence obligation remains open.
- P1 AgentDojo error semantics: versioned v2 conservative policy withholds scores when any native error remains unresolved; complete receipt coverage alone cannot establish clean execution. Error IDs and raw traces remain retained. No unsupported claim of exact official retry/error equivalence is made.
- P2 native scoring semantics: trust and structure are kept separate; no new official score is manufactured. Official golden-fixture equivalence and authentic native verdict provenance remain required external evidence.
- P2 contamination (visible portion): manifest consistency does not prove unseen training data. WebBench authenticity gates and Verilog blocked assertion status prevent the specific promotions; AgentDojo retains its limited contamination scope. Missing remainder skipped.

Validation: 16 new offline trust/cost regressions plus migrated WebBench and Verilog tests. Signed fixture mutations are re-signed when testing structural predicates; tamper tests retain invalid signatures. CLI with an arbitrary root correctly fails. Seven real-checkout/data-dependent tests remain skipped; no datasets downloaded and no native benchmarks run.
