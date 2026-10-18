# PR21 runtime integration follow-ups

Offline source repairs; no training, paid jobs or stored results were changed.

- 4305495285 fixed: curriculum summaries consume the renamed near-zero scalar-loss metric.
- 4305495293 / 4305495295 fixed: corrected string platform IDs use measured manifest GPU names; legacy env mappings remain accepted and missing GPU evidence fails closed.
- 4305495298 fixed: checked-in sweep configs and the generator declare reward versions explicitly.
- 4305495314 fixed: launcher selects a separate Qwen3-8B recipe and reports the model from that recipe; historical 1.5B recipe retained unchanged.
- 4305495326 fixed: Math500 uses the boxed mathematical-answer grader.
- 4305495349 fixed: scalar/text-only legacy Colab cache cannot be reused as a verified checkpoint bundle. A fresh output root is required; no automatic rerun is launched.
- 4305495382 fixed: reward identity binds both the environment wrapper and delegated grading source with a versioned manifest.
- 4305495411 fixed: measured W&B summary endpoints are promoted into the analysis namespace; config-only endpoint declarations are not accepted as observations.

Validation: 232 tests passed across the new synthetic regressions and hybrid framework, hybrid core, and backend/infra suites. These are offline tests, not provider/runtime validation.
