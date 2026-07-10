# atropos/tinker_atropos/tests/ — INDEX

**Purpose:** pytest suite for the trainer's inference server and logprob/distillation machinery. Some tests hit the real Tinker API (need `TINKER_API_KEY`) and self-skip otherwise.

**Key files:**
- `test_serve_integration.py` — starts `serve.py` as a real subprocess and hits it over HTTP against the live Tinker API (requires `TINKER_API_KEY`).
- `test_logprobs_endpoint.py` — unit tests for the `/logprobs` FastAPI endpoint (mocked clients via `TestClient`).
- `test_logprob_alignment.py` — verifies token/logprob alignment between generation and scoring.
- `test_distillation.py` — tests the on-policy distillation path (teacher-logprob advantage).
- `test_logp_steering_env.py` — pure-logic tests for `LogpSteeringEnv` (e.g. `_extract_first_turn`); no network/tinker.
- `test_managed_server.py` — tests the managed (Tinker) server wrapper behavior.

**Find it fast:**
- to test without an API key → `test_logp_steering_env.py`, `test_logprobs_endpoint.py`
- to run full end-to-end serve check → `test_serve_integration.py` (set `TINKER_API_KEY`)
