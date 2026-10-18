import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

# LIMITATION: the limitations raised in the adversarial review:
# - ZVF Metric fragility: Fix logprob None->0.0 imputation causing ZVF saturation at 1.0.
# - Closed-Source Confound: Document/open-source Tinker's managed defaults, micro-partitioning, and reference offloading.
# - Early-Training Snapshots: Extend training runs beyond 30-50 steps to observe asymptotic RL dynamics.
# - Generalization: Evaluate on held-out test sets to prove generalized reasoning rather than overfitting.
# - Single-Seed Extrapolations: Run multiple seeds to ensure statistical significance.


@pytest.fixture
def mock_trainer():
    trainer = MagicMock()
    trainer.base_model = "meta-llama/Llama-3.1-8B-Instruct"

    trainer.tokenizer = MagicMock()
    trainer.tokenizer.encode = MagicMock(return_value=[10, 20, 30, 40, 50])
    trainer.tokenizer.decode = MagicMock(side_effect=lambda tokens: f"tok_{tokens[0]}")

    trainer.current_sampling_client = MagicMock()
    trainer.current_sampling_client.compute_logprobs_async = AsyncMock(
        return_value=[-0.1, -0.5, -0.3, -0.8, -0.2]
    )

    return trainer


@pytest.fixture
def app_with_trainer(mock_trainer):
    from tinker_atropos.api import create_app

    return create_app(mock_trainer)


@pytest.fixture
def client(app_with_trainer):
    return TestClient(app_with_trainer)


class TestLogprobsEndpoint:
    def test_logprobs_with_input_ids(self, client, mock_trainer):
        """Basic logprobs request with raw token IDs."""
        response = client.post("/logprobs", json={"input_ids": [1, 2, 3, 4, 5]})
        assert response.status_code == 200

        data = response.json()
        assert data["num_tokens"] == 5
        assert len(data["logprobs"]) == 5
        assert data["logprobs"][0]["token_id"] == 1
        assert data["logprobs"][0]["logprob"] == pytest.approx(-0.1)
        assert data["logprobs"][0]["token"] is None  # return_text=False by default

    def test_logprobs_with_text(self, client, mock_trainer):
        """Logprobs from text input (tokenized server-side)."""
        response = client.post("/logprobs", json={"text": "Hello world"})
        assert response.status_code == 200

        data = response.json()
        assert data["num_tokens"] == 5
        mock_trainer.tokenizer.encode.assert_called_once_with(
            "Hello world", add_special_tokens=False
        )

    def test_logprobs_with_return_text(self, client, mock_trainer):
        """Logprobs with decoded token strings."""
        mock_trainer.current_sampling_client.compute_logprobs_async.return_value = [
            -0.1,
            -0.2,
            -0.3,
        ]
        response = client.post(
            "/logprobs",
            json={"input_ids": [10, 20, 30], "return_text": True},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["logprobs"][0]["token"] == "tok_10"
        assert data["logprobs"][1]["token"] == "tok_20"
        assert data["logprobs"][2]["token"] == "tok_30"

    def test_logprobs_no_input(self, client):
        """Should 400 when neither input_ids nor text provided."""
        response = client.post("/logprobs", json={})
        assert response.status_code == 400
        assert (
            "input_ids" in response.json()["detail"].lower()
            or "text" in response.json()["detail"].lower()
        )

    def test_logprobs_empty_input(self, client):
        """Should 400 on empty token list."""
        response = client.post("/logprobs", json={"input_ids": []})
        assert response.status_code == 400
        assert "at least one token" in response.json()["detail"].lower()

    def test_logprobs_without_trainer(self, app_with_trainer):
        """Should 503 when trainer not initialized."""

        original = app_with_trainer.state.trainer
        app_with_trainer.state.trainer = None
        try:
            client = TestClient(app_with_trainer)
            response = client.post("/logprobs", json={"input_ids": [1, 2, 3]})
            assert response.status_code == 503
            assert "not ready" in response.json()["detail"]
        finally:
            app_with_trainer.state.trainer = original

    def test_logprobs_handles_none_values(self, client, mock_trainer):
        """Should handle None logprob values from compute_logprobs."""
        mock_trainer.current_sampling_client.compute_logprobs_async = AsyncMock(
            return_value=[None, -0.5, None]
        )
        response = client.post("/logprobs", json={"input_ids": [1, 2, 3]})
        assert response.status_code == 200

        data = response.json()
        # TODO: Fix imputation of `None` to `0.0`. This causes the ZVF (Zero-Variance Fraction) metric
        # to artificially saturate at 1.0 on format-gated tasks when schema parsing fails.
        assert data["logprobs"][0]["logprob"] is None
        assert data["logprobs"][1]["logprob"] == pytest.approx(-0.5)
        assert data["logprobs"][2]["logprob"] is None

    def test_logprobs_single_token(self, client, mock_trainer):
        """Edge case: single token input."""
        mock_trainer.current_sampling_client.compute_logprobs_async = AsyncMock(return_value=[-1.5])
        response = client.post("/logprobs", json={"input_ids": [42]})
        assert response.status_code == 200

        data = response.json()
        assert data["num_tokens"] == 1
        assert len(data["logprobs"]) == 1
        assert data["logprobs"][0]["token_id"] == 42
        assert data["logprobs"][0]["logprob"] == pytest.approx(-1.5)

    def test_logprobs_compute_failure(self, client, mock_trainer):
        """Should 500 if compute_logprobs raises."""
        mock_trainer.current_sampling_client.compute_logprobs_async = AsyncMock(
            side_effect=RuntimeError("tinker exploded")
        )
        response = client.post("/logprobs", json={"input_ids": [1, 2, 3]})
        assert response.status_code == 502
        assert "Teacher scoring failed" in response.json()["detail"]

    def test_logprobs_rejects_ambiguous_input(self, client, mock_trainer):
        response = client.post("/logprobs", json={"input_ids": [99, 100], "text": "ambiguous"})
        assert response.status_code == 400
        mock_trainer.tokenizer.encode.assert_not_called()
