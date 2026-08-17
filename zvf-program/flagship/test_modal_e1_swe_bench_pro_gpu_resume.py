from __future__ import annotations

import unittest

try:
    from .modal_e1_swe_bench_pro_gpu_resume import (
        GENERATION_BACKEND,
        GPU_RATE_USD_PER_SECOND,
        _adapter_module_plan,
        _augment_receipt,
        _gpu_projection,
        _merge_compute_cost,
    )
except ImportError:
    from modal_e1_swe_bench_pro_gpu_resume import (
        GENERATION_BACKEND,
        GPU_RATE_USD_PER_SECOND,
        _adapter_module_plan,
        _augment_receipt,
        _gpu_projection,
        _merge_compute_cost,
    )


class E1GpuResumeTests(unittest.TestCase):
    def test_projection_includes_one_load_and_each_pending_task(self) -> None:
        self.assertEqual(
            _gpu_projection(2),
            round((600 + 2 * 60) * GPU_RATE_USD_PER_SECOND, 6),
        )

    def test_receipt_discloses_mixed_backends_and_gpu_cost(self) -> None:
        receipt = {
            "claim_boundary": "Exact suite.",
            "sampling": {},
            "cost": {},
            "artifacts": {},
            "receipt_sha256": "old",
        }
        generations = [
            {"generation_backend": None, "estimated_modal_gpu_usd": 0.0},
            {
                "generation_backend": GENERATION_BACKEND,
                "estimated_modal_gpu_usd": 0.25,
            },
        ]
        backend = {
            "path": "outputs/run/gpu_backend_receipt.json",
            "estimated_modal_gpu_usd": 0.5,
            "merge": {
                "base_commit": "base",
                "adapter_commit": "adapter",
                "merge_method": "exact",
                "merged_path": "/merged/model",
                "weight_bytes": 100,
                "weight_shard_sha256": {"model.safetensors": "abc"},
                "estimated_modal_cpu_memory_usd": 0.125,
            },
        }

        second_backend = {
            "path": "outputs/run/gpu_backend_receipt_lane2.json",
            "estimated_modal_gpu_usd": 0.25,
            "merge": {
                "base_commit": "base",
                "adapter_commit": "adapter",
                "merge_method": "exact",
                "merged_path": "/merged/model",
                "weight_bytes": 100,
                "weight_shard_sha256": {"model.safetensors": "abc"},
                "estimated_modal_cpu_memory_usd": 0.125,
            },
        }

        result = _augment_receipt(
            receipt, generations, [backend, second_backend]
        )

        self.assertTrue(result["sampling"]["mixed_inference_backends"])
        self.assertEqual(
            result["sampling"]["backend_counts"],
            {GENERATION_BACKEND: 1, "tinker_remote": 1},
        )
        self.assertEqual(result["sampling"]["modal_gpu_lane_startups"], 2)
        self.assertEqual(result["cost"]["estimated_modal_gpu_total_usd"], 1.0)
        self.assertEqual(
            result["cost"]["estimated_modal_total_compute_usd"], 1.125
        )
        self.assertNotEqual(result["receipt_sha256"], "old")

    def test_merge_cost_accounts_for_cpu_and_memory(self) -> None:
        self.assertEqual(_merge_compute_cost(1.0), 0.00049376)

    def test_adapter_plan_maps_fused_qkv_and_experts(self) -> None:
        prefix = "base_model.model.model.layers.0"
        self.assertEqual(
            _adapter_module_plan(f"{prefix}.linear_attn.in_proj_k"),
            (
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                "qkv",
                "k",
            ),
        )
        self.assertEqual(
            _adapter_module_plan(f"{prefix}.mlp.experts.w3"),
            (
                "model.language_model.layers.0.mlp.experts.gate_up_proj",
                "gate_up",
                "w3",
            ),
        )
        self.assertEqual(
            _adapter_module_plan("base_model.model.model.unembed_tokens"),
            ("lm_head.weight", "simple", None),
        )


if __name__ == "__main__":
    unittest.main()
