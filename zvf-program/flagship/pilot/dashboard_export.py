from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from pilot.entropic_gating import compute_entropy_density, eliminate_noise_components
from pilot.spectral_attention import legendre_spectral_projection, spectral_pairwise_distance


class DashboardExportError(RuntimeError):
    """Raised when dashboard export receives invalid data or fails contract checks."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DashboardExportError(message)


def _to_numpy(data: Any) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if isinstance(data, (list, tuple)):
        data = np.array(data, dtype=np.float64)
    if not isinstance(data, np.ndarray):
        raise DashboardExportError(f"Unsupported data type for array conversion: {type(data)}")
    if not np.issubdtype(data.dtype, np.number):
        raise DashboardExportError("Array contains non-numeric data")
    if not np.isfinite(data).all():
        raise DashboardExportError("Array contains non-finite values (NaN or Inf)")
    return data


@dataclass(frozen=True, slots=True)
class SpectralTrajectoryData:
    distances: list[list[float]]
    mode_coefficients: list[list[float]]
    labels: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EntropicGatingData:
    pre_gating_entropy: list[list[float]]
    post_gating_entropy: list[list[float]]
    gating_mask: list[list[float]]
    token_labels: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GradientRecoveryData:
    steps: list[int]
    curves: dict[str, list[float]]
    cosine_sims: dict[str, list[float | None]]
    relative_l2_errors: dict[str, list[float | None]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_spectral_trajectory_data(
    spectral_coeffs: torch.Tensor | np.ndarray | Sequence[Any],
    labels: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SpectralTrajectoryData:
    """Prepare spectral trajectory distance matrix and mode energy payload.

    spectral_coeffs: Tensor or array of shape [B, N, D] or [N, D]
    """
    coeffs_arr = _to_numpy(spectral_coeffs)
    _require(coeffs_arr.ndim in {2, 3}, f"spectral_coeffs must be 2D or 3D, got shape {coeffs_arr.shape}")

    if coeffs_arr.ndim == 2:
        # Shape [N, D] - treat rows as modes
        n_modes, d_model = coeffs_arr.shape
        batch_size = n_modes
        # Pairwise mode L2 distances
        diffs = coeffs_arr[:, None, :] - coeffs_arr[None, :, :]
        # Continuous L2 weighting for legendre modes if applicable
        n_idx = np.arange(n_modes)
        scale_n = 1.0 / (2.0 * n_idx + 1.0)
        dist_matrix = np.sqrt(np.sum((diffs ** 2) * scale_n[None, :, None], axis=-1))
        mode_coeffs = coeffs_arr.tolist()
    else:
        # Shape [B, N, D] - compute pairwise trajectory distances between batch items
        batch_size, n_modes, d_model = coeffs_arr.shape
        coeffs_t = torch.as_tensor(coeffs_arr, dtype=torch.float32)
        dist_matrix_np = np.zeros((batch_size, batch_size), dtype=np.float64)
        for i in range(batch_size):
            for j in range(batch_size):
                d = spectral_pairwise_distance(coeffs_t[i], coeffs_t[j])
                dist_matrix_np[i, j] = float(d.item())
        dist_matrix = dist_matrix_np
        # Average mode coefficients across batch
        mode_coeffs = coeffs_arr.mean(axis=0).tolist()

    if labels is None:
        item_labels = [f"Item_{i}" for i in range(batch_size)]
    else:
        _require(len(labels) == batch_size, f"labels length {len(labels)} mismatch with count {batch_size}")
        item_labels = list(labels)

    meta = {
        "n_modes": int(n_modes),
        "d_model": int(d_model),
        "batch_size": int(batch_size),
        **(metadata or {}),
    }
    return SpectralTrajectoryData(
        distances=dist_matrix.tolist(),
        mode_coefficients=mode_coeffs,
        labels=item_labels,
        metadata=meta,
    )


def prepare_entropic_gating_data(
    logits_or_probs: torch.Tensor | np.ndarray | Sequence[Any],
    n_noise_dims: int = 1,
    token_labels: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> EntropicGatingData:
    """Prepare pre and post Givens entropic gating density heatmap payload."""
    data_arr = _to_numpy(logits_or_probs)
    _require(data_arr.ndim == 2, f"logits_or_probs must be 2D [B, L] or [L, D], got shape {data_arr.shape}")
    rows, cols = data_arr.shape
    _require(1 <= n_noise_dims < cols, f"n_noise_dims={n_noise_dims} invalid for feature dim {cols}")

    t_data = torch.as_tensor(data_arr, dtype=torch.float32)
    # Check if probabilities or logits
    if torch.any(t_data < 0) or not torch.allclose(t_data.sum(dim=-1), torch.ones(rows), atol=1e-2):
        probs = torch.softmax(t_data, dim=-1)
    else:
        probs = t_data

    pre_entropy = compute_entropy_density(probs, dim=-1)
    x_proj, x_rot = eliminate_noise_components(t_data, n_noise_dims=n_noise_dims)

    # Post-gating entropy calculation on rotated/projected features
    post_probs = torch.softmax(x_proj, dim=-1)
    post_entropy = compute_entropy_density(post_probs, dim=-1)

    # Mask showing rotated out noise coordinates
    mask = np.zeros((rows, cols), dtype=np.float64)
    mask[:, (cols - n_noise_dims):] = 1.0

    if pre_entropy.ndim == 1:
        # Map into a 2D grid for heatmap rendering
        pre_grid = pre_entropy.unsqueeze(-1).repeat(1, cols).detach().cpu().numpy()
        post_grid = post_entropy.unsqueeze(-1).repeat(1, cols).detach().cpu().numpy()
    else:
        pre_grid = pre_entropy.detach().cpu().numpy()
        post_grid = post_entropy.detach().cpu().numpy()

    if token_labels is None:
        t_labels = [f"Token_{i}" for i in range(rows)]
    else:
        _require(len(token_labels) == rows, f"token_labels length {len(token_labels)} mismatch with {rows}")
        t_labels = list(token_labels)

    meta = {
        "n_noise_dims": int(n_noise_dims),
        "sequence_length": int(rows),
        "feature_dim": int(cols),
        **(metadata or {}),
    }

    return EntropicGatingData(
        pre_gating_entropy=pre_grid.tolist(),
        post_gating_entropy=post_grid.tolist(),
        gating_mask=mask.tolist(),
        token_labels=t_labels,
        metadata=meta,
    )


def prepare_gradient_recovery_data(
    receipts_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    metadata: Mapping[str, Any] | None = None,
) -> GradientRecoveryData:
    """Prepare step-wise gradient recovery curves across different GRPO conditions."""
    _require(len(receipts_by_condition) > 0, "receipts_by_condition must not be empty")

    steps_set: set[int] | None = None
    curves: dict[str, list[float]] = {}
    cosine_sims: dict[str, list[float | None]] = {}
    relative_l2s: dict[str, list[float | None]] = {}

    for condition, receipts in receipts_by_condition.items():
        _require(len(receipts) > 0, f"Receipt list for condition {condition} is empty")
        cond_steps = [int(r["step"]) for r in receipts]
        if steps_set is None:
            steps_set = set(cond_steps)
            steps_list = cond_steps
        else:
            _require(set(cond_steps) == steps_set, f"Steps mismatch for condition {condition}")

        norm_curve: list[float] = []
        cos_curve: list[float | None] = []
        l2_curve: list[float | None] = []

        for r in receipts:
            # Gradient norm / retention percentage
            norm = r.get("gradient_norm") or r.get("intended_gradient_norm") or r.get("selected_vs_intended_cosine")
            if norm is None:
                # Calculate retention from relative_l2 or fallback
                rel_l2 = r.get("gradient_relative_l2")
                norm = 1.0 - rel_l2 if rel_l2 is not None else 1.0
            norm_curve.append(float(norm))

            cos_val = r.get("gradient_cosine") or r.get("selected_vs_intended_cosine")
            l2_val = r.get("gradient_relative_l2") or r.get("selected_vs_intended_relative_l2")

            cos_curve.append(float(cos_val) if cos_val is not None else None)
            l2_curve.append(float(l2_val) if l2_val is not None else None)

        curves[condition] = norm_curve
        cosine_sims[condition] = cos_curve
        relative_l2s[condition] = l2_curve

    assert steps_set is not None
    sorted_steps = sorted(steps_list)

    meta = {
        "num_conditions": len(curves),
        "total_steps": len(sorted_steps),
        **(metadata or {}),
    }

    return GradientRecoveryData(
        steps=sorted_steps,
        curves=curves,
        cosine_sims=cosine_sims,
        relative_l2_errors=relative_l2s,
        metadata=meta,
    )


def _render_page_wrapper(title: str, body_html: str, data_js: str) -> str:
    """Generate standalone HTML document wrapper with embedded CSS and JS."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: #151d2a;
            --card-border: #232f45;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --accent-cyan: #38bdf8;
            --accent-emerald: #34d399;
            --accent-purple: #c084fc;
            --accent-amber: #fbbf24;
            --accent-rose: #f43f5e;
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}

        [data-theme="light"] {{
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --card-border: #e2e8f0;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --accent-cyan: #0284c7;
            --accent-emerald: #059669;
            --accent-purple: #7c3aed;
            --accent-amber: #d97706;
            --accent-rose: #e11d48;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: var(--font-family);
            line-height: 1.5;
            padding: 24px;
            transition: background-color 0.2s, color 0.2s;
        }}

        .dashboard-container {{
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }}

        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--card-border);
        }}

        .header-title h1 {{
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            color: var(--accent-cyan);
        }}

        .header-title p {{
            font-size: 0.875rem;
            color: var(--text-muted);
            margin-top: 4px;
        }}

        .theme-toggle {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            color: var(--text-main);
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.875rem;
            font-weight: 500;
            transition: all 0.2s;
        }}

        .theme-toggle:hover {{
            border-color: var(--accent-cyan);
        }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
        }}

        .kpi-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}

        .kpi-card .label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }}

        .kpi-card .value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-main);
        }}

        .kpi-card .subtext {{
            font-size: 0.75rem;
            color: var(--accent-emerald);
        }}

        .visualizer-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}

        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .card-header h2 {{
            font-size: 1.125rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .badge {{
            font-size: 0.7rem;
            padding: 2px 8px;
            border-radius: 9999px;
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent-cyan);
            border: 1px solid rgba(56, 189, 248, 0.3);
            font-weight: 600;
        }}

        .chart-container {{
            width: 100%;
            position: relative;
            min-height: 320px;
            display: flex;
            justify-content: center;
            align-items: center;
        }}

        svg {{
            width: 100%;
            height: 100%;
            overflow: visible;
        }}

        .tooltip {{
            position: absolute;
            background: rgba(15, 23, 42, 0.95);
            color: #fff;
            border: 1px solid var(--card-border);
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 0.75rem;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.15s ease-in-out;
            z-index: 100;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        }}

        .controls-row {{
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            align-items: center;
            font-size: 0.85rem;
        }}

        .control-group {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .control-group select, .control-group button {{
            background: var(--bg-color);
            color: var(--text-main);
            border: 1px solid var(--card-border);
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.8rem;
            outline: none;
        }}

        .tabs {{
            display: flex;
            gap: 8px;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 8px;
        }}

        .tab-btn {{
            background: none;
            border: none;
            color: var(--text-muted);
            padding: 6px 16px;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            border-radius: 6px;
            transition: all 0.2s;
        }}

        .tab-btn.active {{
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent-cyan);
        }}

        .tab-content {{
            display: none;
        }}

        .tab-content.active {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="header">
            <div class="header-title">
                <h1>{title}</h1>
                <p>ZAI RL Lab Interactive Pilot Conformance Visualizers</p>
            </div>
            <button class="theme-toggle" onclick="toggleTheme()">🌓 Toggle Theme</button>
        </div>

        {body_html}
    </div>

    <div id="tooltip" class="tooltip"></div>

    <script>
        {data_js}

        function toggleTheme() {{
            const current = document.documentElement.getAttribute('data-theme');
            const target = current === 'light' ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', target);
        }}

        function showTooltip(evt, text) {{
            const tooltip = document.getElementById('tooltip');
            tooltip.innerHTML = text;
            tooltip.style.opacity = '1';
            tooltip.style.left = (evt.pageX + 15) + 'px';
            tooltip.style.top = (evt.pageY - 15) + 'px';
        }}

        function hideTooltip() {{
            const tooltip = document.getElementById('tooltip');
            tooltip.style.opacity = '0';
        }}

        function switchTab(tabId) {{
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            document.getElementById('btn-' + tabId).classList.add('active');
            document.getElementById('tab-' + tabId).classList.add('active');
        }}
    </script>
</body>
</html>"""


def export_spectral_trajectory_html(
    data: SpectralTrajectoryData | Mapping[str, Any],
    output_path: str | Path | None = None,
    title: str = "Spectral Trajectory Distances",
) -> str:
    """Generate interactive HTML visualizer for spectral trajectory distances."""
    if isinstance(data, SpectralTrajectoryData):
        payload = data.to_dict()
    elif isinstance(data, Mapping):
        _require("distances" in data and "labels" in data, "Mapping payload lacks required keys")
        payload = dict(data)
    else:
        raise DashboardExportError(f"Invalid spectral data type: {type(data)}")

    data_json = json.dumps(payload, indent=2)

    body_html = f"""
    <div class="visualizer-card">
        <div class="card-header">
            <h2>Spectral Trajectory Continuous L² Distance Matrix <span class="badge">Legendre Modes</span></h2>
        </div>
        <div class="controls-row">
            <div class="control-group">
                <label>Color Scale:</label>
                <select id="spectral-color-scale" onchange="renderSpectralMatrix()">
                    <option value="cyan">Cyan / Emerald</option>
                    <option value="purple">Purple / Rose</option>
                </select>
            </div>
        </div>
        <div class="chart-container" id="spectral-chart-container">
            <svg id="spectral-matrix-svg" viewBox="0 0 600 500"></svg>
        </div>
    </div>
    """

    data_js = f"""
    const spectralData = {data_json};

    function renderSpectralMatrix() {{
        const svg = document.getElementById('spectral-matrix-svg');
        const matrix = spectralData.distances;
        const labels = spectralData.labels;
        const n = matrix.length;
        if (n === 0) return;

        let maxVal = 0;
        for (let i = 0; i < n; i++) {{
            for (let j = 0; j < n; j++) {{
                if (matrix[i][j] > maxVal) maxVal = matrix[i][j];
            }}
        }}
        if (maxVal === 0) maxVal = 1.0;

        const margin = 80;
        const cellSize = Math.min((500 - margin) / n, 50);
        const width = margin + n * cellSize + 20;
        const height = margin + n * cellSize + 20;

        let html = `<g transform="translate(${{margin}}, ${{margin}})">`;

        // Render cells
        for (let i = 0; i < n; i++) {{
            for (let j = 0; j < n; j++) {{
                const val = matrix[i][j];
                const norm = val / maxVal;
                const r = Math.round(15 + norm * 200);
                const g = Math.round(189 - norm * 100);
                const b = Math.round(248 - norm * 50);
                const fill = `rgb(${{r}}, ${{g}}, ${{b}})`;

                html += `<rect x="${{j * cellSize}}" y="${{i * cellSize}}" width="${{cellSize - 2}}" height="${{cellSize - 2}}"
                         fill="${{fill}}" rx="3"
                         onmouseover="showTooltip(event, '${{labels[i]}} vs ${{labels[j]}}<br>Distance: <b>${{val.toFixed(4)}}</b>')"
                         onmouseout="hideTooltip()"/>`;
                html += `<text x="${{j * cellSize + cellSize / 2}}" y="${{i * cellSize + cellSize / 2 + 4}}"
                         font-size="10" fill="#ffffff" text-anchor="middle">${{val.toFixed(2)}}</text>`;
            }}
        }}

        // Row/Col Labels
        for (let i = 0; i < n; i++) {{
            html += `<text x="-10" y="${{i * cellSize + cellSize / 2 + 4}}" font-size="11" fill="var(--text-muted)" text-anchor="end">${{labels[i]}}</text>`;
            html += `<text x="${{i * cellSize + cellSize / 2}}" y="-10" font-size="11" fill="var(--text-muted)" text-anchor="middle" transform="rotate(-30, ${{i * cellSize + cellSize / 2}}, -10)">${{labels[i]}}</text>`;
        }}

        html += `</g>`;
        svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
        svg.innerHTML = html;
    }}

    document.addEventListener('DOMContentLoaded', renderSpectralMatrix);
    if (document.readyState !== 'loading') renderSpectralMatrix();
    """

    full_html = _render_page_wrapper(title, body_html, data_js)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(full_html, encoding="utf-8")
    return full_html


def export_gating_density_heatmap_html(
    data: EntropicGatingData | Mapping[str, Any],
    output_path: str | Path | None = None,
    title: str = "Givens Entropic Gating Heatmap",
) -> str:
    """Generate interactive HTML visualizer for Givens entropic gating density heatmaps."""
    if isinstance(data, EntropicGatingData):
        payload = data.to_dict()
    elif isinstance(data, Mapping):
        _require("pre_gating_entropy" in data and "post_gating_entropy" in data, "Mapping lacks entropy fields")
        payload = dict(data)
    else:
        raise DashboardExportError(f"Invalid entropic gating data type: {type(data)}")

    data_json = json.dumps(payload, indent=2)

    body_html = f"""
    <div class="visualizer-card">
        <div class="card-header">
            <h2>Givens Planar Unitary Entropic Gating Heatmap <span class="badge">Pre vs Post Noise Rotation</span></h2>
        </div>
        <div class="controls-row">
            <div class="control-group">
                <label>View Mode:</label>
                <select id="gating-view-mode" onchange="renderGatingHeatmap()">
                    <option value="side-by-side">Side-by-Side Comparison</option>
                    <option value="post-only">Post-Gating Projected</option>
                    <option value="difference">Entropy Reduction Density</option>
                </select>
            </div>
        </div>
        <div class="chart-container" id="gating-chart-container">
            <svg id="gating-heatmap-svg" viewBox="0 0 900 450"></svg>
        </div>
    </div>
    """

    data_js = f"""
    const gatingData = {data_json};

    function renderGatingHeatmap() {{
        const svg = document.getElementById('gating-heatmap-svg');
        const pre = gatingData.pre_gating_entropy;
        const post = gatingData.post_gating_entropy;
        const rows = pre.length;
        const cols = pre[0].length;
        const mode = document.getElementById('gating-view-mode').value;

        const cellW = Math.min(750 / cols, 35);
        const cellH = Math.min(350 / rows, 30);
        const margin = 60;

        let html = '';

        if (mode === 'side-by-side') {{
            // Render two heatmaps side-by-side
            const renderGrid = (matrix, titleText, offsetX) => {{
                let res = `<g transform="translate(${{offsetX}}, ${{margin}})">`;
                res += `<text x="${{(cols * cellW) / 2}}" y="-20" font-size="14" font-weight="600" fill="var(--accent-cyan)" text-anchor="middle">${{titleText}}</text>`;
                for (let r = 0; r < rows; r++) {{
                    for (let c = 0; c < cols; c++) {{
                        const val = matrix[r][c];
                        const alpha = Math.min(Math.max(val / 3.0, 0.05), 1.0);
                        const fill = `rgba(56, 189, 248, ${{alpha}})`;
                        res += `<rect x="${{c * cellW}}" y="${{r * cellH}}" width="${{cellW - 1}}" height="${{cellH - 1}}" fill="${{fill}}" rx="2"
                                 onmouseover="showTooltip(event, 'Row ${{r}}, Col ${{c}}<br>Entropy H(p): <b>${{val.toFixed(4)}}</b>')"
                                 onmouseout="hideTooltip()"/>`;
                    }}
                }}
                res += `</g>`;
                return res;
            }};
            html += renderGrid(pre, "Pre-Gating Entropy", 50);
            html += renderGrid(post, "Post-Gating (Givens Projected)", 50 + cols * cellW + 60);
        }} else {{
            // Render single grid
            const targetMatrix = mode === 'post-only' ? post : pre.map((rArr, rIdx) => rArr.map((v, cIdx) => v - post[rIdx][cIdx]));
            const label = mode === 'post-only' ? "Post-Gating Entropy Density" : "Entropy Reduction (ΔH)";
            html += `<g transform="translate(80, ${{margin}})">`;
            html += `<text x="${{(cols * cellW) / 2}}" y="-20" font-size="14" font-weight="600" fill="var(--accent-emerald)" text-anchor="middle">${{label}}</text>`;
            for (let r = 0; r < rows; r++) {{
                for (let c = 0; c < cols; c++) {{
                    const val = targetMatrix[r][c];
                    const alpha = Math.min(Math.max(val / 2.0, 0.05), 1.0);
                    const color = mode === 'post-only' ? `rgba(52, 211, 153, ${{alpha}})` : `rgba(244, 63, 94, ${{alpha}})`;
                    html += `<rect x="${{c * cellW}}" y="${{r * cellH}}" width="${{cellW - 1}}" height="${{cellH - 1}}" fill="${{color}}" rx="2"
                             onmouseover="showTooltip(event, 'Row ${{r}}, Col ${{c}}<br>Value: <b>${{val.toFixed(4)}}</b>')"
                             onmouseout="hideTooltip()"/>`;
                }}
            }}
            html += `</g>`;
        }}

        svg.innerHTML = html;
    }}

    document.addEventListener('DOMContentLoaded', renderGatingHeatmap);
    if (document.readyState !== 'loading') renderGatingHeatmap();
    """

    full_html = _render_page_wrapper(title, body_html, data_js)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(full_html, encoding="utf-8")
    return full_html


def export_gradient_recovery_html(
    data: GradientRecoveryData | Mapping[str, Any],
    output_path: str | Path | None = None,
    title: str = "Gradient Norm Recovery Curves",
) -> str:
    """Generate interactive HTML visualizer for gradient norm recovery curves."""
    if isinstance(data, GradientRecoveryData):
        payload = data.to_dict()
    elif isinstance(data, Mapping):
        _require("steps" in data and "curves" in data, "Mapping lacks steps or curves")
        payload = dict(data)
    else:
        raise DashboardExportError(f"Invalid gradient recovery data type: {type(data)}")

    data_json = json.dumps(payload, indent=2)

    body_html = f"""
    <div class="visualizer-card">
        <div class="card-header">
            <h2>Gradient Norm Retention & Recovery Curves <span class="badge">Comparative Analysis</span></h2>
        </div>
        <div class="controls-row" id="condition-checkboxes">
            <!-- Dynamic checkboxes for conditions -->
        </div>
        <div class="chart-container" id="gradient-chart-container">
            <svg id="gradient-curves-svg" viewBox="0 0 900 400"></svg>
        </div>
    </div>
    """

    data_js = f"""
    const gradientData = {data_json};

    const colorPalette = {{
        'intended_full': '#38bdf8',
        'native_trl': '#f43f5e',
        'epsilon_only': '#fbbf24',
        'reduction_only': '#c084fc',
        'spectral_legendre': '#34d399',
        'entropic_givens': '#60a5fa'
    }};

    function initConditionControls() {{
        const container = document.getElementById('condition-checkboxes');
        let html = '<span style="font-weight:600;margin-right:8px;">Conditions:</span>';
        for (const cond of Object.keys(gradientData.curves)) {{
            const color = colorPalette[cond] || '#94a3b8';
            html += `<label style="display:inline-flex;align-items:center;gap:4px;margin-right:12px;cursor:pointer;">
                        <input type="checkbox" value="${{cond}}" checked onchange="renderGradientCurves()">
                        <span style="color:${{color}};font-weight:600;">${{cond}}</span>
                     </label>`;
        }}
        container.innerHTML = html;
    }}

    function renderGradientCurves() {{
        const svg = document.getElementById('gradient-curves-svg');
        const steps = gradientData.steps;
        const curves = gradientData.curves;
        const activeConds = Array.from(document.querySelectorAll('#condition-checkboxes input:checked')).map(cb => cb.value);

        if (steps.length === 0 || activeConds.length === 0) {{
            svg.innerHTML = '<text x="450" y="200" text-anchor="middle" fill="var(--text-muted)">Select at least one condition to display curves.</text>';
            return;
        }}

        const margin = {{top: 40, right: 40, bottom: 50, left: 60}};
        const width = 900 - margin.left - margin.right;
        const height = 400 - margin.top - margin.bottom;

        const minStep = steps[0];
        const maxStep = steps[steps.length - 1];

        let maxVal = 0;
        activeConds.forEach(cond => {{
            curves[cond].forEach(v => {{ if (v > maxVal) maxVal = v; }});
        }});
        if (maxVal === 0) maxVal = 1.0;

        const xScale = (step) => margin.left + ((step - minStep) / (maxStep - minStep || 1)) * width;
        const yScale = (val) => margin.top + height - (val / maxVal) * height;

        let html = `<g>`;
        // Axes
        html += `<line x1="${{margin.left}}" y1="${{margin.top + height}}" x2="${{margin.left + width}}" y2="${{margin.top + height}}" stroke="var(--card-border)" stroke-width="2"/>`;
        html += `<line x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{margin.top + height}}" stroke="var(--card-border)" stroke-width="2"/>`;

        // Y-ticks
        for (let i = 0; i <= 5; i++) {{
            const yVal = (maxVal * i) / 5;
            const yPos = yScale(yVal);
            html += `<line x1="${{margin.left - 5}}" y1="${{yPos}}" x2="${{margin.left + width}}" y2="${{yPos}}" stroke="var(--card-border)" stroke-dasharray="4" opacity="0.4"/>`;
            html += `<text x="${{margin.left - 10}}" y="${{yPos + 4}}" font-size="10" fill="var(--text-muted)" text-anchor="end">${{yVal.toFixed(2)}}</text>`;
        }}

        // X-ticks
        steps.forEach((step, idx) => {{
            if (idx % Math.ceil(steps.length / 8) === 0 || idx === steps.length - 1) {{
                const xPos = xScale(step);
                html += `<text x="${{xPos}}" y="${{margin.top + height + 20}}" font-size="10" fill="var(--text-muted)" text-anchor="middle">Step ${{step}}</text>`;
            }}
        }});

        // Render line paths
        activeConds.forEach(cond => {{
            const color = colorPalette[cond] || '#94a3b8';
            const values = curves[cond];
            let pathD = '';

            values.forEach((v, idx) => {{
                const x = xScale(steps[idx]);
                const y = yScale(v);
                if (idx === 0) pathD += `M ${{x}} ${{y}}`;
                else pathD += ` L ${{x}} ${{y}}`;
            }});

            html += `<path d="${{pathD}}" fill="none" stroke="${{color}}" stroke-width="3.5" opacity="0.9"/>`;

            // Draw data points
            values.forEach((v, idx) => {{
                const x = xScale(steps[idx]);
                const y = yScale(v);
                html += `<circle cx="${{x}}" cy="${{y}}" r="4" fill="${{color}}"
                         onmouseover="showTooltip(event, 'Condition: <b>${{cond}}</b><br>Step: <b>${{steps[idx]}}</b><br>Retention/Norm: <b>${{v.toFixed(4)}}</b>')"
                         onmouseout="hideTooltip()"/>`;
            }});
        }});

        html += `</g>`;
        svg.innerHTML = html;
    }}

    document.addEventListener('DOMContentLoaded', () => {{
        initConditionControls();
        renderGradientCurves();
    }});
    if (document.readyState !== 'loading') {{
        initConditionControls();
        renderGradientCurves();
    }}
    """

    full_html = _render_page_wrapper(title, body_html, data_js)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(full_html, encoding="utf-8")
    return full_html


def export_comparative_dashboard_html(
    spectral_data: SpectralTrajectoryData | Mapping[str, Any],
    gating_data: EntropicGatingData | Mapping[str, Any],
    gradient_data: GradientRecoveryData | Mapping[str, Any],
    output_path: str | Path | None = None,
    title: str = "ZVF Pilot Conformance Comparative Dashboard",
) -> str:
    """Generate single standalone HTML dashboard containing all three interactive visualizers."""
    spec_dict = spectral_data.to_dict() if isinstance(spectral_data, SpectralTrajectoryData) else dict(spectral_data)
    gate_dict = gating_data.to_dict() if isinstance(gating_data, EntropicGatingData) else dict(gating_data)
    grad_dict = gradient_data.to_dict() if isinstance(gradient_data, GradientRecoveryData) else dict(gradient_data)

    spec_json = json.dumps(spec_dict, indent=2)
    gate_json = json.dumps(gate_dict, indent=2)
    grad_json = json.dumps(grad_dict, indent=2)

    # Compute quick stats for KPIs
    max_spec_dist = float(np.max(spec_dict["distances"])) if spec_dict.get("distances") else 0.0
    pre_e = np.array(gate_dict.get("pre_gating_entropy", [[0]]))
    post_e = np.array(gate_dict.get("post_gating_entropy", [[0]]))
    entropy_red = float(np.mean(pre_e - post_e)) if pre_e.size > 0 else 0.0
    num_conds = len(grad_dict.get("curves", {}))

    body_html = f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <span class="label">Max Spectral Distance</span>
            <span class="value">{max_spec_dist:.4f}</span>
            <span class="subtext">Legendre Trajectory continuous L²</span>
        </div>
        <div class="kpi-card">
            <span class="label">Mean Entropy Reduction</span>
            <span class="value">{entropy_red:.4f}</span>
            <span class="subtext">Givens Planar Gating ΔH</span>
        </div>
        <div class="kpi-card">
            <span class="label">Evaluated Conditions</span>
            <span class="value">{num_conds}</span>
            <span class="subtext">GRPO Pilot Variants</span>
        </div>
    </div>

    <div class="tabs">
        <button id="btn-spectral" class="tab-btn active" onclick="switchTab('spectral')">Spectral Trajectories</button>
        <button id="btn-gating" class="tab-btn" onclick="switchTab('gating')">Entropic Gating Heatmap</button>
        <button id="btn-gradient" class="tab-btn" onclick="switchTab('gradient')">Gradient Recovery Curves</button>
    </div>

    <div id="tab-spectral" class="tab-content active">
        <div class="visualizer-card">
            <div class="card-header">
                <h2>Spectral Trajectory Continuous L² Distance Matrix <span class="badge">Legendre Modes</span></h2>
            </div>
            <div class="chart-container">
                <svg id="spectral-matrix-svg" viewBox="0 0 600 500"></svg>
            </div>
        </div>
    </div>

    <div id="tab-gating" class="tab-content">
        <div class="visualizer-card">
            <div class="card-header">
                <h2>Givens Planar Unitary Entropic Gating Heatmap <span class="badge">Pre vs Post Rotation</span></h2>
            </div>
            <div class="controls-row">
                <div class="control-group">
                    <label>View Mode:</label>
                    <select id="gating-view-mode" onchange="renderGatingHeatmap()">
                        <option value="side-by-side">Side-by-Side Comparison</option>
                        <option value="post-only">Post-Gating Projected</option>
                        <option value="difference">Entropy Reduction Density</option>
                    </select>
                </div>
            </div>
            <div class="chart-container">
                <svg id="gating-heatmap-svg" viewBox="0 0 900 450"></svg>
            </div>
        </div>
    </div>

    <div id="tab-gradient" class="tab-content">
        <div class="visualizer-card">
            <div class="card-header">
                <h2>Gradient Norm Retention & Recovery Curves <span class="badge">Comparative Analysis</span></h2>
            </div>
            <div class="controls-row" id="condition-checkboxes">
                <!-- Dynamic checkboxes -->
            </div>
            <div class="chart-container">
                <svg id="gradient-curves-svg" viewBox="0 0 900 400"></svg>
            </div>
        </div>
    </div>
    """

    data_js = f"""
    const spectralData = {spec_json};
    const gatingData = {gate_json};
    const gradientData = {grad_json};

    const colorPalette = {{
        'intended_full': '#38bdf8',
        'native_trl': '#f43f5e',
        'epsilon_only': '#fbbf24',
        'reduction_only': '#c084fc',
        'spectral_legendre': '#34d399',
        'entropic_givens': '#60a5fa'
    }};

    function renderSpectralMatrix() {{
        const svg = document.getElementById('spectral-matrix-svg');
        if (!svg) return;
        const matrix = spectralData.distances;
        const labels = spectralData.labels;
        const n = matrix.length;
        if (n === 0) return;

        let maxVal = 0;
        for (let i = 0; i < n; i++) {{
            for (let j = 0; j < n; j++) {{
                if (matrix[i][j] > maxVal) maxVal = matrix[i][j];
            }}
        }}
        if (maxVal === 0) maxVal = 1.0;

        const margin = 80;
        const cellSize = Math.min((500 - margin) / n, 50);
        const width = margin + n * cellSize + 20;
        const height = margin + n * cellSize + 20;

        let html = `<g transform="translate(${{margin}}, ${{margin}})">`;
        for (let i = 0; i < n; i++) {{
            for (let j = 0; j < n; j++) {{
                const val = matrix[i][j];
                const norm = val / maxVal;
                const r = Math.round(15 + norm * 200);
                const g = Math.round(189 - norm * 100);
                const b = Math.round(248 - norm * 50);
                const fill = `rgb(${{r}}, ${{g}}, ${{b}})`;

                html += `<rect x="${{j * cellSize}}" y="${{i * cellSize}}" width="${{cellSize - 2}}" height="${{cellSize - 2}}"
                         fill="${{fill}}" rx="3"
                         onmouseover="showTooltip(event, '${{labels[i]}} vs ${{labels[j]}}<br>Distance: <b>${{val.toFixed(4)}}</b>')"
                         onmouseout="hideTooltip()"/>`;
                html += `<text x="${{j * cellSize + cellSize / 2}}" y="${{i * cellSize + cellSize / 2 + 4}}"
                         font-size="10" fill="#ffffff" text-anchor="middle">${{val.toFixed(2)}}</text>`;
            }}
        }}
        for (let i = 0; i < n; i++) {{
            html += `<text x="-10" y="${{i * cellSize + cellSize / 2 + 4}}" font-size="11" fill="var(--text-muted)" text-anchor="end">${{labels[i]}}</text>`;
            html += `<text x="${{i * cellSize + cellSize / 2}}" y="-10" font-size="11" fill="var(--text-muted)" text-anchor="middle" transform="rotate(-30, ${{i * cellSize + cellSize / 2}}, -10)">${{labels[i]}}</text>`;
        }}
        html += `</g>`;
        svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
        svg.innerHTML = html;
    }}

    function renderGatingHeatmap() {{
        const svg = document.getElementById('gating-heatmap-svg');
        if (!svg) return;
        const pre = gatingData.pre_gating_entropy;
        const post = gatingData.post_gating_entropy;
        const rows = pre.length;
        const cols = pre[0].length;
        const modeElem = document.getElementById('gating-view-mode');
        const mode = modeElem ? modeElem.value : 'side-by-side';

        const cellW = Math.min(750 / cols, 35);
        const cellH = Math.min(350 / rows, 30);
        const margin = 60;

        let html = '';
        if (mode === 'side-by-side') {{
            const renderGrid = (matrix, titleText, offsetX) => {{
                let res = `<g transform="translate(${{offsetX}}, ${{margin}})">`;
                res += `<text x="${{(cols * cellW) / 2}}" y="-20" font-size="14" font-weight="600" fill="var(--accent-cyan)" text-anchor="middle">${{titleText}}</text>`;
                for (let r = 0; r < rows; r++) {{
                    for (let c = 0; c < cols; c++) {{
                        const val = matrix[r][c];
                        const alpha = Math.min(Math.max(val / 3.0, 0.05), 1.0);
                        const fill = `rgba(56, 189, 248, ${{alpha}})`;
                        res += `<rect x="${{c * cellW}}" y="${{r * cellH}}" width="${{cellW - 1}}" height="${{cellH - 1}}" fill="${{fill}}" rx="2"
                                 onmouseover="showTooltip(event, 'Row ${{r}}, Col ${{c}}<br>Entropy H(p): <b>${{val.toFixed(4)}}</b>')"
                                 onmouseout="hideTooltip()"/>`;
                    }}
                }}
                res += `</g>`;
                return res;
            }};
            html += renderGrid(pre, "Pre-Gating Entropy", 50);
            html += renderGrid(post, "Post-Gating (Givens Projected)", 50 + cols * cellW + 60);
        }} else {{
            const targetMatrix = mode === 'post-only' ? post : pre.map((rArr, rIdx) => rArr.map((v, cIdx) => v - post[rIdx][cIdx]));
            const label = mode === 'post-only' ? "Post-Gating Entropy Density" : "Entropy Reduction (ΔH)";
            html += `<g transform="translate(80, ${{margin}})">`;
            html += `<text x="${{(cols * cellW) / 2}}" y="-20" font-size="14" font-weight="600" fill="var(--accent-emerald)" text-anchor="middle">${{label}}</text>`;
            for (let r = 0; r < rows; r++) {{
                for (let c = 0; c < cols; c++) {{
                    const val = targetMatrix[r][c];
                    const alpha = Math.min(Math.max(val / 2.0, 0.05), 1.0);
                    const color = mode === 'post-only' ? `rgba(52, 211, 153, ${{alpha}})` : `rgba(244, 63, 94, ${{alpha}})`;
                    html += `<rect x="${{c * cellW}}" y="${{r * cellH}}" width="${{cellW - 1}}" height="${{cellH - 1}}" fill="${{color}}" rx="2"
                             onmouseover="showTooltip(event, 'Row ${{r}}, Col ${{c}}<br>Value: <b>${{val.toFixed(4)}}</b>')"
                             onmouseout="hideTooltip()"/>`;
                }}
            }}
            html += `</g>`;
        }}
        svg.innerHTML = html;
    }}

    function initConditionControls() {{
        const container = document.getElementById('condition-checkboxes');
        if (!container) return;
        let html = '<span style="font-weight:600;margin-right:8px;">Conditions:</span>';
        for (const cond of Object.keys(gradientData.curves)) {{
            const color = colorPalette[cond] || '#94a3b8';
            html += `<label style="display:inline-flex;align-items:center;gap:4px;margin-right:12px;cursor:pointer;">
                        <input type="checkbox" value="${{cond}}" checked onchange="renderGradientCurves()">
                        <span style="color:${{color}};font-weight:600;">${{cond}}</span>
                     </label>`;
        }}
        container.innerHTML = html;
    }}

    function renderGradientCurves() {{
        const svg = document.getElementById('gradient-curves-svg');
        if (!svg) return;
        const steps = gradientData.steps;
        const curves = gradientData.curves;
        const activeConds = Array.from(document.querySelectorAll('#condition-checkboxes input:checked')).map(cb => cb.value);

        if (steps.length === 0 || activeConds.length === 0) {{
            svg.innerHTML = '<text x="450" y="200" text-anchor="middle" fill="var(--text-muted)">Select at least one condition to display curves.</text>';
            return;
        }}

        const margin = {{top: 40, right: 40, bottom: 50, left: 60}};
        const width = 900 - margin.left - margin.right;
        const height = 400 - margin.top - margin.bottom;

        const minStep = steps[0];
        const maxStep = steps[steps.length - 1];

        let maxVal = 0;
        activeConds.forEach(cond => {{
            curves[cond].forEach(v => {{ if (v > maxVal) maxVal = v; }});
        }});
        if (maxVal === 0) maxVal = 1.0;

        const xScale = (step) => margin.left + ((step - minStep) / (maxStep - minStep || 1)) * width;
        const yScale = (val) => margin.top + height - (val / maxVal) * height;

        let html = `<g>`;
        html += `<line x1="${{margin.left}}" y1="${{margin.top + height}}" x2="${{margin.left + width}}" y2="${{margin.top + height}}" stroke="var(--card-border)" stroke-width="2"/>`;
        html += `<line x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{margin.top + height}}" stroke="var(--card-border)" stroke-width="2"/>`;

        for (let i = 0; i <= 5; i++) {{
            const yVal = (maxVal * i) / 5;
            const yPos = yScale(yVal);
            html += `<line x1="${{margin.left - 5}}" y1="${{yPos}}" x2="${{margin.left + width}}" y2="${{yPos}}" stroke="var(--card-border)" stroke-dasharray="4" opacity="0.4"/>`;
            html += `<text x="${{margin.left - 10}}" y="${{yPos + 4}}" font-size="10" fill="var(--text-muted)" text-anchor="end">${{yVal.toFixed(2)}}</text>`;
        }}

        steps.forEach((step, idx) => {{
            if (idx % Math.ceil(steps.length / 8) === 0 || idx === steps.length - 1) {{
                const xPos = xScale(step);
                html += `<text x="${{xPos}}" y="${{margin.top + height + 20}}" font-size="10" fill="var(--text-muted)" text-anchor="middle">Step ${{step}}</text>`;
            }}
        }});

        activeConds.forEach(cond => {{
            const color = colorPalette[cond] || '#94a3b8';
            const values = curves[cond];
            let pathD = '';

            values.forEach((v, idx) => {{
                const x = xScale(steps[idx]);
                const y = yScale(v);
                if (idx === 0) pathD += `M ${{x}} ${{y}}`;
                else pathD += ` L ${{x}} ${{y}}`;
            }});

            html += `<path d="${{pathD}}" fill="none" stroke="${{color}}" stroke-width="3.5" opacity="0.9"/>`;

            values.forEach((v, idx) => {{
                const x = xScale(steps[idx]);
                const y = yScale(v);
                html += `<circle cx="${{x}}" cy="${{y}}" r="4" fill="${{color}}"
                         onmouseover="showTooltip(event, 'Condition: <b>${{cond}}</b><br>Step: <b>${{steps[idx]}}</b><br>Retention/Norm: <b>${{v.toFixed(4)}}</b>')"
                         onmouseout="hideTooltip()"/>`;
            }});
        }});

        html += `</g>`;
        svg.innerHTML = html;
    }}

    function initAll() {{
        renderSpectralMatrix();
        renderGatingHeatmap();
        initConditionControls();
        renderGradientCurves();
    }}

    document.addEventListener('DOMContentLoaded', initAll);
    if (document.readyState !== 'loading') initAll();
    """

    full_html = _render_page_wrapper(title, body_html, data_js)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(full_html, encoding="utf-8")
    return full_html
