# E6 small-scale (substitute: MiniWoB++)

WebBench access is blocked and WebArena needs AWS quota, so this is a **substitute**: MiniWoB++ (miniwob 1.1.0),
12 tasks x 3 episodes = 36 episodes, base Qwen3.6-35B-A3B on Tinker (non-thinking, temp 0, max_tokens 256).

Result: **15/36 = 0.417** episode success (Wilson95 0.271-0.578). Per-task breakdown in `result.json`.

What ran:
- Tasks: `random.Random(20260926).sample(sorted(miniwob ids), 12)`; episode seeds 0,1,2 (`raw/selection.json`).
- Agent (`code/run_miniwob.py`): each step sends the utterance, previous actions and a text DOM dump (ref, tag, text, bbox)
  and gets one action: CLICK(ref) / TYPE(ref,"text") / CLICK_XY(x,y) / KEY("k"). Max 10 steps. Success = native reward > 0.
- Timers disabled via a patched copy of the MiniWoB html (`code/html`, EPISODE_MAX_TIME=1e6), because LLM latency
  exceeds the 10 s wall clock. The aborted native-timer run is kept in `raw/aborted_run1_10s_timer/` and is not scored.
- Actor calls go through `code/tinker_shim.py` (local HTTP -> Tinker), which logs usage to `raw/shim_usage.jsonl`.

Rerun (from repo root):
```
set -a; source .env; set +a
SHIM_LOG=$PWD/outputs/e1_e14_small_scale_2026-09-26/E6/raw/shim_usage.jsonl \
  ~/.local/share/uv/tools/tinker/bin/python outputs/e1_e14_small_scale_2026-09-26/E6/code/tinker_shim.py 18769 &
SHIM_URL=http://127.0.0.1:18769/chat .venv/bin/python outputs/e1_e14_small_scale_2026-09-26/E6/code/run_miniwob.py outputs/e1_e14_small_scale_2026-09-26/E6/raw
```
Caveat: there is no drag or select primitive, so drag-box, draw-line and choose-list score 0/9. This is not a WebBench score.
