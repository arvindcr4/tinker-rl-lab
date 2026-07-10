# ai-scientist-v2-integration/ — INDEX

**Purpose:** Patches + launcher that make AI-Scientist-v2 (BFTS tree-search variant) run on this repo with Tinker/TRL GRPO templates.

**Key files:**
- `README.md` — what each patch fixes (agent prompt, exec templates, timeouts, deps) + how to launch.
- `patch.sh` — copies modified files into `~/ai-scientist-v2` (agent, idea templates, config).
- `launch.sh` — convenience launcher for `launch_scientist_bfts.py` (default idea = tinker template).
- `bfts_config.yaml` — BFTS run config; `exec.timeout` raised to 7200s for Tinker latency.

**Subfolders:**
- `ai_scientist/` — mirrored/patched AI-Scientist-v2 source: `ideas/` templates + `treesearch/` agent (see its INDEX.md).

**Find it fast:**
- to apply the integration → `patch.sh`
- to launch a run → `launch.sh`
- to change run timeout/report models → `bfts_config.yaml`
