---
status: accepted
date: 2026-07-11
---

# Deep research-engineering modules own high-risk behavior

Experiment entry points, submission audits, and paper variants remain discoverable at their historical paths, but those paths are compatibility adapters only. The GRPO run module owns paid sampling, optimization, telemetry, and checkpoint manifests; the audit runner owns structured audit results and exit policy; and the figure module owns renderer selection, results adapters, fallback provenance, and figure manifests. We chose this over deleting old paths or preserving independent copies because external notebooks and paper build instructions still use those paths, while duplicated implementations had already allowed resume, import, and provenance behavior to drift.
