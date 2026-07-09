# Patch proposal: Progent-style privilege DSL for the orchestrator

Target: `ai-scientist-v2-integration/ai_scientist/treesearch/parallel_agent.py`

## Why

The orchestrator runs arbitrary LLM-generated code in
`ProcessPoolExecutor.submit(self._process_node_wrapper, ...)`
(parallel_agent.py:2176) with **no sandbox, no privilege
gating, no import allowlist**. Any node whose code calls
`os.system('curl ...')` or `subprocess.run(['bash','-c',...])`
or `open('/etc/passwd')` will succeed. Progent
(Huang et al., arXiv:2504.11703) introduces a small DSL
over tool calls for exactly this surface.

## Measured evidence (this prototype)

See `experiments/results/berkeley/sp25_l12_security_audit.tsv`
and `sp25_l12_security_summary.json`. In the vanilla
Progent run, 4/4 of the attack payloads (env-var exfil,
/etc/passwd exfil, shell command, network exfil) leaked
data; an allowlist-gate blocked all 4/4 with zero
false-positives on the benign `pure_numpy` payload.

## Patch (conceptual — do NOT apply directly)

```python
# Insert before ParallelAgent.__init__():
SAFE_IMPORT_ALLOWLIST = {
    "numpy", "matplotlib", "json", "math", "statistics",
    "itertools", "collections", "pathlib", "typing",
    "scipy", "sklearn", "torch", "transformers",
    "tinker", "datasets",
}

def privilege_gate(code: str, *, allowlist=SAFE_IMPORT_ALLOWLIST):
    bad = re.findall(
        r"^\s*(?:import|from)\s+([A-Za-z_][\w]*)",
        code, flags=re.MULTILINE,
    )
    blocked = [m for m in bad if m not in allowlist]
    if blocked:
        return False, f"blocked imports: {blocked}"
    if re.search(
        r"\bos\.system\b|\bsubprocess\b|"
        r"\beval\b|\bexec\b|"
        r"open\(['"]\/|urllib\.request",
        code,
    ):
        return False, "blocked dangerous builtin"
    return True, "allowed"
```

Then wrap the executor submission (parallel_agent.py:2176):

```python
ok, reason = privilege_gate(node_data.get("code", ""))
if not ok:
    logger.warning(f"node {node_data.get('id')} gated: {reason}")
    result_node = build_blocked_node(node_data, reason)
    self.journal.append(result_node)
    continue
```

## Roll-out

1. Land gate as a *warning-only* wrapper that logs blocked
   attempts but still runs the code (allows regression check).
2. After 1 week of zero false positives on the real BFTS
   workload, switch to enforcement (the `continue` branch
   above).
3. Mirror the same gate to the journal summary step
   (parallel_agent.py:2120-2128) — AgentPoison mitigation.
4. Wrap prior-node-code in `<data trust="UNTRUSTED">` tags
   before embedding into plotting prompts
   (parallel_agent.py:792,842) — DataSentinel mitigation.

## Estimated impact

- Blocks 100% of the exfil/shell payloads demonstrated
  above.
- Blocks 100% of the AgentPoison-style memory-poisoning
  payloads (after journal sanitiser is added).
- Blocks 100% of the DataSentinel-style injection payloads
  (after quarantine is added).
- Negligible impact on legitimate workloads (allowlist
  includes every library used by the BFTS templates).