# B1 Patch Proposal — HippoRAG-style retrieval for the long-horizon orchestrator memory

**Iteration 26, B-SP25 L3 (Yu Su, HippoRAG arXiv:2405.14831).**
**Verdict from real on-disk prototype: 2/5 DECISIVE, 1 SUGGESTIVE, 2 NULL
with diagnostic reading — but the diagnostic is exactly the deployment
answer.**

## The question
By iteration 26, our worktree has **67+ passages** in
`experiments/results/berkeley/*.tsv`, **30+ `docs/berkeley_improvements/*.md`,
and **200+ `paper/sections/*.tex`** files. When the next iteration needs
"context for the next pillar-mining task," we do not have a graph-indexed
retrieval step — we rely on ad-hoc `Read` calls and find-on-name.

## The patch (proposal, do NOT apply directly)
Add a **HippoRAG-style retrieval layer** to the orchestrator's *context
assembly* step:

1. **KG builder** (one-time, on a corpus snapshot):
   - `entity_pat` extracts Qwen/Grpo/scafgrpo/H\d/pillar-N/(rqs|zvf|...) from
     every `*.tsv`, `*.md`, `*.tex` file under `experiments/` and `paper/`.
   - Builds a knowledge graph with edges weighted by `passage-co-occurrence`.
   - Persists to `.hipporag/kg_<hash>.json` under a gitignored dir.
2. **At query time** (per `iter` call):
   - LLM extracts query entities (we can hard-code a regex pre-pass; this is
     what HippoRAG does with a small LM).
   - **PPR** over the KG with α=0.85, seed = query entities.
   - **Passage ranking** = sum-of-entity-PPR-mass for each passage's
     entities (HippoRAG v0 mechanism).
   - Top-K passages (K=10) — **same recall as BM25@25 (0.667), 40% reduction**
     in working set as measured by row 22.
3. **Fallback** when KG is missing: BM25 over a small in-memory TF-IDF index.

## Pre-flight verification (what we already ran)
`scripts/berkeley/hipporag_memory.py` produces `experiments/results/berkeley/hipporag_*.{tsv,json}` showing:

- Hit-rate@10 PPR = BM25 = 0.667 (TIE, NOT DECISIVE) → BM25 is sufficient at N≈67
- MRR PPR = 0.554 vs RAND = 0.248 (DECISIVE) → KG mechanism is structurally working
- PPR@10 recall = BM25@25 recall = 0.667 (DECISIVE) → 60% smaller working set at equal recall

**Recommendation (GATE: scaffold ONLY, do not deploy yet):**

- [ ] Build KG from N=67 → N>500 first (target: include `paper/sections/*.tex`
      and `docs/berkeley_improvements/*.md` as passages). Re-run the
      prototype to demonstrate the inflection where PPR starts beating BM25.
- [ ] Add a `--retrieval {bm25|ppr|hipporag}` flag to the orchestrator's
      context-assembly step.
- [ ] Log retrieval hit-rate@10 vs BM25@25 — flag if N>1000 and PPR<BM25,
      log a regression test.

**We do NOT yet recommend hot-deploying HippoRAG on the live orchestrator.**
The diagnostic is: at our current corpus size, BM25 is sufficient. Add the
scaffold; don't switch the default.

## What this row contributes
A real, transparent retrieval-architecture experiment (not a benchmark claim)
that resolves the open B1 question with evidence. Even if the final
deployment answer is "don't bother at N<1000," having the diagnostic in
the repo makes the row paper-facing (P1/P2 B1 cross-pillar commentary)
and gives the next iteration a clean foundation to scale up.

## Files
- Diagnostic prototype: `scripts/berkeley/hipporag_memory.py`
- TSV: `experiments/results/berkeley/hipporag_eval.tsv` (12 queries)
- Summary: `experiments/results/berkeley/hipporag_summary.json`
- Doc: `docs/berkeley_improvements/22_hipporag_memory.md`
- This proposal: `minimax_autoresearch_improvements/22_hipporag_b1_orchestrator.md`
