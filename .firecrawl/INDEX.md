# .firecrawl/ — INDEX

**Purpose:** Cached raw Firecrawl web-search result dumps gathered while researching ZVF (zero-variance fraction) diagnostics and adversarial/reviewer-review practices for the paper.

**Contents (dump):** 5 JSON files `zvf-search.json` … `zvf-search5.json`, each a Firecrawl `/search` response `{success, data:{web:[...]}, id, creditsUsed}` with ~10 web results (url/title/description/markdown). Queries were noisy keyword searches (many off-topic "four pillars"/"zero-shot vision" hits; `zvf-search5.json` = adversarial code-review results). Treat as scratch research inputs, not curated sources.

**Find it fast:**
- to see adversarial-review search hits → `zvf-search5.json`
- to pull a result's full page markdown → `data.web[i].markdown` in any file
