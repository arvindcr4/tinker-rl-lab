"""Seed tau2's native document-embedding cache from the original index02 bundle (no re-embedding).

The native cache is valid only if its stored doc_ids list equals the current KnowledgeBase iteration
order (filesystem glob order), otherwise tau2 deletes it and re-embeds. This permutes the original 698
text-embedding-3-large vectors (by document id, content hash checked) into the current order and writes
them via the native EmbeddingsCache.put, so the cache key is computed by native code.
Run with the tau2 venv python, cwd = run directory (native cache path is ./data/.embeddings_cache).
"""
import hashlib
import pickle
import sys

import numpy as np

from tau2.domains.banking_knowledge.environment import get_knowledge_base  # noqa
from tau2.knowledge.embeddings_cache import EmbeddingsCache

SRC = sys.argv[1]  # original index02 .pkl
kb = get_knowledge_base()
docs = [{"id": d.id, "text": d.content, "title": d.title} for d in kb.documents.values()]
orig = pickle.load(open(SRC, "rb"))
pos = {i: n for n, i in enumerate(orig["doc_ids"])}
assert len(docs) == len(orig["doc_ids"]) == 698 and set(pos) == {d["id"] for d in docs}
emb = np.asarray(orig["embeddings"])[[pos[d["id"]] for d in docs]]
assert emb.shape == (698, 3072) and np.isfinite(emb).all()
cache = EmbeddingsCache()
cache.put(docs, "openai", emb, [d["id"] for d in docs], {"model": "text-embedding-3-large"})
got = cache.get(docs, "openai", {"model": "text-embedding-3-large"})
assert got is not None and got[1] == [d["id"] for d in docs]
print("seeded", list(cache.metadata), "reordered:", orig["doc_ids"] != got[1],
      "orig_pkl_sha256", hashlib.sha256(open(SRC, "rb").read()).hexdigest())
