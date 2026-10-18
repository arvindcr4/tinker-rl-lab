"""Opt-in bounded public inference. Static evidence is the safe default.

For a multi-replica deployment, use one shared transactional quota service rather
than separate SQLite files. A configured DB must live on persistent storage.
"""

import os
import sqlite3
from pathlib import Path


def reserve_call(prompt, *, environ=None):
    env = os.environ if environ is None else environ
    if env.get("LIVE_INFERENCE_POLICY") != "bounded_public":
        return "live_inference_disabled"
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4096:
        return "invalid_input_size"
    try:
        limit = int(env.get("LIVE_TOTAL_CALL_LIMIT", "0"))
        db_path = Path(env.get("LIVE_BUDGET_DB", ""))
        if not 0 < limit <= 100 or not db_path.is_absolute() or not db_path.parent.is_dir():
            return "quota_not_configured"
        with sqlite3.connect(str(db_path), timeout=5, isolation_level=None) as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY CHECK(id=1), used INTEGER NOT NULL)"
            )
            db.execute("INSERT OR IGNORE INTO budget VALUES (1, 0)")
            used = db.execute("SELECT used FROM budget WHERE id=1").fetchone()[0]
            if type(used) is not int or used < 0:
                db.execute("ROLLBACK")
                return "quota_unavailable"
            if used >= limit:
                db.execute("ROLLBACK")
                return "shared_call_limit_reached"
            db.execute("UPDATE budget SET used=used+1 WHERE id=1")
            db.execute("COMMIT")
    except (OSError, ValueError, sqlite3.Error):
        return "quota_unavailable"
    return None
