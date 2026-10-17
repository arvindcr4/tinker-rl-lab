"""Durable, transactional request ledger for a single shared SQLite filesystem.

Opening never initializes/reset spending. Provisioning/import of reconciled
history is explicit. Deployments without one shared durable transactional store
must not launch paid work. Modal Dict is not an authoritative financial store.
Amounts are integer nanodollars; estimates are not provider-billed charges.
"""

from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

SCALE = Decimal(1_000_000_000)


def units(value):
    if isinstance(value, bool):
        raise ValueError("cost cannot be boolean")
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise ValueError("cost must be finite and nonnegative")
    return int((amount * SCALE).to_integral_value(rounding=ROUND_CEILING))


class CampaignLedger:
    def __init__(self, path: str | Path, *, campaign: str):
        self.path, self.campaign = Path(path), campaign
        if not self.path.is_file():
            raise RuntimeError("established campaign ledger missing; reconcile before launch")
        with self._transaction() as db:
            row = db.execute("SELECT campaign FROM authorization").fetchone()
            if row != (campaign,):
                raise RuntimeError("campaign ledger identity mismatch")

    @classmethod
    def provision(cls, path, *, campaign, maximum_usd, reconciled_spend_usd):
        """Administrative-only explicit initialization, never called by serving."""
        path = Path(path)
        if path.exists():
            raise RuntimeError("refusing to replace established ledger")
        db = sqlite3.connect(path)
        try:
            db.executescript("""CREATE TABLE authorization(campaign TEXT PRIMARY KEY, maximum INTEGER NOT NULL, prior INTEGER NOT NULL);
            CREATE TABLE requests(id TEXT PRIMARY KEY, identity TEXT NOT NULL, state TEXT NOT NULL, ceiling INTEGER NOT NULL, estimate INTEGER NOT NULL, billed INTEGER);
            """)
            maximum, prior = units(maximum_usd), units(reconciled_spend_usd)
            if prior > maximum:
                raise ValueError("reconciled spend exceeds authorization")
            db.execute("INSERT INTO authorization VALUES(?,?,?)", (campaign, maximum, prior))
            db.commit()
        finally:
            db.close()
        return cls(path, campaign=campaign)

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(
            f"file:{self.path}?mode=rw", uri=True, timeout=30, isolation_level=None
        )
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def reserve(self, request_id, projected_usd, *, identity):
        ceiling = units(projected_usd)
        if not request_id or not identity:
            raise ValueError("request and immutable execution identity required")
        with self._transaction() as db:
            old = db.execute(
                "SELECT identity,ceiling FROM requests WHERE id=?", (request_id,)
            ).fetchone()
            if old:
                if old != (identity, ceiling):
                    raise RuntimeError("idempotency key reused with different request")
                raise RuntimeError("request already reserved; reconcile instead of sampling again")
            maximum, prior = db.execute("SELECT maximum,prior FROM authorization").fetchone()
            committed = db.execute(
                "SELECT COALESCE(SUM(CASE WHEN billed IS NULL THEN ceiling ELSE billed END),0) FROM requests"
            ).fetchone()[0]
            if prior + committed + ceiling > maximum:
                raise RuntimeError("campaign budget exhausted")
            db.execute(
                "INSERT INTO requests VALUES(?,?, 'RESERVED',?,0,NULL)",
                (request_id, identity, ceiling),
            )
        return request_id

    def settle(self, request_id, estimated_usd):
        estimate = units(estimated_usd)
        with self._transaction() as db:
            row = db.execute(
                "SELECT state,ceiling,estimate FROM requests WHERE id=?", (request_id,)
            ).fetchone()
            if row is None:
                raise RuntimeError("unknown reservation")
            if row[0] in {"SAMPLED", "SETTLED"}:
                if row[2] != estimate:
                    raise RuntimeError("conflicting settlement")
                return
            if estimate > row[1]:
                raise RuntimeError("actual work exceeded reserved ceiling; reconciliation required")
            # Keep the entire ceiling until provider billing is final.
            db.execute(
                "UPDATE requests SET state='SAMPLED',estimate=? WHERE id=?", (estimate, request_id)
            )

    def fail(self, request_id):
        with self._transaction() as db:
            if db.execute("SELECT 1 FROM requests WHERE id=?", (request_id,)).fetchone() is None:
                raise RuntimeError("unknown reservation")
            db.execute(
                "UPDATE requests SET state='FAILED_BILLING_PENDING' WHERE id=? AND state='RESERVED'",
                (request_id,),
            )

    def reconcile(self, request_id, provider_billed_usd):
        """Called only after independently verified final provider billing."""
        billed = units(provider_billed_usd)
        with self._transaction() as db:
            row = db.execute(
                "SELECT state,billed FROM requests WHERE id=?", (request_id,)
            ).fetchone()
            if row is None or row[0] == "RESERVED":
                raise RuntimeError("request not ready for reconciliation")
            if row[1] is not None and row[1] != billed:
                raise RuntimeError("conflicting final billing")
            db.execute(
                "UPDATE requests SET state='SETTLED',billed=? WHERE id=?", (billed, request_id)
            )

    def snapshot(self):
        with self._transaction() as db:
            maximum, prior = db.execute("SELECT maximum,prior FROM authorization").fetchone()
            billed, pending, estimated = db.execute(
                "SELECT COALESCE(SUM(billed),0),COALESCE(SUM(CASE WHEN billed IS NULL THEN ceiling ELSE 0 END),0),COALESCE(SUM(estimate),0) FROM requests"
            ).fetchone()
        return {
            "maximum_usd": str(Decimal(maximum) / SCALE),
            "charged_usd": str(Decimal(prior + billed) / SCALE),
            "reserved_usd": str(Decimal(pending) / SCALE),
            "estimated_usd": str(Decimal(estimated) / SCALE),
        }
