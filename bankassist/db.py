"""SQLite support database: support_tickets, agent_logs."""
from __future__ import annotations

import json
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "support.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS support_tickets (
    ticket_id     TEXT PRIMARY KEY,           -- unique 6-digit number
    customer_name TEXT NOT NULL,
    message       TEXT NOT NULL,
    category      TEXT,
    status        TEXT NOT NULL DEFAULT 'Unresolved',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_logs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT NOT NULL,
    customer_name  TEXT,
    message        TEXT,
    classification TEXT,
    confidence     REAL,
    agent_path     TEXT,
    response       TEXT,
    ticket_id      TEXT,
    ticket_action  TEXT,
    success        INTEGER,
    latency_ms     REAL,
    prompt_trace   TEXT,
    user_rating    INTEGER,           -- +1 / -1 from the UI feedback loop
    corrected_label TEXT
);
"""

SEED = [
    ("650932", "Priya Sharma", "Net banking login OTP not received", "net banking", "Resolved", 9),
    ("784521", "Rahul Verma", "Debit card replacement still hasn't arrived", "debit card", "In Progress", 4),
    ("312874", "Anita Desai", "Charged twice for the same UPI payment", "UPI payment", "Unresolved", 2),
    ("459013", "Vikram Singh", "Loan EMI deducted on the wrong date", "loan", "Resolved", 15),
    ("901245", "Meera Iyer", "Credit card limit increase request pending", "credit card", "In Progress", 6),
    ("128456", "Arjun Nair", "Cheque book request not processed", "cheque", "Closed", 30),
    ("567890", "Sneha Kapoor", "Mobile app crashes during fund transfer", "mobile app", "Unresolved", 1),
]


class SupportDB:
    def __init__(self, path: str | Path = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
            if c.execute("SELECT COUNT(*) FROM support_tickets").fetchone()[0] == 0:
                self.seed(c)

    def _conn(self):
        return sqlite3.connect(self.path)

    def seed(self, c):
        now = datetime.now()
        for tid, name, msg, cat, status, days in SEED:
            created = (now - timedelta(days=days)).isoformat(timespec="seconds")
            c.execute("INSERT INTO support_tickets VALUES (?,?,?,?,?,?,?)", (tid, name, msg, cat, status, created, created))

    # ------------------------------------------------------------- tickets --
    def ticket_exists(self, ticket_id: str) -> bool:
        with self._conn() as c:
            return c.execute("SELECT 1 FROM support_tickets WHERE ticket_id=?", (ticket_id,)).fetchone() is not None

    def new_ticket_id(self) -> str:
        while True:
            tid = str(random.randint(100000, 999999))
            if not self.ticket_exists(tid):
                return tid

    def create_ticket(self, customer_name: str, message: str, category: str) -> str:
        tid, now = self.new_ticket_id(), datetime.now().isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute("INSERT INTO support_tickets VALUES (?,?,?,?,?,?,?)",
                      (tid, customer_name, message, category, "Unresolved", now, now))
        return tid

    def get_ticket(self, ticket_id: str) -> dict | None:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            row = c.execute("SELECT * FROM support_tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
        return dict(row) if row else None

    def update_status(self, ticket_id: str, status: str):
        with self._conn() as c:
            c.execute("UPDATE support_tickets SET status=?, updated_at=? WHERE ticket_id=?",
                      (status, datetime.now().isoformat(timespec="seconds"), ticket_id))

    def tickets_df(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql("SELECT * FROM support_tickets ORDER BY created_at DESC", c)

    def tickets_for(self, customer_name: str) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql("SELECT * FROM support_tickets WHERE lower(customer_name)=lower(?) ORDER BY created_at DESC",
                               c, params=(customer_name,))

    # ---------------------------------------------------------------- logs --
    def log(self, rec: dict) -> int:
        rec = {**rec, "prompt_trace": json.dumps(rec.get("prompt_trace", []))}
        cols = ["ts", "customer_name", "message", "classification", "confidence", "agent_path", "response",
                "ticket_id", "ticket_action", "success", "latency_ms", "prompt_trace"]
        with self._conn() as c:
            cur = c.execute(f"INSERT INTO agent_logs ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                            [rec.get(k) for k in cols])
            return cur.lastrowid

    def rate(self, log_id: int, rating: int, corrected_label: str | None = None):
        with self._conn() as c:
            c.execute("UPDATE agent_logs SET user_rating=?, corrected_label=? WHERE id=?", (rating, corrected_label, log_id))

    def logs_df(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql("SELECT * FROM agent_logs ORDER BY id DESC", c)

    def reset(self):
        self.path.unlink(missing_ok=True)
        self.__init__(self.path)
