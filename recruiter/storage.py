"""SQLite persistence for jobs, candidates, analysis runs, human reviews and experiments."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from recruiter.schemas import HumanReview, ShortlistReport, utcnow

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "recruiter.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candidates (
    candidate_key TEXT PRIMARY KEY,          -- run_id:candidate_id
    run_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    source_file TEXT,
    content_hash TEXT,
    profile_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(job_id),
    provider TEXT NOT NULL,
    n_candidates INTEGER NOT NULL,
    report_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    run_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    reviewer TEXT,
    notes TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, candidate_id)
);
CREATE TABLE IF NOT EXISTS review_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    reviewer TEXT,
    notes TEXT,
    at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider TEXT NOT NULL,
    config_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: str | Path = DEFAULT_DB):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False) if str(path) == ":memory:" else None
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = self._memory_conn or sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            if self._memory_conn is None:
                conn.close()

    # ------------------------------------------------------------------ runs
    def save_run(self, report: ShortlistReport, job_text: str, resume_texts: dict[str, str] | None = None) -> str:
        job_id = hashlib.sha1(job_text.encode("utf-8")).hexdigest()[:12]
        resume_texts = resume_texts or {}
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?)",
                (job_id, report.job.title, job_text, report.job.model_dump_json(), utcnow()),
            )
            for c in report.candidates:
                digest = hashlib.sha1(resume_texts.get(c.candidate_id, "").encode()).hexdigest()[:16]
                conn.execute(
                    "INSERT OR REPLACE INTO candidates VALUES (?,?,?,?,?,?,?)",
                    (f"{report.run_id}:{c.candidate_id}", report.run_id, c.candidate_id, c.display_name,
                     c.source_file, digest, c.model_dump_json()),
                )
            conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?)",
                (report.run_id, job_id, report.provider, len(report.candidates), report.model_dump_json(), report.created_at),
            )
        return report.run_id

    def list_runs(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT r.run_id, j.title, r.provider, r.n_candidates, r.created_at FROM runs r "
                "JOIN jobs j ON j.job_id = r.job_id ORDER BY r.created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def load_run(self, run_id: str) -> ShortlistReport | None:
        with self._conn() as conn:
            row = conn.execute("SELECT report_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return ShortlistReport.model_validate_json(row["report_json"]) if row else None

    def delete_run(self, run_id: str) -> None:
        with self._conn() as conn:
            for table in ("runs", "candidates", "reviews", "review_audit"):
                conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))

    # --------------------------------------------------------------- reviews
    def save_review(self, run_id: str, review: HumanReview) -> None:
        review.updated_at = utcnow()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO reviews VALUES (?,?,?,?,?,?)",
                (run_id, review.candidate_id, review.decision, review.reviewer, review.notes, review.updated_at),
            )
            conn.execute(
                "INSERT INTO review_audit (run_id, candidate_id, decision, reviewer, notes, at) VALUES (?,?,?,?,?,?)",
                (run_id, review.candidate_id, review.decision, review.reviewer, review.notes, review.updated_at),
            )

    def reviews(self, run_id: str) -> dict[str, HumanReview]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM reviews WHERE run_id = ?", (run_id,)).fetchall()
        return {r["candidate_id"]: HumanReview(candidate_id=r["candidate_id"], decision=r["decision"],
                                               reviewer=r["reviewer"] or "", notes=r["notes"] or "",
                                               updated_at=r["updated_at"]) for r in rows}

    def audit_log(self, run_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM review_audit WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
        return [dict(r) for r in rows]

    # ----------------------------------------------------------- experiments
    def save_experiment(self, experiment_id: str, name: str, provider: str, config: dict, metrics: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO experiments VALUES (?,?,?,?,?,?)",
                (experiment_id, name, provider, json.dumps(config), json.dumps(metrics), utcnow()),
            )

    def experiments(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["config"] = json.loads(d.pop("config_json"))
            d["metrics"] = json.loads(d.pop("metrics_json"))
            out.append(d)
        return out
