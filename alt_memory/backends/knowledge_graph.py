"""Temporal knowledge graph - SQLite-backed entity-relationship store."""

import json
import logging
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from alt_memory.config import sanitize_iso_temporal

logger = logging.getLogger(__name__)




def _is_date_only_temporal(value: str) -> bool:
    return isinstance(value, str) and len(value) == 10 and value[4] == "-" and value[7] == "-"


def _temporal_start_key(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if _is_date_only_temporal(value):
        return f"{value}T00:00:00Z"
    return value


def _temporal_end_key(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if _is_date_only_temporal(value):
        return f"{value}T23:59:59Z"
    return value


_TEMPORAL_COLUMN_WHITELIST = {"t.valid_from", "t.valid_to", "valid_from", "valid_to"}


def _sql_temporal_start_expr(column: str) -> str:
    if column not in _TEMPORAL_COLUMN_WHITELIST:
        raise ValueError(f"Invalid column name: {column!r}")
    return (
        f"CASE WHEN length({column}) = 10 "
        f"AND substr({column}, 5, 1) = '-' "
        f"AND substr({column}, 8, 1) = '-' "
        f"THEN {column} || 'T00:00:00Z' ELSE {column} END"
    )


def _sql_temporal_end_expr(column: str) -> str:
    if column not in _TEMPORAL_COLUMN_WHITELIST:
        raise ValueError(f"Invalid column name: {column!r}")
    return (
        f"CASE WHEN length({column}) = 10 "
        f"AND substr({column}, 5, 1) = '-' "
        f"AND substr({column}, 8, 1) = '-' "
        f"THEN {column} || 'T23:59:59Z' ELSE {column} END"
    )


def _temporal_filter_sql(as_of: str) -> tuple[str, list[str]]:
    as_of_key = _temporal_start_key(as_of)
    valid_from_expr = _sql_temporal_start_expr("t.valid_from")
    valid_to_expr = _sql_temporal_end_expr("t.valid_to")
    return (
        f" AND (t.valid_from IS NULL OR {valid_from_expr} <= ?) "
        f"AND (t.valid_to IS NULL OR {valid_to_expr} >= ?)",
        [as_of_key, as_of_key],
    )


class KnowledgeGraph:
    """Lightweight temporal knowledge graph."""

    def __init__(self, path: str):
        self._db_path = Path(path)
        if self._db_path.suffix == ".sqlite3" or self._db_path.suffix == ".db":
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._db_path = self._db_path / "knowledge.db"
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = None
        self._lock = threading.Lock()
        self._conn_lock = threading.Lock()
        self._closed = False
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("""CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL,
                valid_from TEXT, valid_to TEXT, source TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')))""")
        conn.execute("""CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT DEFAULT 'unknown',
                properties TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_predicate ON facts(predicate)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_object ON facts(object)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")
        self._migrate_schema(conn)
        conn.commit()
        self._connection = conn

    _MIGRATE_COL_WHITELIST = {
        "confidence": "REAL DEFAULT 1.0",
        "source_node": "TEXT",
        "source_file": "TEXT",
        "source_entity_id": "TEXT",
        "adapter_name": "TEXT",
    }

    def _migrate_schema(self, conn) -> None:
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(facts)").fetchall()}
        for col, dtype in self._MIGRATE_COL_WHITELIST.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE facts ADD COLUMN {col} {dtype}")
        if "source_drawer_id" in existing and "source_entity_id" not in existing:
            conn.execute("ALTER TABLE facts ADD COLUMN source_entity_id TEXT")
            conn.execute("UPDATE facts SET source_entity_id = source_drawer_id WHERE source_entity_id IS NULL")
            logger.info("_migrate_schema: migrated source_drawer_id → source_entity_id")

    def _conn(self):
        if self._closed:
            raise RuntimeError("KnowledgeGraph is closed")
        if self._connection is None:
            with self._conn_lock:
                if self._connection is None:
                    self._connection = sqlite3.connect(str(self._db_path), check_same_thread=False)
                    self._connection.row_factory = sqlite3.Row
        return self._connection

    @staticmethod
    def _entity_id(name: str) -> str:
        return name.lower().replace(" ", "_").replace("'", "")

    # ── Write operations ──────────────────────────────────────────────────

    def add(self, subject: str, predicate: str, obj: str,
            valid_from: Optional[str] = None, valid_to: Optional[str] = None,
            source: str = "") -> int:
        """Add a fact. Backward-compatible — keeps existing signature."""
        pred = predicate.lower().replace(" ", "_")
        with self._lock:
            cur = self._conn().execute(
                "INSERT INTO facts (subject, predicate, object, valid_from, valid_to, source) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (subject, pred, obj, valid_from, valid_to, source))
            self._conn().commit()
            return cur.lastrowid

    def add_entity(self, name: str, entity_type: str = "unknown",
                   properties: dict = None) -> str:
        """Add or update an entity node."""
        eid = self._entity_id(name)
        props = json.dumps(properties or {})
        with self._lock:
            conn = self._conn()
            conn.execute(
                "INSERT OR REPLACE INTO entities (id, name, type, properties) VALUES (?, ?, ?, ?)",
                (eid, name, entity_type, props))
            conn.commit()
        return eid

    def batch_add_entities(self, entities: list[tuple[str, str, dict]]) -> list[str]:
        """Add multiple entities in a single transaction.

        Each tuple is ``(name, entity_type, properties)``.
        Returns list of entity IDs in the same order as input.
        """
        eids: list[str] = []
        rows: list[tuple] = []
        for name, entity_type, properties in entities:
            eid = self._entity_id(name)
            eids.append(eid)
            props = json.dumps(properties or {})
            rows.append((eid, name, entity_type, props))
        with self._lock:
            conn = self._conn()
            conn.executemany(
                "INSERT OR REPLACE INTO entities (id, name, type, properties) VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        return eids

    def add_triple(self, subject: str, predicate: str, obj: str,
                   valid_from: Optional[str] = None,
                   valid_to: Optional[str] = None,
                   confidence: float = 1.0,
                   source_node: Optional[str] = None,
                   source_file: Optional[str] = None,
                    source_entity_id: Optional[str] = None,
                    adapter_name: Optional[str] = None) -> int:
        """Add a relationship triple with full provenance and temporal validation.

        Sanitizes temporal values, rejects inverted intervals, auto-creates
        entity entries, and updates existing active triples with new metadata.

        Returns the integer row ID of the facts table entry.
        """
        valid_from = sanitize_iso_temporal(valid_from, "valid_from")
        valid_to = sanitize_iso_temporal(valid_to, "valid_to")
        if (valid_from is not None and valid_to is not None
                and _temporal_end_key(valid_to) < _temporal_start_key(valid_from)):
            raise ValueError(
                f"valid_to={valid_to!r} is before valid_from={valid_from!r}; "
                "an inverted interval would be invisible to every KG query")
        sub_id = self._entity_id(subject)
        obj_id = self._entity_id(obj)
        pred = predicate.lower().replace(" ", "_")
        with self._lock:
            conn = self._conn()
            conn.execute(
                "INSERT OR IGNORE INTO entities (id, name) VALUES (?, ?)",
                (sub_id, subject))
            conn.execute(
                "INSERT OR IGNORE INTO entities (id, name) VALUES (?, ?)",
                (obj_id, obj))
            existing = conn.execute(
                "SELECT id FROM facts WHERE subject=? AND predicate=? AND object=? AND valid_to IS NULL",
                (subject, pred, obj)).fetchone()
            if existing:
                conn.execute(
                    "UPDATE facts SET valid_from=?, valid_to=?, confidence=?, "
                    "source_node=?, source_file=?, source_entity_id=?, adapter_name=? "
                    "WHERE id=?",
                    (valid_from, valid_to, confidence, source_node, source_file,
                     source_entity_id, adapter_name, existing["id"]))
                conn.commit()
                return existing["id"]
            conn.execute(
                "INSERT INTO facts (subject, predicate, object, valid_from, valid_to, "
                "confidence, source_node, source_file, source_entity_id, adapter_name) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (subject, pred, obj, valid_from, valid_to,
                 confidence, source_node, source_file, source_entity_id, adapter_name))
            conn.commit()
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def batch_add_triples(self, triples: list[tuple]) -> list[int]:
        """Add multiple triples in a single transaction.

        Each tuple is ``(subject, predicate, object, valid_from, valid_to,
        confidence, source_node, source_file, source_entity_id, adapter_name)``.
        All fields after ``object`` are optional (pass ``None`` for defaults).

        Returns a list of row IDs.
        """
        # First pass: parse and validate all tuples (lock-free).
        # If any fails (inverted interval, bad temporal) no SQL is touched.
        parsed: list[tuple] = []
        for t in triples:
            subject, predicate, obj = t[0], t[1], t[2]
            valid_from = (sanitize_iso_temporal(t[3], "valid_from")
                          if len(t) > 3 and t[3] else None)
            valid_to = (sanitize_iso_temporal(t[4], "valid_to")
                        if len(t) > 4 and t[4] else None)
            if (valid_from is not None and valid_to is not None
                    and _temporal_end_key(valid_to) < _temporal_start_key(valid_from)):
                raise ValueError(
                    f"valid_to={valid_to!r} is before valid_from={valid_from!r}; "
                    "an inverted interval would be invisible to every KG query")
            pred = predicate.lower().replace(" ", "_")
            confidence = float(t[5]) if len(t) > 5 and t[5] is not None else 1.0
            source_node = t[6] if len(t) > 6 else None
            source_file = t[7] if len(t) > 7 else None
            source_entity_id = t[8] if len(t) > 8 else None
            adapter_name = t[9] if len(t) > 9 else None
            parsed.append((subject, pred, obj, valid_from, valid_to, confidence,
                          source_node, source_file, source_entity_id, adapter_name))

        # Second pass: insert all inside a single transaction.
        ids: list[int] = []
        with self._lock:
            conn = self._conn()
            for (subject, pred, obj, valid_from, valid_to, confidence,
                 source_node, source_file, source_entity_id, adapter_name) in parsed:
                sub_id = self._entity_id(subject)
                obj_id = self._entity_id(obj)
                conn.execute(
                    "INSERT OR IGNORE INTO entities (id, name) VALUES (?, ?)",
                    (sub_id, subject))
                conn.execute(
                    "INSERT OR IGNORE INTO entities (id, name) VALUES (?, ?)",
                    (obj_id, obj))
                existing = conn.execute(
                    "SELECT id FROM facts WHERE subject=? AND predicate=? AND object=? AND valid_to IS NULL",
                    (subject, pred, obj)).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE facts SET valid_from=?, valid_to=?, confidence=?, "
                        "source_node=?, source_file=?, source_entity_id=?, adapter_name=? WHERE id=?",
                        (valid_from, valid_to, confidence, source_node, source_file,
                         source_entity_id, adapter_name, existing["id"]))
                    ids.append(existing["id"])
                else:
                    conn.execute(
                        "INSERT INTO facts (subject, predicate, object, valid_from, valid_to, "
                        "confidence, source_node, source_file, source_entity_id, adapter_name) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (subject, pred, obj, valid_from, valid_to, confidence,
                         source_node, source_file, source_entity_id, adapter_name))
                    ids.append(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
            conn.commit()
        return ids

    def invalidate(self, subject: str, predicate: str, object: str,
                   ended: Optional[str] = None) -> int:
        """Mark a fact as no longer valid.

        Returns number of rows updated.
        """
        if ended is None:
            ended = date.today().isoformat()
        ended = sanitize_iso_temporal(ended, "ended")
        with self._lock:
            conn = self._conn()
            pred = predicate.lower().replace(" ", "_")
            rows = conn.execute(
                "SELECT id, valid_from FROM facts "
                "WHERE subject=? AND predicate=? AND object=? AND valid_to IS NULL",
                (subject, pred, object)).fetchall()
            for row in rows:
                vf = row["valid_from"]
                if vf is not None and _temporal_end_key(ended) < _temporal_start_key(vf):
                    raise ValueError(
                        f"valid_to={ended!r} is before valid_from={vf!r}; "
                        "an inverted interval would be invisible to every KG query")
            cur = conn.execute(
                "UPDATE facts SET valid_to=? "
                "WHERE subject=? AND predicate=? AND object=? AND valid_to IS NULL",
                (ended, subject, pred, object))
            conn.commit()
            return cur.rowcount

    # ── Query operations ──────────────────────────────────────────────────

    def query(self, entity: Optional[str] = None, predicate: Optional[str] = None,
              as_of: Optional[str] = None, direction: str = "both") -> list[dict]:
        """Query facts. Backward-compatible signature and return format."""
        conditions = []
        params = []
        if entity:
            if direction in ("outgoing", "both"):
                conditions.append("(subject = ?)")
                params.append(entity)
            if direction in ("incoming", "both"):
                conditions.append("(object = ?)")
                params.append(entity)
            if len(conditions) == 2:
                conditions = [f"({' OR '.join(conditions)})"]
        if predicate:
            pred = predicate.lower().replace(" ", "_")
            conditions.append("predicate = ?")
            params.append(pred)
        if as_of:
            as_of = sanitize_iso_temporal(as_of, "as_of")
            as_of_key = _temporal_start_key(as_of)
            valid_from_expr = _sql_temporal_start_expr("valid_from")
            valid_to_expr = _sql_temporal_end_expr("valid_to")
            conditions.append(
                f"(valid_from IS NULL OR {valid_from_expr} <= ?) "
                f"AND (valid_to IS NULL OR {valid_to_expr} >= ?)"
            )
            params.extend([as_of_key, as_of_key])
        sql = "SELECT * FROM facts"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY created_at DESC"
        with self._lock:
            rows = self._conn().execute(sql, params).fetchall()
        return [{"id": r[0], "subject": r[1], "predicate": r[2], "object": r[3],
                 "valid_from": r[4], "valid_to": r[5], "source": r[6], "created_at": r[7]}
                for r in rows]

    def query_entity(self, name: str, as_of: Optional[str] = None,
                     direction: str = "outgoing") -> list[dict]:
        """Get all relationships for an entity with temporal filtering.

        direction: 'outgoing' (entity -> ?), 'incoming' (? -> entity), 'both'
        """
        as_of = sanitize_iso_temporal(as_of, "as_of")
        results = []
        temporal_sql = ""
        temporal_params = []
        if as_of:
            temporal_sql, temporal_params = _temporal_filter_sql(as_of)
        with self._lock:
            conn = self._conn()
            if direction in ("outgoing", "both"):
                rows = conn.execute(
                    "SELECT * FROM facts AS t WHERE t.subject = ?" + temporal_sql,
                    [name] + temporal_params).fetchall()
                for r in rows:
                    results.append({
                        "direction": "outgoing",
                        "subject": name,
                        "predicate": r["predicate"],
                        "object": r["object"],
                        "valid_from": r["valid_from"],
                        "valid_to": r["valid_to"],
                        "confidence": r["confidence"] if r["confidence"] is not None else 1.0,
                        "source_node": r["source_node"] if "source_node" in r.keys() else None,
                        "source_file": r["source_file"] if "source_file" in r.keys() else None,
                        "source_entity_id": r["source_entity_id"] if "source_entity_id" in r.keys() else None,
                        "adapter_name": r["adapter_name"] if "adapter_name" in r.keys() else None,
                        "current": r["valid_to"] is None,
                    })
            if direction in ("incoming", "both"):
                rows = conn.execute(
                    "SELECT * FROM facts AS t WHERE t.object = ?" + temporal_sql,
                    [name] + temporal_params).fetchall()
                for r in rows:
                    results.append({
                        "direction": "incoming",
                        "subject": r["subject"],
                        "predicate": r["predicate"],
                        "object": name,
                        "valid_from": r["valid_from"],
                        "valid_to": r["valid_to"],
                        "confidence": r["confidence"] if r["confidence"] is not None else 1.0,
                        "source_node": r["source_node"] if "source_node" in r.keys() else None,
                        "source_file": r["source_file"] if "source_file" in r.keys() else None,
                        "source_entity_id": r["source_entity_id"] if "source_entity_id" in r.keys() else None,
                        "adapter_name": r["adapter_name"] if "adapter_name" in r.keys() else None,
                        "current": r["valid_to"] is None,
                    })
        return results

    def query_relationship(self, predicate: str,
                           as_of: Optional[str] = None) -> list[dict]:
        """Get all triples with a given relationship type."""
        as_of = sanitize_iso_temporal(as_of, "as_of")
        pred = predicate.lower().replace(" ", "_")
        query = "SELECT * FROM facts AS t WHERE t.predicate = ?"
        params = [pred]
        if as_of:
            temporal_sql, temporal_params = _temporal_filter_sql(as_of)
            query += temporal_sql
            params.extend(temporal_params)
        results = []
        with self._lock:
            for r in self._conn().execute(query, params).fetchall():
                results.append({
                    "subject": r["subject"],
                    "predicate": pred,
                    "object": r["object"],
                    "valid_from": r["valid_from"],
                    "valid_to": r["valid_to"],
                    "current": r["valid_to"] is None,
                })
        return results

    def timeline(self, entity: Optional[str] = None) -> list[dict]:
        """Get all facts in chronological order (valid_from ASC NULLS LAST)."""
        with self._lock:
            conn = self._conn()
            if entity:
                rows = conn.execute(
                    "SELECT * FROM facts AS t "
                    "WHERE (t.subject = ? OR t.object = ?) "
                    "ORDER BY t.valid_from ASC NULLS LAST LIMIT 100",
                    (entity, entity)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM facts AS t "
                    "ORDER BY t.valid_from ASC NULLS LAST LIMIT 100").fetchall()
        return [
            {
                "subject": r["subject"],
                "predicate": r["predicate"],
                "object": r["object"],
                "valid_from": r["valid_from"],
                "valid_to": r["valid_to"],
                "current": r["valid_to"] is None,
            }
            for r in rows
        ]

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            conn = self._conn()
            total = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            subjects = conn.execute(
                "SELECT COUNT(DISTINCT subject) FROM facts").fetchone()[0]
            predicates_list = [
                r[0] for r in conn.execute(
                    "SELECT DISTINCT predicate FROM facts ORDER BY predicate").fetchall()
            ]
            entities = conn.execute(
                "SELECT COUNT(*) FROM entities").fetchone()[0]
            current = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE valid_to IS NULL").fetchone()[0]
            expired = total - current
        return {
            "total_facts": total,
            "unique_subjects": subjects,
            "unique_predicates": len(predicates_list),
            "entities": entities,
            "current_facts": current,
            "expired_facts": expired,
            "relationship_types": predicates_list,
        }

    # ── Seed from known facts ─────────────────────────────────────────────

    def seed_from_entity_facts(self, entity_facts: dict) -> None:
        """Seed the knowledge graph from fact_checker.py ENTITY_FACTS.

        Uses batch operations for efficiency — one transaction for entities,
        one for triples.
        """
        entity_batch: list[tuple[str, str, dict]] = []
        triple_batch: list[tuple] = []
        for key, facts in entity_facts.items():
            name = facts.get("full_name", key.capitalize())
            etype = facts.get("type", "person")
            entity_batch.append((name, etype, {
                "gender": facts.get("gender", ""),
                "birthday": facts.get("birthday", ""),
            }))
            parent = facts.get("parent")
            if parent:
                triple_batch.append((
                    name, "child_of", parent,
                    facts.get("birthday"), None, 1.0, None, None, None, None))
            partner = facts.get("partner")
            if partner:
                triple_batch.append((
                    name, "married_to", partner,
                    None, None, 1.0, None, None, None, None))
            relationship = facts.get("relationship", "")
            if relationship == "daughter":
                p = (facts.get("parent") or "").capitalize()
                if p and p != name:
                    triple_batch.append((
                        name, "is_child_of", p,
                        facts.get("birthday"), None, 1.0, None, None, None, None))
            elif relationship == "husband":
                p = (facts.get("partner") or "").capitalize()
                if p and p != name:
                    triple_batch.append((
                        name, "is_partner_of", p,
                        None, None, 1.0, None, None, None, None))
            elif relationship == "brother":
                sib = (facts.get("sibling") or "").capitalize()
                if sib and sib != name:
                    triple_batch.append((
                        name, "is_sibling_of", sib,
                        None, None, 1.0, None, None, None, None))
            elif relationship == "dog":
                owner = (facts.get("owner") or "").capitalize()
                if owner and owner != name:
                    triple_batch.append((
                        name, "is_pet_of", owner,
                        None, None, 1.0, None, None, None, None))
                entity_batch.append((name, "animal", {}))
            for interest in facts.get("interests", []):
                triple_batch.append((
                    name, "loves", interest.capitalize(),
                    "2025-01-01", None, 1.0, None, None, None, None))

        if entity_batch:
            self.batch_add_entities(entity_batch)
        if triple_batch:
            self.batch_add_triples(triple_batch)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
