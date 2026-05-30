"""Alt Memory v4 — FAISS-powered dimension with realms, domains, entities, hybrid search, entity graph, nodes."""

import contextlib
import json
import logging
import os
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from alt_memory.backends.embedder import get_embedder
from alt_memory.backends.faiss_store import FaissStore
from alt_memory.backends.knowledge_graph import KnowledgeGraph
from alt_memory.config import _SAFE_NAME_RE
from alt_memory.searcher import _hybrid_rank

logger = logging.getLogger(__name__)

# Closet infrastructure constants
NORMALIZE_VERSION = 2
NODE_CHAR_LIMIT = 2000
NODE_EXTRACT_WINDOW = 5000

# Files/dirs to skip during directory walks
SKIP_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".next", "coverage", ".ruff_cache", ".mypy_cache",
    ".pytest_cache", ".cache", ".tox", ".nox", ".idea", ".vscode",
    ".ipynb_checkpoints", ".eggs", "htmlcov", "target", ".terraform",
    "vendor", ".opencode", ".claude", ".alt-memory",
})

# Common capitalized words that look like proper nouns but are usually
# sentence-starters or filler. Filtered out of entity extraction.
_ENTITY_STOPLIST = frozenset(
    {
        "The", "This", "That", "These", "Those",
        "I", "You", "He", "She", "It", "We", "They",
        "My", "Your", "His", "Her", "Its", "Our", "Their",
        "Mine", "Yours", "Hers", "Ours", "Theirs",
        "A", "An",
        "When", "Where", "What", "Why", "Who", "Which", "How",
        "After", "Before", "Then", "Now", "Here", "There",
        "And", "But", "Or", "Yet", "So", "If", "Else", "For", "Nor",
        "Yes", "No", "Maybe", "Okay", "Not", "None", "Nothing",
        "User", "Assistant", "System", "Tool", "Human", "Bot",
        "All", "Each", "Every", "Both", "Few", "Many", "Much",
        "Some", "Any",
        "Only", "Just", "Also", "Very", "Too",
        "Please", "Hello", "Hi", "Hey", "Goodbye", "Bye", "Thanks", "Thank", "Sorry",
        "Note", "Warning", "Error", "Info", "Debug",
        "Todo", "FIXME", "HACK", "XXX",
        "True", "False", "Null", "NaN",
        "Type", "Value", "Key", "Name", "File", "Line",
        "Up", "Down", "Left", "Right", "On", "Off",
        "First", "Second", "Third", "Last", "Next", "Previous",
        "Top", "Bottom", "Middle", "Center",
        "New", "Old", "Good", "Bad", "Great", "Little",
        "Big", "Large", "Small", "High", "Low", "Long", "Short",
        "Once", "Twice", "Often", "Always", "Never", "Sometimes",
        "Yesterday", "Today", "Tomorrow",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
        "Dr", "Prof", "Mr", "Ms", "Mrs", "Sir", "Lady", "Lord",
        "Capt", "Sgt", "Rep", "Sen", "Gov", "Pres", "VP",
        "Using", "Building", "Creating", "Working", "Running", "Testing",
        "Developing", "Deploying", "Installing", "Configuring", "Setting",
        "Getting", "Putting", "Making", "Doing", "Going", "Taking",
        "Adding", "Removing", "Updating", "Fixing", "Starting", "Stopping",
    }
)

_STORE_BACKUP_FILES = ["index.faiss", "metadata.db", "seq.txt", "chroma.sqlite3"]
ENTITY_UPSERT_BATCH_SIZE = 1000
DEFAULT_MAX_FILE_SIZE = 500 * 1024 * 1024


class MineAlreadyRunning(RuntimeError):
    """Raised when another mine already holds the per-dimension lock."""


class MineValidationError(RuntimeError):
    """Raised at end of mine when PRAGMA quick_check reports errors."""

    def __init__(self, dim_path: str, errors: list[str]) -> None:
        if not errors:
            raise ValueError("MineValidationError requires at least one error string")
        if not dim_path:
            raise ValueError("MineValidationError requires a non-empty dimension path")
        super().__init__(f"FTS5/SQLite quick_check failed: {len(errors)} issue(s)")
        self.dim_path = dim_path
        self.errors: tuple[str, ...] = tuple(errors)


@contextlib.contextmanager
def mine_lock(source_file: str):
    """Cross-platform file lock for mine operations."""
    import hashlib
    lock_dir = os.path.join(os.path.expanduser("~"), ".alt-memory", "locks")
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(
        lock_dir, hashlib.sha256(source_file.encode()).hexdigest()[:16] + ".lock"
    )
    if not os.path.exists(lock_path):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY, 0o600)
            os.close(fd)
        except FileExistsError:
            pass
    lf = open(lock_path, "r+b")
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lf.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lf, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(lf, fcntl.LOCK_UN)
        except Exception:
            logger.debug("Mine-lock release failed", exc_info=True)
        lf.close()


# ── Per-dimension mine lock (non-blocking, re-entrant) ─────────────────────


_dim_lock_holders = threading.local()


def _holder_state():
    keys = getattr(_dim_lock_holders, "keys", None)
    pid = getattr(_dim_lock_holders, "pid", None)
    current_pid = os.getpid()
    if keys is None or pid != current_pid:
        keys = set()
        _dim_lock_holders.keys = keys
        _dim_lock_holders.pid = current_pid
    return keys


def _held_by_this_thread(lock_key: str) -> bool:
    return lock_key in _holder_state()


def _mark_held(lock_key: str) -> None:
    _holder_state().add(lock_key)


def _mark_released(lock_key: str) -> None:
    _holder_state().discard(lock_key)


_LOCK_SENTINEL_BYTES = 1


def _read_lock_holder(lock_file) -> str:
    try:
        lock_file.seek(_LOCK_SENTINEL_BYTES)
        content = lock_file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        content = content.strip()
    except OSError:
        return "another writer (identity not recorded)"
    if not content:
        return "another writer (identity not recorded)"
    parts = content.split(maxsplit=1)
    pid = parts[0] if parts else "?"
    cmd = parts[1].strip() if len(parts) > 1 else ""
    return f"PID {pid} ({cmd})" if cmd else f"PID {pid}"


def _write_lock_holder(lock_file) -> None:
    try:
        import sys as _sys
        ident = f"{os.getpid()} {' '.join(_sys.argv[:3])}".strip()
        ident_bytes = ident.encode("utf-8")
        lock_file.seek(_LOCK_SENTINEL_BYTES)
        lock_file.truncate(_LOCK_SENTINEL_BYTES + len(ident_bytes))
        lock_file.write(ident_bytes)
        lock_file.flush()
    except (OSError, UnicodeError):
        pass


@contextlib.contextmanager
def mine_dimension_lock(dim_path: str):
    """Per-dimension non-blocking lock around the full mine pipeline.
    Non-blocking: raises MineAlreadyRunning if another mine is active on
    this dimension. Re-entrant: same thread passes through without re-acquiring.
    """
    lock_dir = os.path.join(os.path.expanduser("~"), ".alt-memory", "locks")
    os.makedirs(lock_dir, exist_ok=True)
    resolved = os.path.realpath(os.path.expanduser(dim_path))
    lock_key_source = os.path.normcase(resolved)
    import hashlib as _hashlib
    dim_key = _hashlib.sha256(lock_key_source.encode()).hexdigest()[:16]
    lock_path = os.path.join(lock_dir, f"mine_dimension_{dim_key}.lock")

    if _held_by_this_thread(dim_key):
        yield
        return

    if not os.path.exists(lock_path):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY, 0o600)
            os.close(fd)
        except FileExistsError:
            pass
    lf = open(lock_path, "r+b")
    acquired = False
    try:
        lf.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(lf.fileno(), msvcrt.LK_NBLCK, 1)
                acquired = True
            except OSError as exc:
                holder = _read_lock_holder(lf)
                raise MineAlreadyRunning(
                    f"dimension {resolved} is held by {holder}; "
                    "wait for it to finish or stop the holder before retrying"
                ) from exc
        else:
            import fcntl
            try:
                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError as exc:
                holder = _read_lock_holder(lf)
                raise MineAlreadyRunning(
                    f"dimension {resolved} is held by {holder}; "
                    "wait for it to finish or stop the holder before retrying"
                ) from exc
        _write_lock_holder(lf)
        _mark_held(dim_key)
        try:
            yield
        finally:
            _mark_released(dim_key)
    finally:
        if acquired:
            try:
                if os.name == "nt":
                    import msvcrt
                    lf.seek(0)
                    msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lf, fcntl.LOCK_UN)
            except Exception:
                pass
        lf.close()


def _validate_dimension_fts5_after_mine(dim_path: str) -> None:
    """Raise MineValidationError if SQLite quick_check reports errors after a mine."""
    from alt_memory.repair_utils import sqlite_integrity_errors
    errors = sqlite_integrity_errors(str(Path(dim_path).expanduser() / "dimension.db"))
    if errors:
        raise MineValidationError(dim_path, errors)


def bulk_check_mined(dim_path: str) -> dict[str, float]:
    """Return dict mapping source_file -> source_mtime for all entities at current normalize_version."""
    base = Path(dim_path).expanduser().resolve()
    conn = sqlite3.connect(str(base / "dimension.db"))
    try:
        rows = conn.execute(
            "SELECT DISTINCT source_file, json_extract(metadata, '$.source_mtime') as mtime "
            "FROM entities WHERE source_file != '' AND source_file IS NOT NULL "
            "AND json_extract(metadata, '$.normalize_version') = ?",
            (NORMALIZE_VERSION,),
        ).fetchall()
        return {row[0]: float(row[1]) for row in rows if row[1] is not None}
    except (sqlite3.Error, ValueError, TypeError):
        return {}
    finally:
        conn.close()


def prefetch_mined_set(dim_path: str, extract_mode: Optional[str] = None) -> set[str]:
    """Return set of source_file paths already mined at current normalize_version.

    When ``extract_mode`` is set, only files with matching mode are returned.
    """
    base = Path(dim_path).expanduser().resolve()
    conn = sqlite3.connect(str(base / "dimension.db"))
    try:
        rows = conn.execute(
            "SELECT source_file, metadata FROM nodes WHERE source_file != '' AND source_file IS NOT NULL"
        ).fetchall()
        result: set[str] = set()
        for source_file, meta_json in rows:
            if not source_file:
                continue
            meta = json.loads(meta_json or "{}")
            stored_version = meta.get("normalize_version", 1)
            if stored_version < NORMALIZE_VERSION:
                continue
            if extract_mode is not None and not _metadata_matches_extract_mode(meta, extract_mode):
                continue
            result.add(source_file)
        return result
    except (sqlite3.Error, json.JSONDecodeError):
        return set()
    finally:
        conn.close()


def _sanitize(name: str, kind: str = "name") -> str:
    name = name.strip().lower().replace(" ", "_").replace("-", "_")
    if not _SAFE_NAME_RE.match(name):
        raise ValueError(f"Invalid {kind}: {name!r}")
    return name


@dataclass
class SearchResult:
    id: str
    text: str
    distance: float
    metadata: dict
    realm: str = ""
    domain: str = ""


class Dimension:
    """Main dimension - realms, domains, entities, search."""

    def __init__(self, path: str = "~/.alt-memory", backend: str = "faiss"):
        self._base = Path(path).expanduser().resolve()
        self._data_dir = self._base / "data"
        self._backend = backend
        self._lock = threading.RLock()
        self._initialized = False

        self._db: Optional[sqlite3.Connection] = None
        self._store: Optional[FaissStore] = None
        self._embedder: Optional[Any] = None
        self._kg: Optional[KnowledgeGraph] = None

        # Hybrid search state (FTS5)
        self._fts_enabled = False

    def _db_execute(self, sql, params=None):
        with self._lock:
            if params is not None:
                return self._db.execute(sql, params)
            return self._db.execute(sql)

    def _db_commit(self):
        with self._lock:
            self._db.commit()

    def _create_store(self, dimension: int = 384):
        if self._backend == "chroma":
            from alt_memory.backends.chroma_store import ChromaStore
            return ChromaStore(str(self._data_dir), dimension=dimension)
        from alt_memory.backends.faiss_store import FaissStore
        return FaissStore(str(self._data_dir), dimension=dimension)

    def set_backend(self, backend: str, reindex: bool = True) -> dict:
        """Switch the vector store backend (faiss or chroma).

        Persists to ``dimension.json``, closes the old store, and creates
        a new one. When ``reindex=True``, all existing entities are
        re-embedded into the new store.
        """
        valid = {"faiss", "chroma"}
        if backend not in valid:
            raise ValueError(f"Unknown backend: {backend!r}. Valid: {sorted(valid)}")

        if backend == self._backend:
            return {"backend": backend, "reindexed": 0, "note": "already active"}

        old_backend = self._backend
        self._backend = backend

        config_path = self._base / "dimension.json"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        config["backend"] = backend
        with open(config_path, "w") as f:
            json.dump(config, f)

        info = {"backend": backend, "previous_backend": old_backend}

        if reindex:
            count = self._reindex_embeddings()
            info["reindexed"] = count
        else:
            if self._store:
                self._store.close()
            self._store = self._create_store()
            info["reindexed"] = 0

        return info

    def init(self) -> bool:
        """Initialize or open the dimension. Returns True if newly created."""
        self._base.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        config_path = self._base / "dimension.json"
        is_new = not config_path.exists()

        if not is_new:
            with open(config_path) as f:
                config = json.load(f)
            self._backend = config.get("backend", self._backend)

        self._db = sqlite3.connect(str(self._base / "dimension.db"), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("""CREATE TABLE IF NOT EXISTS realms (
                name TEXT PRIMARY KEY, description TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')))""")
        self._db.execute("""CREATE TABLE IF NOT EXISTS domains (
                name TEXT, realm TEXT, description TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (name, realm),
                FOREIGN KEY (realm) REFERENCES realms(name))""")
        self._db.execute("""CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY, realm TEXT NOT NULL, domain TEXT NOT NULL,
                content TEXT NOT NULL, metadata TEXT DEFAULT '{}',
                source_file TEXT DEFAULT '', created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (realm, domain) REFERENCES domains(realm, name))""")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_entities_realm_domain ON entities(realm, domain)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_entities_source ON entities(source_file)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_entities_created_at ON entities(created_at)")
        self._db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(content, id)")
        self._db.execute("""CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY, content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                source_file TEXT DEFAULT '', realm TEXT DEFAULT '', domain TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')))""")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_nodes_source ON nodes(source_file)")
        self._db.commit()

        self._store = self._create_store()
        self._embedder = self._load_embedder(is_new=is_new)
        self._store.dimension = self._embedder.dimension
        self._kg = KnowledgeGraph(str(self._data_dir))
        self._fts_enabled = True

        if is_new:
            with open(self._base / "dimension.json", "w") as f:
                json.dump({
                    "version": 3, "created_at": datetime.now(timezone.utc).isoformat(),
                    "backend": self._backend,
                    "embedding": self._embedder_config_name(self._embedder),
                    "dimension": 384,
                }, f)

        self._initialized = True
        logger.info("Dimension %s at %s", "created" if is_new else "opened", self._base)
        return is_new

    def close(self) -> None:
        with self._lock:
            self._save_embedder()
            if self._store:
                self._store.close()
            if self._kg:
                self._kg.close()
            if self._db:
                self._db.close()
            self._initialized = False

    def _save_embedder(self):
        if self._embedder and self._embedder.is_fitted:
            save = getattr(self._embedder, "save", None)
            if save:
                try:
                    save(str(self._data_dir))
                except Exception:
                    pass

    def _load_embedder(self, is_new: bool = False, device: Optional[str] = None) -> Any:
        config_path = self._base / "dimension.json"
        model = "sentence"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            model = config.get("embedding", "numpy")
            mapping = {
                "numpy_tfidf_svd": "numpy",
                "sentence_transformers": "sentence",
                "spacy_glove": "spacy",
            }
            model = mapping.get(model, model)
        else:
            from alt_memory.config import AltMemoryConfig
            model = AltMemoryConfig().default_embedder
        model = self._resolve_embedder(model, config_path)
        if device is None:
            from alt_memory.config import AltMemoryConfig
            device = AltMemoryConfig().embedding_device
        return get_embedder(model=model, model_dir=str(self._data_dir), device=device)

    @staticmethod
    def _resolve_embedder(model: str, config_path: Path) -> str:
        """Check if the configured embedder's dependency is available; fall back to numpy if not."""
        if model == "sentence":
            try:
                from sentence_transformers import SentenceTransformer
                SentenceTransformer
            except ImportError:
                logger.warning("sentence_transformers not installed — falling back to numpy. Install with: pip install sentence-transformers")
                model = "numpy"
        elif model == "spacy":
            try:
                import spacy
                spacy
            except ImportError:
                logger.warning("spacy not installed — falling back to numpy. Install with: pip install spacy")
                model = "numpy"
        elif model == "bert":
            try:
                import onnxruntime
                onnxruntime
            except ImportError:
                logger.info("onnxruntime not available for bert — using numpy backend")
                model = "numpy"
        elif model == "minilm" or model == "embeddinggemma":
            try:
                import onnxruntime
                onnxruntime
            except ImportError:
                logger.warning("onnxruntime not installed for %s — falling back to numpy", model)
                model = "numpy"
        if model == "numpy" and config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            if config.get("embedding") != "numpy_tfidf_svd":
                config["embedding"] = "numpy_tfidf_svd"
                with open(config_path, "w") as f:
                    json.dump(config, f)
        return model

    @staticmethod
    def _embedder_config_name(embedder: Any) -> str:
        """Map embedder instance back to its dimension.json config name."""
        cls_name = type(embedder).__name__
        mapping = {
            "NumpyEmbedder": "numpy_tfidf_svd",
            "SentenceTransformerEmbedder": "sentence_transformers",
            "SpacyGloveEmbedder": "spacy_glove",
            "OnnxEmbedder": "minilm",
            "EmbeddinggemmaONNX": "embeddinggemma",
            "NumpyBertEmbedder": "bert",
        }
        return mapping.get(cls_name, "numpy_tfidf_svd")

    @property
    def kg(self) -> KnowledgeGraph:
        if not self._kg:
            raise RuntimeError("Dimension not initialized")
        return self._kg

    # -- Realm management --

    def list_realms(self) -> list[dict]:
        rows = self._db_execute("""SELECT r.name, r.description, r.created_at,
                COUNT(e.id) as entity_count FROM realms r
                LEFT JOIN entities e ON e.realm = r.name
                GROUP BY r.name ORDER BY r.name""").fetchall()
        return [{"name": row[0], "description": row[1], "created_at": row[2], "entity_count": row[3]}
                for row in rows]

    def get_or_create_realm(self, name: str, description: str = "") -> str:
        name = _sanitize(name, "realm")
        with self._lock:
            self._db.execute("INSERT OR IGNORE INTO realms (name, description) VALUES (?, ?)",
                             (name, description))
            self._db.commit()
        return name

    def delete_realm(self, name: str) -> bool:
        name = _sanitize(name, "realm")
        with self._lock:
            ids = [r[0] for r in self._db.execute(
                "SELECT id FROM entities WHERE realm = ?", (name,)).fetchall()]
            if ids and self._store:
                self._store.delete(ids=ids)
            self._db.execute("DELETE FROM entities_fts WHERE id IN (SELECT id FROM entities WHERE realm = ?)", (name,))
            self._db.execute("DELETE FROM entities WHERE realm = ?", (name,))
            self._db.execute("DELETE FROM domains WHERE realm = ?", (name,))
            self._db.execute("DELETE FROM realms WHERE name = ?", (name,))
            self._db.commit()
        return True

    # -- Domain management --

    def list_domains(self, realm: Optional[str] = None) -> list[dict]:
        if realm:
            rows = self._db_execute("""SELECT d.name, d.realm, d.description, d.created_at,
                    COUNT(e.id) as entity_count FROM domains d
                    LEFT JOIN entities e ON e.domain = d.name AND e.realm = d.realm
                    WHERE d.realm = ? GROUP BY d.name, d.realm ORDER BY d.name""", (realm,))
        else:
            rows = self._db_execute("""SELECT d.name, d.realm, d.description, d.created_at,
                    COUNT(e.id) as entity_count FROM domains d
                    LEFT JOIN entities e ON e.domain = d.name AND e.realm = d.realm
                    GROUP BY d.name, d.realm ORDER BY d.realm, d.name""")
        return [{"name": row[0], "realm": row[1], "description": row[2], "created_at": row[3],
                 "entity_count": row[4]} for row in rows.fetchall()]

    def get_or_create_domain(self, realm: str, name: str, description: str = "") -> str:
        realm = _sanitize(realm, "realm")
        name = _sanitize(name, "domain")
        self.get_or_create_realm(realm)
        with self._lock:
            self._db.execute("INSERT OR IGNORE INTO domains (name, realm, description) VALUES (?, ?, ?)",
                             (name, realm, description))
            self._db.commit()
        return name

    def delete_domain(self, realm: str, name: str) -> bool:
        realm = _sanitize(realm, "realm")
        name = _sanitize(name, "domain")
        with self._lock:
            ids = [r[0] for r in self._db.execute(
                "SELECT id FROM entities WHERE realm = ? AND domain = ?", (realm, name)).fetchall()]
            if ids and self._store:
                self._store.delete(ids=ids)
            self._db.execute("DELETE FROM entities_fts WHERE id IN (SELECT id FROM entities WHERE realm = ? AND domain = ?)", (realm, name))
            self._db.execute("DELETE FROM entities WHERE realm = ? AND domain = ?", (realm, name))
            self._db.execute("DELETE FROM domains WHERE realm = ? AND name = ?", (realm, name))
            self._db.commit()
        return True

    # -- Entity operations --

    def add_entity(self, realm: str, domain: str, content: str,
                   metadata: Optional[dict] = None,
                   source_file: str = "",
                   entity_id: Optional[str] = None) -> str:
        if not content.strip():
            raise ValueError("Content cannot be empty")
        realm = _sanitize(realm, "realm")
        domain = _sanitize(domain, "domain")
        self.get_or_create_domain(realm, domain)
        if entity_id is None:
            entity_id = self._store.next_id()
        meta = dict(metadata or {})
        meta["realm"] = realm
        meta["domain"] = domain
        if source_file is not None:
            meta["source_file"] = source_file
        embedding = self._embedder.embed([content])[0]
        embedding_2d = embedding.reshape(1, -1)
        with self._lock:
            self._store.add(ids=[entity_id], texts=[content], metadatas=[meta],
                            embeddings=embedding_2d)
            self._db.execute(
                "INSERT OR REPLACE INTO entities (id, realm, domain, content, metadata, source_file) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (entity_id, realm, domain, content, json.dumps(meta), source_file))
            self._db.execute(
                "DELETE FROM entities_fts WHERE id = ?",
                (entity_id,))
            self._db.execute(
                "INSERT INTO entities_fts (id, content) VALUES (?, ?)",
                (entity_id, content))
            self._db.commit()
        self._save_embedder()
        logger.debug("Added entity %s in %s/%s", entity_id, realm, domain)
        return entity_id

    def batch_add_entities(
        self,
        entities: list[tuple[str, str, str, dict, str, Optional[str]]],
    ) -> list[str]:
        """Add multiple entities in a single transaction.

        Each tuple is ``(realm, domain, content, metadata, source_file, entity_id)``.
        If ``entity_id`` is None, one is auto-generated.

        Returns list of entity IDs in the same order as input.
        """
        if not entities:
            return []

        realms = set()
        ids: list[str] = []
        texts: list[str] = []
        all_metas: list[dict] = []
        sql_rows: list[tuple] = []
        fts_rows: list[tuple] = []

        for i, (realm, domain, content, metadata, source_file, entity_id) in enumerate(entities):
            if not content.strip():
                continue
            realm = _sanitize(realm, "realm")
            domain = _sanitize(domain, "domain")
            realms.add((realm, domain))
            if entity_id is None:
                entity_id = self._store.next_id()
            meta = dict(metadata or {})
            meta["realm"] = realm
            meta["domain"] = domain
            if source_file is not None:
                meta["source_file"] = source_file
            ids.append(entity_id)
            texts.append(content)
            all_metas.append(meta)
            sql_rows.append((entity_id, realm, domain, content, json.dumps(meta), source_file))
            fts_rows.append((entity_id, content))

        if not ids:
            return []

        for realm, domain in realms:
            self.get_or_create_domain(realm, domain)

        embeddings = self._embedder.embed(texts)
        embeddings_2d = embeddings.reshape(len(texts), -1) if embeddings.ndim == 1 else embeddings

        with self._lock:
            self._store.add(ids=ids, texts=texts, metadatas=all_metas, embeddings=embeddings_2d)
            self._db.executemany(
                "INSERT OR REPLACE INTO entities (id, realm, domain, content, metadata, source_file) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                sql_rows,
            )
            self._db.executemany("DELETE FROM entities_fts WHERE id = ?",
                                 [(eid,) for eid, _ in fts_rows])
            self._db.executemany(
                "INSERT INTO entities_fts (id, content) VALUES (?, ?)",
                fts_rows,
            )
            self._db.commit()

        self._save_embedder()
        logger.debug("batch_add_entities: added %d entities", len(ids))
        return ids

    def get_entity(self, entity_id: str) -> Optional[dict]:
        row = self._db_execute(
            "SELECT id, realm, domain, content, metadata, source_file, created_at "
            "FROM entities WHERE id = ?", (entity_id,)).fetchone()
        if not row:
            return None
        return {"id": row[0], "realm": row[1], "domain": row[2], "content": row[3],
                "metadata": json.loads(row[4] or "{}"), "source_file": row[5],
                "created_at": row[6]}

    def list_entities(self, realm: Optional[str] = None, domain: Optional[str] = None,
                      limit: int = 20, offset: int = 0) -> list[dict]:
        conditions = []
        params = []
        if realm:
            conditions.append("realm = ?")
            params.append(realm)
        if domain:
            conditions.append("domain = ?")
            params.append(domain)
        sql = "SELECT id, realm, domain, content, metadata, source_file, created_at FROM entities"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self._db_execute(sql, params).fetchall()
        return [{"id": row[0], "realm": row[1], "domain": row[2],
                 "content": row[3],
                 "metadata": json.loads(row[4] or "{}"), "source_file": row[5],
                 "created_at": row[6]} for row in rows]

    def delete_entity(self, entity_id: str) -> bool:
        with self._lock:
            row = self._db.execute("SELECT id FROM entities WHERE id = ?", (entity_id,)).fetchone()
            if not row:
                return False
            if self._store:
                self._store.delete(ids=[entity_id])
            self._db.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
            self._db.execute("DELETE FROM entities_fts WHERE id = ?", (entity_id,))
            self._db.commit()
        return True

    def delete_entities(self, entity_ids: list[str]) -> int:
        if not entity_ids:
            return 0
        with self._lock:
            if self._store:
                self._store.delete(ids=entity_ids)
            placeholders = ",".join("?" * len(entity_ids))
            self._db.execute(f"DELETE FROM entities WHERE id IN ({placeholders})", entity_ids)
            self._db.execute(f"DELETE FROM entities_fts WHERE id IN ({placeholders})", entity_ids)
            self._db.commit()
        return len(entity_ids)

    def update_entity(self, entity_id: str, content: Optional[str] = None,
                      metadata: Optional[dict] = None,
                      realm: Optional[str] = None,
                      domain: Optional[str] = None) -> bool:
        existing = self.get_entity(entity_id)
        if not existing:
            return False
        new_content = content if content is not None else existing["content"]
        new_meta = dict(existing["metadata"])
        if metadata:
            new_meta.update(metadata)
        if realm:
            new_meta["realm"] = _sanitize(realm, "realm")
        if domain:
            new_meta["domain"] = _sanitize(domain, "domain")
        with self._lock:
            embedding = self._embedder.embed([new_content])[0]
            embedding_2d = embedding.reshape(1, -1)
            self._store.upsert(ids=[entity_id], texts=[new_content],
                                metadatas=[new_meta], embeddings=embedding_2d)
            if content is not None:
                self._db.execute("DELETE FROM entities_fts WHERE id = ?", (entity_id,))
                self._db.execute("INSERT OR REPLACE INTO entities_fts (id, content) VALUES (?, ?)",
                                  (entity_id, new_content))
            new_realm = new_meta.get("realm", existing["realm"])
            new_domain = new_meta.get("domain", existing["domain"])
            self._db.execute(
                "UPDATE entities SET content=?, metadata=?, realm=?, domain=? WHERE id=?",
                (new_content, json.dumps(new_meta), new_realm, new_domain, entity_id))
            self._db.commit()
        self._save_embedder()
        return True

    # -- Search (hybrid: FTS5 BM25 + FAISS vector) --

    def search(self, query: str, n_results: int = 10,
               realm: Optional[str] = None, domain: Optional[str] = None,
               mode: str = "hybrid") -> list[SearchResult]:
        """Search across all entities. mode: 'vector', 'keyword', or 'hybrid'."""
        if not self._store or self._store.count() == 0:
            if mode == "hybrid":
                return self._keyword_search(query, n_results, realm, domain)
            if mode != "keyword":
                return []

        if mode == "keyword":
            return self._keyword_search(query, n_results, realm, domain)
        elif mode == "vector":
            return self._vector_search(query, n_results, realm, domain)
        else:
            return self._hybrid_search(query, n_results, realm, domain)

    def search_memories(
        self,
        query: str,
        n_results: int = 5,
        realm: Optional[str] = None,
        domain: Optional[str] = None,
        max_distance: float = 0.0,
        candidate_strategy: str = "vector",
        vector_disabled: bool = False,
    ) -> dict:
        """Programmatic search — returns a dict instead of SearchResult list.

        Used by the MCP server and other callers that need structured data.

        Args:
            query: Natural language search query.
            n_results: Max results to return.
            realm: Optional realm filter.
            domain: Optional domain filter.
            max_distance: Max cosine distance threshold. 0 = identical, 2 = opposite.
                Results with distance > this are filtered out. 0.0 disables filtering.
            candidate_strategy: ``"vector"`` (default) or ``"union"`` (vector + BM25).
            vector_disabled: When True, route to keyword-only search.
        """
        from alt_memory.searcher import validate_candidate_strategy, apply_candidate_strategy

        validate_candidate_strategy(candidate_strategy)

        mode = "keyword" if vector_disabled else "hybrid"
        sr = self.search(query, n_results, realm, domain, mode=mode)
        if vector_disabled or not self._store or self._store.count() == 0:
            return {
                "query": query,
                "filters": {"realm": realm, "domain": domain},
                "results": [
                    {
                        "id": r.id,
                        "text": r.text,
                        "distance": None if vector_disabled else r.distance,
                        "metadata": r.metadata,
                        "realm": r.realm,
                        "domain": r.domain,
                    }
                    for r in sr
                ],
                **({"fallback": "keyword_only"} if vector_disabled else {}),
            }

        # Over-fetch for re-ranking
        vec_results = self._vector_search(query, n_results * 3, realm, domain)
        kw_results = self._keyword_search(query, n_results * 3, realm, domain)

        # Apply max_distance threshold
        if max_distance > 0.0:
            vec_results = [r for r in vec_results if r.distance <= max_distance]

        # Convert to dict hits
        hits: list[dict] = []
        for r in vec_results:
            entry = {
                "id": r.id,
                "text": r.text,
                "distance": r.distance,
                "metadata": r.metadata,
                "realm": r.realm,
                "domain": r.domain,
                "_source_file_full": r.metadata.get("source_file", ""),
                "_chunk_index": r.metadata.get("chunk_index"),
            }
            hits.append(entry)

        # Apply candidate strategy (union = merge BM25-only hits)
        kw_dicts = [
            {
                "text": r.text,
                "distance": r.distance,
                "metadata": r.metadata,
                "source_file": r.metadata.get("source_file", ""),
                "_source_file_full": r.metadata.get("source_file", ""),
                "_chunk_index": r.metadata.get("chunk_index"),
            }
            for r in kw_results
        ]
        apply_candidate_strategy(
            candidate_strategy,
            hits,
            query,
            kw_dicts,
            n_results,
            max_distance=max_distance,
        )

        # Node boost — tiered by node reference count
        source_files = set()
        for h in hits:
            src = h.get("_source_file_full") or h.get("metadata", {}).get("source_file")
            if src:
                source_files.add(src)
        boosted = self._get_node_source_boosts(source_files) if source_files else {}

        if boosted:
            for h in hits:
                eid = h.get("id")
                if eid in boosted:
                    h["node_boost"] = boosted[eid]

        # Node preview — show matching node content for boosted results
        node_preview_map: dict[str, str] = {}
        if source_files:
            placeholders = ",".join("?" * len(source_files))
            node_rows = self._db_execute(
                f"SELECT id, content, source_file FROM nodes WHERE source_file IN ({placeholders}) LIMIT 100",
                list(source_files),
            ).fetchall()
            for cid, ccontent, csrc in node_rows:
                key = csrc
                if key not in node_preview_map and ccontent.strip():
                    node_preview_map[key] = ccontent[:200]

        # Drawer-grep hydration — for boosted entities, find the best keyword-matching
        # chunk from the same source file and expand with neighbor context.
        hydrated_source_files: set[str] = set()
        for h in hits:
            if h.get("node_boost", 0) > 0:
                src = h.get("_source_file_full") or h.get("metadata", {}).get("source_file", "")
                if src:
                    hydrated_source_files.add(src)
        if hydrated_source_files:
            from alt_memory.searcher import _tokenize
            query_tokens = _tokenize(query)
            for src in hydrated_source_files:
                neighbor_rows = self._db_execute(
                    "SELECT id, content, metadata FROM entities WHERE source_file = ? ORDER BY created_at",
                    (src,),
                ).fetchall()
                if len(neighbor_rows) <= 1:
                    continue
                scored: list[tuple[int, str, int]] = []
                for nid, ncontent, nmeta_json in neighbor_rows:
                    nmeta = json.loads(nmeta_json or "{}")
                    nchunk = nmeta.get("chunk_index", 0)
                    ntok = _tokenize(ncontent)
                    score = sum(1 for t in query_tokens if t in ntok)
                    scored.append((score, ncontent, nchunk))
                scored.sort(key=lambda x: -x[0])
                if scored and scored[0][0] > 0:
                    best_score, best_text, best_chunk = scored[0]
                    neighbors: list[str] = [best_text]
                    for score, ntext, nchunk in scored:
                        if abs(nchunk - best_chunk) == 1:
                            neighbors.append(ntext)
                    hydrated = "\n\n---\n\n".join(neighbors)
                    for h in hits:
                        hsrc = h.get("_source_file_full") or h.get("metadata", {}).get("source_file", "")
                        if hsrc == src:
                            h["hydrated_text"] = hydrated

        # BM25 hybrid re-rank
        hits = _hybrid_rank(hits, query)[:n_results]

        # Attach node_preview to results
        for h in hits:
            src = h.get("_source_file_full") or h.get("metadata", {}).get("source_file", "")
            if src in node_preview_map:
                h["node_preview"] = node_preview_map[src]

        # Clean internal fields
        for h in hits:
            h.pop("_source_file_full", None)
            h.pop("_chunk_index", None)
            h.setdefault("node_boost", 0.0)

        return {
            "query": query,
            "filters": {"realm": realm, "domain": domain},
            "results": hits,
        }

    def _vector_search(self, query: str, n_results: int = 10,
                       realm: Optional[str] = None, domain: Optional[str] = None) -> list[SearchResult]:
        query_emb = self._embedder.embed([query])[0]
        ids, texts, distances, metadatas = self._store.search(query_emb, n_results=n_results * 2)
        results = []
        for i in range(len(ids)):
            meta = metadatas[i] if i < len(metadatas) else {}
            dr = meta.get("realm", "")
            dd = meta.get("domain", "")
            if realm and dr != realm:
                continue
            if domain and dd != domain:
                continue
            results.append(SearchResult(id=ids[i], text=texts[i],
                            distance=float(distances[i]) if i < len(distances) else 0.0,
                            metadata=meta, realm=dr, domain=dd))
            if len(results) >= n_results:
                break
        return results

    def _keyword_search(self, query: str, n_results: int = 10,
                        realm: Optional[str] = None, domain: Optional[str] = None) -> list[SearchResult]:
        if not self._fts_enabled:
            return self._vector_search(query, n_results, realm, domain)
        fts_query = self._build_fts_query(query)
        sql = """SELECT e.id, e.realm, e.domain, e.content, e.metadata,
                 rank FROM entities_fts f
                 JOIN entities e ON e.id = f.id
                 WHERE entities_fts MATCH ?"""
        params = [fts_query]
        if realm:
            sql += " AND e.realm = ?"
            params.append(realm)
        if domain:
            sql += " AND e.domain = ?"
            params.append(domain)
        sql += " ORDER BY rank LIMIT ?"
        params.append(n_results)
        try:
            rows = self._db_execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            logger.warning("FTS5 query failed, falling back to vector search", exc_info=True)
            return self._vector_search(query, n_results, realm, domain)
        results = []
        for r in rows:
            meta = json.loads(r[4] or "{}")
            results.append(SearchResult(id=r[0], text=r[3], distance=1.0 - float(r[5]),
                            metadata=meta, realm=r[1], domain=r[2]))
        return results

    @staticmethod
    def _build_fts_query(query: str) -> str:
        """Build an FTS5 query string with prefix matching for bare terms.

        FTS5 natively supports AND, OR, NOT, NEAR/N, ``"phrase"``, and
        ``term*`` prefix syntax. If the query already contains any of these
        operators it is passed through verbatim. Bare space-separated terms
        gain ``*`` prefix wildcards so partial input matches.
        """
        tokens = query.strip().split()
        if not tokens:
            return query
        for t in tokens:
            upper = t.upper()
            if upper in ("AND", "OR", "NOT") or upper.startswith("NEAR"):
                return query
        if any(t.startswith('"') for t in tokens):
            return query
        return " AND ".join(t + "*" for t in tokens)

    def _hybrid_search(self, query: str, n_results: int = 10,
                        realm: Optional[str] = None, domain: Optional[str] = None) -> list[SearchResult]:
        vec_results = self._vector_search(query, n_results * 2, realm, domain)
        kw_results = self._keyword_search(query, n_results * 2, realm, domain)

        # Normalize keyword distances to [0, 1] to match cosine similarity scale
        if kw_results:
            kw_max = max(r.distance for r in kw_results)
            if kw_max > 0:
                kw_results = [
                    SearchResult(
                        id=r.id, text=r.text, distance=r.distance / kw_max,
                        metadata=r.metadata, realm=r.realm, domain=r.domain,
                    )
                    for r in kw_results
                ]

        seen = set()
        combined: list[SearchResult] = []
        for r in vec_results:
            seen.add(r.id)
            combined.append(r)
        for r in kw_results:
            if r.id not in seen:
                combined.append(r)

        combined.sort(key=lambda x: -x.distance)

        # Node boost — tiered by node reference count
        source_files = set()
        for r in combined:
            src = r.metadata.get("source_file") if r.metadata else None
            if src:
                source_files.add(src)
        if source_files:
            boosted = self._get_node_source_boosts(source_files)
            if boosted:
                for i, r in enumerate(combined):
                    boost_val = boosted.get(r.id, 0.0)
                    if boost_val > 0:
                        combined[i] = SearchResult(
                            id=r.id, text=r.text,
                            distance=r.distance + boost_val,
                            metadata=r.metadata, realm=r.realm, domain=r.domain,
                        )

        combined.sort(key=lambda x: -x.distance)
        return combined[:n_results]

    NODE_RANK_BOOSTS = [0.40, 0.25, 0.15, 0.08, 0.04]

    def _get_node_source_boosts(self, source_files: set[str]) -> dict[str, float]:
        """Return dict mapping entity_id -> boost_value based on node reference count rank.

        Entities from source_files with more node references get higher
        boosts. Up to 5 tiers, then a flat floor of the last value.
        """
        if not source_files:
            return {}
        placeholders = ",".join("?" * len(source_files))
        rows = self._db_execute(
            f"SELECT e.id, COUNT(c.id) as ref_count FROM entities e "
            f"INNER JOIN nodes c ON c.source_file = e.source_file "
            f"WHERE c.source_file IN ({placeholders}) "
            f"GROUP BY e.id ORDER BY ref_count DESC",
            list(source_files),
        ).fetchall()
        boosts: dict[str, float] = {}
        for rank, (eid, _) in enumerate(rows):
            boosts[eid] = self.NODE_RANK_BOOSTS[rank] if rank < len(self.NODE_RANK_BOOSTS) else self.NODE_RANK_BOOSTS[-1]
        return boosts

    def _get_node_source_ids(self, source_files: set[str]) -> set[str]:
        """Return entity IDs whose source_file is referenced in nodes."""
        if not source_files:
            return set()
        placeholders = ",".join("?" * len(source_files))
        rows = self._db_execute(
            f"SELECT DISTINCT e.id FROM entities e INNER JOIN nodes c ON c.source_file = e.source_file "
            f"WHERE c.source_file IN ({placeholders})",
            list(source_files),
        ).fetchall()
        return {r[0] for r in rows}

    # -- FTS maintenance --

    def rebuild_fts(self) -> None:
        """Rebuild FTS index from all entity contents."""
        with self._lock:
            self._db.execute("DELETE FROM entities_fts")
            rows = self._db.execute("SELECT id, content FROM entities").fetchall()
            self._db.executemany("INSERT INTO entities_fts (id, content) VALUES (?, ?)",
                                 rows)
            self._db.commit()
        logger.info("Rebuilt FTS index with %d entities", len(rows))

    # -- Status --

    def status(self) -> dict:
        if not self._initialized:
            return {"initialized": False}
        realms = self.list_realms()
        total_entities = self._db_execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        total_domains = self._db_execute("SELECT COUNT(*) FROM domains").fetchone()[0]
        embedder_name = getattr(self._embedder, "name", "unknown") if self._embedder else "none"
        return {"initialized": True, "path": str(self._base), "realms": len(realms),
                "domains": total_domains, "entities": total_entities,
                "realms_detail": realms, "embedding": embedder_name,
                "fts_enabled": self._fts_enabled}

    def diagnose(self) -> dict:
        """Return state diagnosis: status, message, and suggested action."""
        result = {"status": "unknown", "message": "", "action": ""}
        if not self._base.exists():
            result["status"] = "missing"
            result["message"] = f"No dimension at {self._base}"
            result["action"] = "Run dimension.init() to create"
            return result
        db_path = self._base / "dimension.db"
        if not db_path.exists():
            result["status"] = "not_initialized"
            result["message"] = "Dimension directory exists but dimension.db not found"
            result["action"] = "Run dimension.init() to initialize"
            return result
        try:
            count = self._db_execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        except Exception:
            result["status"] = "unreadable"
            result["message"] = "dimension.db exists but is unreadable"
            result["action"] = "Check file permissions or run repair"
            return result
        if count == 0:
            result["status"] = "empty"
            result["message"] = "Dimension has no entities yet"
            result["action"] = "Run mining to populate"
            return result
        vec_count = self._store.count() if self._store else 0
        if vec_count == 0:
            result["status"] = "store_empty"
            result["message"] = f"SQLite has {count} entities but vector store is empty"
            result["action"] = "Run rebuild_index or rebuild_from_sqlite"
            return result
        if count != vec_count:
            result["status"] = "diverged"
            result["message"] = f"SQLite has {count} entities but vector store has {vec_count}"
            result["action"] = "Run scan_dimension to find corrupt IDs, then prune"
            return result
        result["status"] = "healthy"
        result["message"] = f"Dimension healthy: {count} entities, {vec_count} vectors"
        return result

    # -- Agent record --

    def record_write(self, agent_name: str, entry: str, topic: str = "general",
                    realm: str = "") -> str:
        target_realm = realm or f"agent_{_sanitize(agent_name)}"
        self.add_entity(realm=target_realm, domain="record", content=entry,
                        metadata={"agent": agent_name, "topic": topic, "type": "record"})
        return target_realm

    def record_read(self, agent_name: str, last_n: int = 10, realm: str = "") -> list[dict]:
        target_realm = realm or f"agent_{_sanitize(agent_name)}"
        return self.list_entities(realm=target_realm, domain="record", limit=last_n)

    # -- Duplicate check --

    def check_duplicate(self, content: str, threshold: float = 0.9) -> Optional[dict]:
        if self._store.count() == 0:
            return None
        query_emb = self._embedder.embed([content])[0]
        ids, texts, distances, metadatas = self._store.search(query_emb, n_results=1)
        if ids and distances[0] >= threshold:
            return {"id": ids[0], "text": texts[0], "similarity": distances[0],
                    "metadata": metadatas[0] if metadatas else {}}
        return None

    # -- Taxonomy --

    def get_taxonomy(self) -> dict:
        """Full taxonomy tree: realm → domain → entity_count."""
        rows = self._db_execute(
            "SELECT realm, domain, COUNT(*) as cnt FROM entities GROUP BY realm, domain ORDER BY realm, domain"
        ).fetchall()
        tree = {}
        for realm, domain, cnt in rows:
            tree.setdefault(realm, {})[domain] = cnt
        return tree

    # -- Reconnect --

    def reconnect(self) -> bool:
        """Re-initialize database, FAISS, and KG connections in-place."""
        with self._lock:
            self._save_embedder()
            if self._store:
                self._store.close()
            if self._kg:
                self._kg.close()
            if self._db:
                self._db.close()
            self._db = sqlite3.connect(str(self._base / "dimension.db"), check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._fts_enabled = bool(self._db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='entities_fts'"
            ).fetchone())
            self._store = self._create_store()
            self._embedder = self._load_embedder()
            self._store.dimension = self._embedder.dimension
            self._kg = KnowledgeGraph(str(self._data_dir))
            self._initialized = True
        return True

    # -- Embedder management --

    _EMBEDDER_CONFIG_NAMES: dict[str, str] = {
        "numpy": "numpy_tfidf_svd",
        "numpy_tfidf_svd": "numpy_tfidf_svd",
        "sentence": "sentence_transformers",
        "sentence_transformers": "sentence_transformers",
        "spacy": "spacy_glove",
        "spacy_glove": "spacy_glove",
        "minilm": "minilm",
        "onnx": "minilm",
        "embeddinggemma": "embeddinggemma",
        "gemma": "embeddinggemma",
        "bert": "bert",
        "numpy_bert": "bert",
    }

    def set_embedder(self, model: str, reindex: bool = True, device: Optional[str] = None) -> dict:
        """Switch to a different embedding model.

        ``model`` accepts short names (``"sentence"``, ``"spacy"``,
        ``"numpy"``, ``"minilm"``, ``"embeddinggemma"``) or config names
        (``"sentence_transformers"``, ``"spacy_glove"``, ``"numpy_tfidf_svd"``).

        ``device`` controls the ONNX runtime device (``auto``, ``cpu``,
        ``cuda``, ``coreml``, ``dml``). Falls back to config
        ``embedding_device`` if not provided.

        When ``reindex=True``, all existing entities are re-embedded with
        the new model and the FAISS index is rebuilt.
        """
        config_name = self._EMBEDDER_CONFIG_NAMES.get(model)
        if config_name is None:
            valid = sorted(set(self._EMBEDDER_CONFIG_NAMES.values()))
            raise ValueError(
                f"Unknown embedder model: {model!r}. "
                f"Valid options: {valid}"
            )

        # Write new config
        config_path = self._base / "dimension.json"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        config["embedding"] = config_name
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Load new embedder
        self._embedder = self._load_embedder(device=device)

        info = {
            "model": config_name,
            "dimension": self._embedder.dimension,
        }

        if reindex:
            count = self._reindex_embeddings()
            info["reindexed"] = count
        else:
            logger.warning(
                "set_embedder(reindex=False): embedder changed without reindexing; "
                "existing vectors will be mismatched with the new model"
            )

        return info

    def _reindex_embeddings(self) -> int:
        """Re-embed all entities with the current embedder and rebuild store.

        Creates a backup of store files before clearing so a failed embed
        can be rolled back. Removes the backup on success.
        """
        rows = self._db_execute(
            "SELECT id, content, metadata, source_file FROM entities"
        ).fetchall()
        if not rows:
            return 0

        ids = [r[0] for r in rows]
        contents = [r[1] for r in rows]
        vectors = self._embedder.embed(contents)
        metadatas = []
        for r in rows:
            md = json.loads(r[2]) if isinstance(r[2], str) and r[2] else {}
            md["source_file"] = r[3] or ""
            metadatas.append(md)

        if self._store:
            self._store.close()

        import shutil
        backup_dir = self._data_dir.parent / ".reindex_backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        for fname in _STORE_BACKUP_FILES:
            src = self._data_dir / fname
            if src.exists():
                shutil.copy2(src, backup_dir / fname)

        try:
            for fname in _STORE_BACKUP_FILES:
                (self._data_dir / fname).unlink(missing_ok=True)

            self._store = self._create_store()
            self._store.add(ids, contents, metadatas, vectors)
        except Exception:
            for fname in _STORE_BACKUP_FILES:
                bak = backup_dir / fname
                if bak.exists():
                    shutil.copy2(bak, self._data_dir / fname)
                else:
                    (self._data_dir / fname).unlink(missing_ok=True)
            self._store = self._create_store()
            raise
        finally:
            if backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)

        return len(ids)

    # -- Memories filed away --

    def memories_filed_away(self) -> dict:
        """Check when the last memory was saved and total counts."""
        last = self._db_execute(
            "SELECT created_at, content FROM entities ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        total = self._db_execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        preview = None
        last_time = None
        if last:
            try:
                last_time = last["created_at"]
                content = last["content"]
            except (TypeError, IndexError):
                last_time = last[0]
                content = last[1] if len(last) > 1 else ""
            if content:
                preview = (content[:100] + "...") if len(content) > 100 else content
        return {
            "total_entities": total,
            "last_saved_at": last_time,
            "last_content_preview": preview,
        }


# ==================== Closet-line entity extraction (for record_ingest) ====================


def _candidate_entity_words(text: str) -> list[str]:
    from alt_memory.entity_detector import _extract_candidate_words
    result = _extract_candidate_words(text, min_len=1, deduplicate=True)
    return result


_DATE_LINE = re.compile(
    r"^(\s*(?:[-*]\s+)?)"                          # optional list prefix
    r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}"                # YYYY-MM-DD or YYYY/MM/DD
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{4})"               # MM/DD/YYYY or DD/MM/YYYY
    r"(\s*[:\-]\s*.+)?$",                           # optional text after colon/dash
    re.MULTILINE,
)


def _build_date_line_segment(text: str) -> str:
    """Collapse date-prefixed lines into a compact entry."""
    lines = text.strip().splitlines()
    clean: list[str] = []
    for line in lines:
        m = _DATE_LINE.match(line)
        if m:
            prefix = m.group(1) or ""
            date = m.group(2) or ""
            rest = (m.group(3) or "").strip()
            if rest:
                rest = rest.lstrip(":-\t ")
            clean.append(f"{prefix}{date} | {rest}".strip())
        else:
            clean.append(line.strip())
    return "\n".join(clean)


def build_node_lines(
    text: str,
    existing: dict[str, str],
    source_line: Optional[str] = None,
    drawer_ids: Optional[list[str]] = None,
) -> list[dict]:
    """Build node-line entries from text, grouping named-entity snippets.

    Parameters
    ----------
    text : str
        Source text to mine.
    existing : dict[str, str]
        Existing node lines keyed by entity name — maps to their current
        accumulated content.
    source_line : str, optional
        Optional source-file line for provenance.

    Returns
    -------
    list[dict]
        List of ``{entity: str, content: str}`` dicts. These should be
        upserted by the caller.
    """
    if not text.strip():
        return [{"entity": "_meta", "content": "\n"}]

    entities = _candidate_entity_words(text)

    # If no entities found, fall back to "_meta" with collapsed dates.
    if not entities:
        collapsed = _build_date_line_segment(text)
        if len(collapsed) > NODE_CHAR_LIMIT:
            collapsed = collapsed[:NODE_CHAR_LIMIT]
        existing_meta = existing.get("_meta", "")
        merged = existing_meta + "\n" + collapsed if existing_meta else collapsed
        if len(merged) > NODE_CHAR_LIMIT:
            merged = merged[-NODE_CHAR_LIMIT:]
        return [{"entity": "_meta", "content": merged.strip()}]

    result: list[dict] = []
    seen_entities: set[str] = set()
    for e in entities:
        if e in seen_entities:
            continue
        seen_entities.add(e)
        match = re.search(r'\b' + re.escape(e) + r'\b', text, re.IGNORECASE)
        pos = match.start() if match else text.lower().find(e.lower())
        window_start = max(0, pos - NODE_EXTRACT_WINDOW)
        window_end = min(len(text), window_start + 2 * NODE_EXTRACT_WINDOW + len(e))
        snippet = text[window_start:window_end].strip()
        if len(snippet) > NODE_CHAR_LIMIT:
            snippet = snippet[:NODE_CHAR_LIMIT]
        existing_content = existing.get(e, "")
        merged = existing_content + "\n" + snippet if existing_content else snippet
        if len(merged) > NODE_CHAR_LIMIT:
            merged = merged[-NODE_CHAR_LIMIT:]
        result.append({"entity": e, "content": merged.strip()})
    return result


# ==================== Standalone collection helpers (for node_llm etc.) ====================


def _open_dimension_db(dim_path: str) -> sqlite3.Connection:
    base = Path(dim_path).expanduser().resolve()
    conn = sqlite3.connect(str(base / "dimension.db"), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY, content TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            source_file TEXT DEFAULT '', realm TEXT DEFAULT '', domain TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')))""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_source ON nodes(source_file)")
    conn.commit()
    return conn


class _EntityCollection:
    """Minimal ChromaDB-compatible wrapper around the entities SQLite table."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]

    def get(self, limit: int = 5000, offset: int = 0, include: Optional[list[str]] = None):
        rows = self._conn.execute(
            "SELECT id, content, metadata FROM entities ORDER BY created_at LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return {
            "ids": [r[0] for r in rows],
            "documents": [r[1] for r in rows] if (not include or "documents" in include) else [],
            "metadatas": [json.loads(r[2] or "{}") for r in rows]
            if (not include or "metadatas" in include)
            else [],
        }


class _NodeCollection:
    """Minimal ChromaDB-compatible wrapper around the nodes SQLite table."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def delete(self, where: Optional[dict] = None) -> None:
        if where and "source_file" in where:
            self._conn.execute("DELETE FROM nodes WHERE source_file = ?", (where["source_file"],))
            self._conn.commit()

    def upsert(self, documents: list[str], ids: list[str], metadatas: Optional[list[dict]] = None) -> None:
        if metadatas is None:
            metadatas = [{} for _ in range(len(documents))]
        for doc, cid, meta in zip(documents, ids, metadatas):
            self._conn.execute(
                "INSERT OR REPLACE INTO nodes (id, content, metadata, source_file, realm, domain) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    cid,
                    doc,
                    json.dumps(meta),
                    meta.get("source_file", ""),
                    meta.get("realm", ""),
                    meta.get("domain", ""),
                ),
            )
        self._conn.commit()


def get_collection(
    dim_path: str,
    collection_name: Optional[str] = None,
    create: bool = True,
) -> _EntityCollection:
    _ = collection_name  # kept for API compat; we only have one collection
    conn = _open_dimension_db(dim_path)
    return _EntityCollection(conn)


def get_nodes_collection(dim_path: str, create: bool = True) -> _NodeCollection:
    conn = _open_dimension_db(dim_path)
    return _NodeCollection(conn)


def _metadata_matches_extract_mode(meta: dict, extract_mode: Optional[str]) -> bool:
    if extract_mode is None:
        return True
    stored_mode = meta.get("extract_mode")
    return stored_mode == extract_mode or (extract_mode == "exchange" and stored_mode is None)


def file_already_mined(
    db_or_conn,
    source_file: str,
    check_mtime: bool = False,
    extract_mode: Optional[str] = None,
) -> bool:
    """Check if a source file has already been mined.

    Returns False (so the file gets re-mined) when:
      - no nodes exist for this source_file
      - the stored ``normalize_version`` is missing or older than the current
        schema (triggers silent rebuild after a normalization upgrade)
      - ``check_mtime=True`` and the file's mtime differs from the stored one

    When ``extract_mode`` is set, idempotency is scoped to that extraction
    mode so exchange-mode and general-mode drawers can coexist for the same
    source transcript. Legacy drawers without extract_mode are treated as
    exchange-mode drawers.

    Parameters
    ----------
    db_or_conn :
        Either a Dimension instance (with ``_db`` attribute) or a raw sqlite3 Connection.
    source_file :
        File path to check.
    check_mtime :
        If True, also re-mine if the source file's mtime has changed.
    extract_mode :
        Optional mode scope for extraction mode idempotency.
    """
    try:
        if hasattr(db_or_conn, "_db"):
            conn = db_or_conn._db
        else:
            conn = db_or_conn
        rows = conn.execute(
            "SELECT metadata FROM nodes WHERE source_file = ?",
            (source_file,),
        ).fetchall()
        if not rows:
            return False
        for (meta_json,) in rows:
            meta = json.loads(meta_json or "{}")
            if extract_mode is not None and not _metadata_matches_extract_mode(meta, extract_mode):
                continue
            stored_version = meta.get("normalize_version", 1)
            if stored_version < NORMALIZE_VERSION:
                continue
            if not check_mtime:
                return True
            try:
                current_mtime = os.path.getmtime(source_file)
            except OSError:
                return True
            stored_mtime = meta.get("source_mtime")
            if stored_mtime is not None and abs(float(stored_mtime) - current_mtime) < 0.001:
                return True
        return False
    except (sqlite3.Error, ValueError, json.JSONDecodeError):
        return False


def purge_file_nodes(nodes_col, source_file: str) -> int:
    try:
        result = nodes_col.delete(where={"source_file": source_file})
        return len(result[0]) if result and result[0] else 0
    except Exception:
        logger.debug("Closet purge failed for %s", source_file, exc_info=True)
        return 0


def upsert_node_lines(nodes_col, node_id_base, lines, metadata):
    node_num = 1
    current_lines: list = []
    current_chars = 0
    nodes_written = 0

    def _flush():
        nonlocal nodes_written
        if not current_lines:
            return
        node_id = f"{node_id_base}_{node_num:02d}"
        text = "\n".join(current_lines)
        nodes_col.upsert(documents=[text], ids=[node_id], metadatas=[metadata])
        nodes_written += 1

    for line in lines:
        line_len = len(line)
        if current_chars > 0 and current_chars + line_len + 1 > NODE_CHAR_LIMIT:
            _flush()
            node_num += 1
            current_lines = []
            current_chars = 0
        current_lines.append(line)
        current_chars += line_len + 1
    _flush()
    return nodes_written
