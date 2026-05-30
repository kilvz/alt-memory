"""MCP (Model Context Protocol) server for Alt Memory.
 
Exposes all Dimension operations as JSON-RPC tools over stdio or SSE transport.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Optional

try:
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse

    HAS_HTTP = True
except ImportError:
    HAS_HTTP = False

from alt_memory import dialect
from alt_memory.dialect import aaak_compress, aaak_decompress, aaak_parse_entry
from alt_memory.entity_registry import EntityRegistry
from alt_memory import dim_graph
from alt_memory.sync import sync_dimension
from alt_memory.config import AltMemoryConfig
from alt_memory.dimension import Dimension

try:
    from alt_memory.layers import MemoryStack

    HAS_LAYERS = True
except ImportError:
    HAS_LAYERS = False

try:
    from alt_memory import miner

    HAS_MINER = True
except ImportError:
    HAS_MINER = False


logger = logging.getLogger(__name__)

_WAL_REDACT_KEYS = frozenset(
    {"content", "content_preview", "document", "entry", "entry_preview", "query", "text"}
)
_WAL_FILE = os.path.join(
    os.environ.get("ALT_MEMORY_HOME", os.path.expanduser("~/.alt-memory")),
    "mcp_wal.jsonl",
)


def _init_mcp_logging() -> None:
    """Init logging: stderr-by-default, optionally append to ALT_MEMORY_LOG_FILE."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    log_file = os.environ.get("ALT_MEMORY_LOG_FILE", "").strip()
    file_handler_error: Exception | None = None
    if log_file:
        try:
            handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
        except (OSError, ValueError) as exc:
            file_handler_error = exc
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=handlers, force=True,
    )
    if file_handler_error is not None:
        logger.warning(
            "ALT_MEMORY_LOG_FILE=%r could not be opened (%s); using stderr only",
            log_file, file_handler_error,
        )


_init_mcp_logging()


def _wal_log(dimension: str, method: str, params: dict, result: Any = None, error: str = "") -> None:
    safe_params = {}
    for k, v in params.items():
        if any(rk in k for rk in _WAL_REDACT_KEYS):
            safe_params[k] = f"[REDACTED {len(v)} chars]" if isinstance(v, str) else "[REDACTED]"
        else:
            safe_params[k] = v
    entry = {
        "timestamp": datetime.now().isoformat(),
        "dimension": dimension,
        "method": method,
        "params": safe_params,
        "result_status": "ok" if not error else "error",
        "error": error,
    }
    try:
        with open(_WAL_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        logger.error("WAL write failed: %s", e)


# Idle watchdog -- stale MCP server auto-exit
_MCP_IDLE_HOURS_ENV = "ALT_MEMORY_MCP_IDLE_HOURS"
_MCP_IDLE_HOURS_DEFAULT = 8.0
_last_request_time: float = time.monotonic()


def _mcp_idle_timeout_secs() -> float:
    raw = os.environ.get(_MCP_IDLE_HOURS_ENV, "")
    if raw:
        try:
            hours = float(raw)
            return max(0.0, hours) * 3600
        except ValueError:
            return 0.0
    return _MCP_IDLE_HOURS_DEFAULT * 3600


def _start_idle_exit_watchdog() -> None:
    timeout = _mcp_idle_timeout_secs()
    if timeout <= 0:
        return
    check_interval = min(60.0, timeout / 4)

    def _watchdog() -> None:
        while True:
            time.sleep(check_interval)
            idle = time.monotonic() - _last_request_time
            if idle >= timeout:
                logger.info(
                    "MCP server idle for %.1f h (limit %.1f h); exiting to release file handles.",
                    idle / 3600, timeout / 3600,
                )
                os._exit(0)

    t = threading.Thread(target=_watchdog, name="mcp-idle-watchdog", daemon=True)
    t.start()


def _maybe_eager_warmup_embedder(dim: Dimension) -> None:
    raw = os.environ.get("ALT_MEMORY_EAGER_WARMUP", "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return
    if raw not in ("1", "true", "yes", "on"):
        logger.warning(
            "ALT_MEMORY_EAGER_WARMUP=%r not recognized (use 1/true/yes/on); warmup disabled", raw
        )
        return
    try:
        dim.search("warmup", n_results=1)
        logger.info("Eager warmup: embedder ready (dim=%s)", dim._base)
    except Exception as exc:
        logger.exception("Eager warmup failed (dim=%s): %s", dim._base, exc)


VERSION = "4.3.0"
MCP_PROTOCOL_VERSION = "2024-11-05"

_TOOL_DEFINITIONS: list[dict] = []

AAAK_SPEC = (
    "AAAK is a compressed memory dialect that Alt Memory uses for efficient storage. "
    "It is designed to be readable by both humans and LLMs without decoding.\n\n"
    "FORMAT:\n"
    "  ENTITIES: 3-letter uppercase codes. ALC=Alice, JOR=Jordan, RIL=Riley, MAX=Max, BEN=Ben.\n"
    "  EMOTIONS: *action markers* before/during text. *warm*=joy, *fierce*=determined, *raw*=vulnerable, *bloom*=tenderness.\n"
    "  STRUCTURE: Pipe-separated fields. FAM: family | PROJ: projects | ⚠: warnings/reminders.\n"
    "  DATES: ISO format (2026-03-31). COUNTS: Nx = N mentions (e.g., 570x).\n"
    "  IMPORTANCE: ★ to ★★★★★ (1-5 scale).\n"
    "  GATES: gate_facts, gate_events, gate_discoveries, gate_preferences, gate_advice.\n"
    "  WINGS: alt_memory, documents, reference, benchmark, agent_*\n"
    "  ROOMS: Hyphenated slugs representing named ideas (e.g., chromadb-setup, gpu-pricing).\n\n"
    "EXAMPLE:\n"
    "  FAM: ALC→♡JOR | 2D(kids): RIL(18,sports) MAX(11,chess+swimming) | BEN(contributor)\n\n"
    "Read AAAK naturally — expand codes mentally, treat *markers* as emotional context.\n"
    "When WRITING AAAK: use entity codes, mark emotions, keep structure tight."
)


def _build_tool_definitions() -> list[dict]:
    """Build MCP tool definitions from the registered methods."""
    if _TOOL_DEFINITIONS:
        return _TOOL_DEFINITIONS
    tools = [
        {
            "name": "search",
            "description": "Search across all dimension entities using hybrid (vector+keyword), vector-only, or keyword-only mode",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query text"},
                    "n_results": {"type": "integer", "description": "Max results to return (default 10)"},
                    "realm": {"type": "string", "description": "Restrict to a realm"},
                    "domain": {"type": "string", "description": "Restrict to a domain"},
                    "mode": {"type": "string", "description": "Search mode: hybrid, vector, or keyword"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_status",
            "description": "Get dimension status: entity count, realm/domain breakdown, embedding type",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "list_realms",
            "description": "List all realms in the dimension",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "create_realm",
            "description": "Create a new realm",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Realm name"},
                    "description": {"type": "string", "description": "Optional description"},
                },
                "required": ["name"],
            },
        },
        {
            "name": "delete_realm",
            "description": "Delete a realm and all its domains and entities",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Realm name"},
                },
                "required": ["name"],
            },
        },
        {
            "name": "list_domains",
            "description": "List domains in a realm",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "realm": {"type": "string", "description": "Realm name (optional; lists all domains if omitted)"},
                },
            },
        },
        {
            "name": "create_domain",
            "description": "Create a new domain in a realm",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "realm": {"type": "string", "description": "Realm name"},
                    "name": {"type": "string", "description": "Domain name"},
                    "description": {"type": "string", "description": "Optional description"},
                },
                "required": ["realm", "name"],
            },
        },
        {
            "name": "delete_domain",
            "description": "Delete a domain from a realm",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "realm": {"type": "string", "description": "Realm name"},
                    "name": {"type": "string", "description": "Domain name"},
                },
                "required": ["realm", "name"],
            },
        },
        {
            "name": "add_entity",
            "description": "Add a new entity with content to a realm/domain",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "realm": {"type": "string", "description": "Realm name"},
                    "domain": {"type": "string", "description": "Domain name"},
                    "content": {"type": "string", "description": "Entity content text"},
                    "metadata": {"type": "object", "description": "Optional metadata dict"},
                    "source_file": {"type": "string", "description": "Optional source file path"},
                    "entity_id": {"type": "string", "description": "Optional entity ID (auto-generated if omitted)"},
                },
                "required": ["realm", "domain", "content"],
            },
        },
        {
            "name": "get_entity",
            "description": "Get an entity by its ID",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string", "description": "Entity ID"},
                },
                "required": ["entity_id"],
            },
        },
        {
            "name": "delete_entity",
            "description": "Delete an entity by its ID",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string", "description": "Entity ID"},
                },
                "required": ["entity_id"],
            },
        },
        {
            "name": "kg_add",
            "description": "Add a fact to the knowledge graph",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "Subject entity"},
                    "predicate": {"type": "string", "description": "Relationship type"},
                    "object": {"type": "string", "description": "Object entity"},
                    "valid_from": {"type": "string", "description": "ISO date when fact becomes valid"},
                    "valid_to": {"type": "string", "description": "ISO date when fact expires"},
                    "source": {"type": "string", "description": "Optional source identifier"},
                },
                "required": ["subject", "predicate", "object"],
            },
        },
        {
            "name": "kg_query",
            "description": "Query the knowledge graph for facts about an entity",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "Entity name to query"},
                    "predicate": {"type": "string", "description": "Filter by predicate"},
                    "as_of": {"type": "string", "description": "ISO date to query temporally"},
                    "all": {"type": "boolean", "description": "Return all facts"},
                    "direction": {"type": "string", "description": "Query direction: outgoing, incoming, both (default both)"},
                },
            },
        },
        {
            "name": "kg_invalidate",
            "description": "Mark a knowledge graph fact as no longer true",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "Subject entity"},
                    "predicate": {"type": "string", "description": "Relationship type"},
                    "object": {"type": "string", "description": "Object entity"},
                    "ended": {"type": "string", "description": "ISO date when fact stopped being true (default today)"},
                },
                "required": ["subject", "predicate", "object"],
            },
        },
        {
            "name": "kg_stats",
            "description": "Get knowledge graph statistics",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "record_write",
            "description": "Write a record entry for an agent",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "description": "Agent name"},
                    "entry": {"type": "string", "description": "Record entry text"},
                    "topic": {"type": "string", "description": "Topic tag"},
                    "realm": {"type": "string", "description": "Target realm (default agent_<name>)"},
                },
                "required": ["agent", "entry"],
            },
        },
        {
            "name": "record_read",
            "description": "Read recent record entries for an agent",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "description": "Agent name"},
                    "last_n": {"type": "integer", "description": "Number of entries to read"},
                    "realm": {"type": "string", "description": "Realm to read from (default agent_<name>)"},
                },
                "required": ["agent"],
            },
        },
        {
            "name": "list_entities",
            "description": "List entities with optional realm/domain filter and pagination",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "realm": {"type": "string", "description": "Filter by realm"},
                    "domain": {"type": "string", "description": "Filter by domain"},
                    "limit": {"type": "integer", "description": "Max results (default 20)"},
                    "offset": {"type": "integer", "description": "Offset for pagination"},
                },
            },
        },
        {
            "name": "mine_text",
            "description": "Mine text content into the dimension",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text content to mine"},
                    "realm": {"type": "string", "description": "Target realm"},
                    "domain": {"type": "string", "description": "Target domain"},
                    "source": {"type": "string", "description": "Optional source identifier"},
                    "chunk": {"type": "boolean", "description": "Chunk long text into multiple entities (default true)"},
                },
                "required": ["text", "realm", "domain"],
            },
        },
        {
            "name": "aaak_compress",
            "description": "Compress text using AAAK dialect",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to compress"},
                    "max_len": {"type": "integer", "description": "Maximum output length (default 500)"},
                },
                "required": ["text"],
            },
        },
        {
            "name": "aaak_decompress",
            "description": "Decompress AAAK-encoded text",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "AAAK text to decompress"},
                },
                "required": ["text"],
            },
        },
        {
            "name": "aaak_parse",
            "description": "Parse a single AAAK entry into its components",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "AAAK entry text"},
                },
                "required": ["text"],
            },
        },
    ]
    _TOOL_DEFINITIONS.extend(tools)

    # -- Additional tools --

    extra_tools = [
        {
            "name": "update_entity",
            "description": "Update an existing entity's content and/or metadata (realm, domain)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string", "description": "Entity ID"},
                    "content": {"type": "string", "description": "New content (optional)"},
                    "metadata": {"type": "object", "description": "New metadata (optional)"},
                    "realm": {"type": "string", "description": "New realm (optional)"},
                    "domain": {"type": "string", "description": "New domain (optional)"},
                },
                "required": ["entity_id"],
            },
        },
        {
            "name": "check_duplicate",
            "description": "Check if content already exists in the dimension before filing",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to check"},
                    "threshold": {"type": "number", "description": "Similarity threshold 0-1 (default 0.9)"},
                },
                "required": ["content"],
            },
        },
        {
            "name": "kg_timeline",
            "description": "Chronological timeline of facts. Shows the story of an entity (or everything) in order.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "Entity to get timeline for (optional — omit for full timeline)"},
                },
            },
        },
        {
            "name": "create_tunnel",
            "description": "Create a cross-realm tunnel linking two dimension locations",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_realm": {"type": "string", "description": "Realm of the source"},
                    "source_domain": {"type": "string", "description": "Domain in the source realm"},
                    "target_realm": {"type": "string", "description": "Realm of the target"},
                    "target_domain": {"type": "string", "description": "Domain in the target realm"},
                    "label": {"type": "string", "description": "Description of the connection"},
                    "source_entity_id": {"type": "string", "description": "Optional specific entity ID"},
                    "target_entity_id": {"type": "string", "description": "Optional specific entity ID"},
                },
                "required": ["source_realm", "source_domain", "target_realm", "target_domain"],
            },
        },
        {
            "name": "delete_tunnel",
            "description": "Delete an explicit tunnel by its entity ID",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tunnel_id": {"type": "string", "description": "Tunnel ID to delete"},
                },
                "required": ["tunnel_id"],
            },
        },
        {
            "name": "find_tunnels",
            "description": "Find domains that bridge two realms — the gateways connecting different domains",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "realm_a": {"type": "string", "description": "First realm (optional)"},
                    "realm_b": {"type": "string", "description": "Second realm (optional)"},
                },
            },
        },
        {
            "name": "follow_tunnels",
            "description": "Follow tunnels from a domain to see what it connects to in other realms",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "realm": {"type": "string", "description": "Realm to start from"},
                    "domain": {"type": "string", "description": "Domain to follow tunnels from"},
                },
                "required": ["realm", "domain"],
            },
        },
        {
            "name": "list_tunnels",
            "description": "List all explicit cross-realm tunnels. Optionally filter by realm.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "realm": {"type": "string", "description": "Filter tunnels by realm"},
                },
            },
        },
        {
            "name": "traverse",
            "description": "Walk the dimension graph from a domain. Shows connected ideas across realms.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "start_domain": {"type": "string", "description": "Domain to start from"},
                    "max_hops": {"type": "integer", "description": "How many connections to follow (default: 2)"},
                },
                "required": ["start_domain"],
            },
        },
        {
            "name": "graph_stats",
            "description": "Dimension graph overview: total domains, tunnel connections, edges between realms.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "get_taxonomy",
            "description": "Full taxonomy: realm → domain → entity count",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "get_aaak_spec",
            "description": "Get the AAAK dialect specification — the compressed memory format",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "memories_filed_away",
            "description": "Check if a recent dimension checkpoint was saved. Returns entity count and timestamp.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "sync",
            "description": "Prune entities whose source files are gitignored, deleted, or moved",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_dir": {"type": "string", "description": "Project root to scope the sync"},
                    "realm": {"type": "string", "description": "Limit to one realm"},
                    "apply": {"type": "boolean", "description": "Actually delete entities; default is dry-run preview"},
                },
            },
        },
        {
            "name": "reconnect",
            "description": "Force reconnect to the dimension database. Use after external changes.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "hook_settings",
            "description": "Get or set hook behavior. silent_save: True = save directly, desktop_toast: True = show notification.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "silent_save": {"type": "boolean", "description": "True = silent direct save"},
                    "desktop_toast": {"type": "boolean", "description": "True = show desktop toast via notify-send"},
                },
            },
        },
        {
            "name": "mine_file",
            "description": "Mine a single file into the dimension (chunks, extracts metadata, stores entities)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to the file to mine"},
                    "realm": {"type": "string", "description": "Target realm"},
                    "domain": {"type": "string", "description": "Target domain"},
                },
                "required": ["filepath", "realm", "domain"],
            },
        },
        {
            "name": "batch_mine",
            "description": "Mine all matching files in a directory into the dimension",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory to scan for files"},
                    "realm": {"type": "string", "description": "Target realm (optional; auto-detected from dir name)"},
                    "pattern": {"type": "string", "description": "Glob pattern to filter files (e.g. *.md, *.txt)"},
                },
                "required": ["directory"],
            },
        },
        {
            "name": "rebuild_fts",
            "description": "Rebuild the FTS5 full-text search index from all entity contents",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "get_backend",
            "description": "Get the current vector store backend (faiss or chroma)",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "set_backend",
            "description": "Switch between vector store backends (faiss or chroma) and optionally reindex all entities",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "description": "Backend name: faiss or chroma",
                        "enum": ["faiss", "chroma"],
                    },
                    "reindex": {"type": "boolean", "description": "Re-embed all existing entities (default true)"},
                },
                "required": ["backend"],
            },
        },
        {
            "name": "set_embedder",
            "description": "Switch to a different embedding model and optionally reindex all entities",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Embedder name: sentence, spacy, numpy, minilm, embeddinggemma"},
                    "device": {"type": "string", "description": "ONNX device: auto, cpu, cuda, coreml, dml (default from config)"},
                    "reindex": {"type": "boolean", "description": "Re-embed all existing entities (default true)"},
                },
                "required": ["model"],
            },
        },
        {
            "name": "get_default_embedder",
            "description": "Get the default embedder model for new dimensions (global config)",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "set_default_embedder",
            "description": "Set the default embedder model for new dimensions (global config ~/.alt-memory/config.json)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Embedder name: sentence, spacy, numpy, minilm, embeddinggemma",
                        "enum": ["sentence", "spacy", "numpy", "minilm", "embeddinggemma"],
                    },
                },
                "required": ["model"],
            },
        },
        {
            "name": "list_agents",
            "description": "List all agent record writers in the dimension",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "get_people_map",
            "description": "Get the people map (name variant to canonical name mappings)",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "set_people_map",
            "description": "Set the people map (name variant to canonical name mappings)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "map": {
                        "type": "object",
                        "description": "Mapping of name variants to canonical names, e.g. {\"Alex\": \"Alexander\"}",
                    },
                },
                "required": ["map"],
            },
        },
        {
            "name": "batch_add_entities",
            "description": "Add multiple entities in a single transaction",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "description": "List of entity objects, each with realm, domain, content, and optional metadata/source_file/entity_id",
                        "items": {
                            "type": "object",
                            "properties": {
                                "realm": {"type": "string"},
                                "domain": {"type": "string"},
                                "content": {"type": "string"},
                                "metadata": {"type": "object"},
                                "source_file": {"type": "string"},
                                "entity_id": {"type": "string"},
                            },
                            "required": ["realm", "domain", "content"],
                        },
                    },
                },
                "required": ["entities"],
            },
        },
        {
            "name": "delete_entities",
            "description": "Delete multiple entities by their IDs in a single operation",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entity_ids": {
                        "type": "array",
                        "description": "List of entity IDs to delete",
                        "items": {"type": "string"},
                    },
                },
                "required": ["entity_ids"],
            },
        },
        {
            "name": "get_persona",
            "description": "Get the current active persona name",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "set_persona",
            "description": "Set the active persona, creating a persona_<name> realm if needed",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Persona name"},
                },
                "required": ["name"],
            },
        },
        {
            "name": "switch_persona",
            "description": "Alias for set_persona — switch to a different active persona",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Persona name"},
                },
                "required": ["name"],
            },
        },
        {
            "name": "import_entities",
            "description": "Import entities from a JSON-serializable list",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "description": "List of entity dicts (each must have realm, domain, content)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "realm": {"type": "string"},
                                "domain": {"type": "string"},
                                "content": {"type": "string"},
                                "metadata": {"type": "object"},
                                "source_file": {"type": "string"},
                                "entity_id": {"type": "string"},
                            },
                            "required": ["realm", "domain", "content"],
                        },
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "If true, use provided entity_id (overwrites existing); otherwise auto-generate",
                    },
                },
                "required": ["entities"],
            },
        },
        {
            "name": "export_collection",
            "description": "Export all entities as a JSON-serializable list",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "realm": {"type": "string", "description": "Filter by realm (optional)"},
                    "domain": {"type": "string", "description": "Filter by domain (optional)"},
                },
            },
        },
    ]
    _TOOL_DEFINITIONS.extend(extra_tools)
    return _TOOL_DEFINITIONS


class MCPServer:
    """JSON-RPC MCP server dispatching to Dimension operations."""

    def __init__(self, dim: Dimension):
        self.dim = dim
        self.entity_registry = EntityRegistry(dim)
        self.memory_stack = MemoryStack(dim) if HAS_LAYERS else None

        self._metadata_cache: Optional[dict] = None
        self._metadata_cache_time: float = 0.0
        self._metadata_cache_ttl: float = 5.0

        self._methods: dict[str, tuple] = {}
        self._register_all()

    def _invalidate_metadata_cache(self) -> None:
        self._metadata_cache = None
        self._metadata_cache_time = 0.0

    def _fetch_all_metadata(self) -> dict:
        now = time.monotonic()
        if self._metadata_cache is not None and (now - self._metadata_cache_time) < self._metadata_cache_ttl:
            return self._metadata_cache
        realms = self.dim.list_realms()
        domains = self.dim._db_execute(
            "SELECT name, realm, description, created_at FROM domains ORDER BY realm, name"
        ).fetchall()
        total = self.dim._db_execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        total_domains = len(domains)
        data = {
            "realms": realms,
            "domains": [{"name": r[0], "realm": r[1], "description": r[2], "created_at": r[3]} for r in domains],
            "total_entities": total,
            "total_domains": total_domains,
        }
        self._metadata_cache = data
        self._metadata_cache_time = now
        return data

    def _register(self, name: str, handler: callable, required: list[str] | None = None):
        self._methods[name] = (handler, required or [])

    def _register_all(self):
        # MCP protocol methods
        self._register("initialize", self._mcp_initialize, [])
        self._register("notifications/initialized", self._mcp_noop, [])
        self._register("tools/list", self._mcp_tools_list, [])
        self._register("tools/call", self._mcp_tools_call, ["name", "arguments"])

        # Legacy JSON-RPC methods
        self._register("ping", self._ping, [])
        self._register("get_status", self._get_status, [])
        self._register("init_dimension", self._init_dimension, [])
        self._register("close_dimension", self._close_dimension, [])

        self._register("list_realms", self._list_realms, [])
        self._register("create_realm", self._create_realm, ["name"])
        self._register("delete_realm", self._delete_realm, ["name"])

        self._register("list_domains", self._list_domains, [])
        self._register("create_domain", self._create_domain, ["realm", "name"])
        self._register("delete_domain", self._delete_domain, ["realm", "name"])

        self._register("add_entity", self._add_entity, ["realm", "domain", "content"])
        self._register("get_entity", self._get_entity, ["entity_id"])
        self._register("list_entities", self._list_entities, [])
        self._register("update_entity", self._update_entity, ["entity_id"])
        self._register("delete_entity", self._delete_entity, ["entity_id"])

        self._register("search", self._search, ["query"])
        self._register("check_duplicate", self._check_duplicate, ["content"])

        self._register("kg_add", self._kg_add, ["subject", "predicate", "object"])
        self._register("kg_query", self._kg_query, [])
        self._register("kg_invalidate", self._kg_invalidate, ["subject", "predicate", "object"])
        self._register("kg_stats", self._kg_stats, [])
        self._register("kg_timeline", self._kg_timeline, [])

        self._register("record_write", self._record_write, ["agent", "entry"])
        self._register("record_read", self._record_read, ["agent"])
        self._register("memory_write", self._memory_write, ["agent", "layer", "content"])
        self._register("memory_read", self._memory_read, ["agent"])
        self._register("memory_summarize", self._memory_summarize, ["agent"])

        self._register("mine_file", self._mine_file, ["filepath", "realm", "domain"])
        self._register("mine_text", self._mine_text, ["text", "realm", "domain"])
        self._register("batch_mine", self._batch_mine, ["directory"])

        self._register("aaak_compress", self._aaak_compress, ["text"])
        self._register("aaak_decompress", self._aaak_decompress, ["text"])
        self._register("aaak_parse", self._aaak_parse, ["text"])

        self._register("rebuild_fts", self._rebuild_fts, [])

        # New tools
        self._register("create_tunnel", self._create_tunnel, ["source_realm", "source_domain", "target_realm", "target_domain"])
        self._register("delete_tunnel", self._delete_tunnel, ["tunnel_id"])
        self._register("find_tunnels", self._find_tunnels, [])
        self._register("follow_tunnels", self._follow_tunnels, ["realm", "domain"])
        self._register("list_tunnels", self._list_tunnels, [])
        self._register("traverse", self._traverse, ["start_domain"])
        self._register("graph_stats", self._graph_stats, [])
        self._register("get_taxonomy", self._get_taxonomy, [])
        self._register("get_aaak_spec", self._get_aaak_spec, [])
        self._register("memories_filed_away", self._memories_filed_away, [])
        self._register("sync", self._sync, [])
        self._register("reconnect", self._reconnect, [])
        self._register("hook_settings", self._hook_settings, [])
        self._register("get_backend", self._get_backend, [])
        self._register("set_backend", self._set_backend, ["backend"])
        self._register("set_embedder", self._set_embedder, ["model"])
        self._register("get_default_embedder", self._get_default_embedder, [])
        self._register("set_default_embedder", self._set_default_embedder, ["model"])
        self._register("list_agents", self._list_agents, [])
        self._register("get_people_map", self._get_people_map, [])
        self._register("set_people_map", self._set_people_map, ["map"])

        self._register("batch_add_entities", self._batch_add_entities, ["entities"])
        self._register("delete_entities", self._delete_entities, ["entity_ids"])
        self._register("get_persona", self._get_persona, [])
        self._register("set_persona", self._set_persona, ["name"])
        self._register("switch_persona", self._switch_persona, ["name"])
        self._register("import_entities", self._import_entities, ["entities"])
        self._register("export_collection", self._export_collection, [])

    def handle_request(self, raw: str) -> str | None:
        """Process one JSON-RPC request line, return JSON response string.
        Returns None for notifications (no ``id`` field).
        """
        global _last_request_time
        _last_request_time = time.monotonic()

        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            return json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error: invalid JSON"}})

        req_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params", {})

        if not isinstance(params, dict):
            err = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Params must be a JSON object"}}
            return json.dumps(err)

        handler_info = self._methods.get(method)
        if handler_info is None:
            err = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}
            return json.dumps(err)

        handler, required = handler_info
        missing = [p for p in required if p not in params or params[p] is None or (isinstance(params[p], str) and not params[p].strip())]
        if missing:
            err = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"Missing required parameter(s): {', '.join(missing)}"}}
            return json.dumps(err)

        try:
            result = handler(params)
            if req_id is None:
                return None
            return json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result})
        except Exception as e:
            logger.exception("Error handling %s: %s", method, e)
            err = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32603, "message": str(e)}}
            if logger.isEnabledFor(logging.DEBUG):
                err["error"]["data"] = traceback.format_exc()
            return json.dumps(err)

    # -- MCP protocol handlers ---------------------------------------------------

    def _mcp_initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "alt-memory", "version": VERSION},
        }

    def _mcp_noop(self, params: dict) -> dict:
        return {}

    def _mcp_tools_list(self, params: dict) -> dict:
        return {"tools": _build_tool_definitions()}

    def _mcp_tools_call(self, params: dict) -> dict:
        name = params["name"]
        args = params.get("arguments", {})
        handler_info = self._methods.get(name)
        if handler_info is None:
            raise ValueError(f"Unknown tool: {name}")
        handler, required = handler_info

        # Whitelist arguments to declared schema properties only.
        # Prevents callers from spoofing internal params.
        import inspect

        try:
            sig = inspect.signature(handler)
            accepts_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
        except (ValueError, TypeError):
            accepts_var_keyword = False

        # Build schema properties map from _TOOL_DEFINITIONS
        schema_props = {}
        for td in _TOOL_DEFINITIONS:
            if td.get("name") == name:
                schema_props = td.get("inputSchema", {}).get("properties", {})
                break

        if not accepts_var_keyword:
            unknown = [k for k in args if k not in schema_props and k != "wait_for_previous"]
            if unknown:
                quoted = ", ".join(f"'{k}'" for k in unknown)
                word = "parameter" if len(unknown) == 1 else "parameters"
                raise ValueError(f"Unknown {word} {quoted} for tool {name}")
            args = {k: v for k, v in args.items() if k in schema_props}

        # Coerce argument types based on input_schema
        for key, value in list(args.items()):
            prop_schema = schema_props.get(key, {})
            declared_type = prop_schema.get("type")
            try:
                if declared_type == "integer" and not isinstance(value, int):
                    args[key] = int(value)
                elif declared_type == "number" and not isinstance(value, (int, float)):
                    args[key] = float(value)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid value for parameter '{key}'")
        args.pop("wait_for_previous", None)

        missing = [p for p in required if p not in args or args[p] is None or (isinstance(args[p], str) and not args[p].strip())]
        if missing:
            raise ValueError(f"Missing required parameter(s): {', '.join(missing)}")
        result = handler(args)
        if isinstance(result, str):
            text = result
        else:
            text = json.dumps(result, indent=2, ensure_ascii=False)
        return {"content": [{"type": "text", "text": text}]}

    # -- Handler implementations ------------------------------------------------

    def _ping(self, params: dict) -> dict:
        return {"pong": True, "version": VERSION}

    def _get_status(self, params: dict) -> dict:
        s = self.dim.status()
        cached = self._fetch_all_metadata()
        s["entities"] = cached["total_entities"]
        s["domains"] = cached["total_domains"]
        s["realms"] = len(cached["realms"])
        s["realms_detail"] = cached["realms"]
        return s

    def _init_dimension(self, params: dict) -> dict:
        return {"created": self.dim.init()}

    def _close_dimension(self, params: dict) -> dict:
        self.dim.close()
        return {"closed": True}

    def _list_realms(self, params: dict) -> list:
        return self._fetch_all_metadata()["realms"]

    def _create_realm(self, params: dict) -> dict:
        name = self.dim.get_or_create_realm(params["name"], params.get("description", ""))
        self._invalidate_metadata_cache()
        return {"name": name}

    def _delete_realm(self, params: dict) -> dict:
        result = {"deleted": self.dim.delete_realm(params["name"])}
        self._invalidate_metadata_cache()
        return result

    def _list_domains(self, params: dict) -> list:
        realm = params.get("realm")
        if realm:
            return self.dim.list_domains(realm=realm)
        return self._fetch_all_metadata()["domains"]

    def _create_domain(self, params: dict) -> dict:
        name = self.dim.get_or_create_domain(params["realm"], params["name"], params.get("description", ""))
        self._invalidate_metadata_cache()
        return {"name": name}

    def _delete_domain(self, params: dict) -> dict:
        result = {"deleted": self.dim.delete_domain(params["realm"], params["name"])}
        self._invalidate_metadata_cache()
        return result

    def _add_entity(self, params: dict) -> dict:
        _wal_log(str(self.dim._base), "add_entity", {
            "realm": params["realm"],
            "domain": params["domain"],
            "content": params["content"],
            "metadata": params.get("metadata"),
            "entity_id": params.get("entity_id"),
        })
        did = self.dim.add_entity(
            params["realm"],
            params["domain"],
            params["content"],
            metadata=params.get("metadata"),
            source_file=params.get("source_file", ""),
            entity_id=params.get("entity_id"),
        )
        self._invalidate_metadata_cache()
        return {"entity_id": did}

    def _get_entity(self, params: dict) -> dict | None:
        entity = self.dim.get_entity(params["entity_id"])
        if entity is None:
            return {"found": False}
        # source_file is the absolute filesystem path — reduce to basename
        # so MCP clients don't leak filesystem structure to untrusted agents
        if isinstance(entity, dict) and entity.get("source_file"):
            entity["source_file"] = Path(entity["source_file"]).name
        if isinstance(entity, dict) and entity.get("metadata", {}).get("source_file"):
            entity["metadata"]["source_file"] = Path(entity["metadata"]["source_file"]).name
        return entity

    def _list_entities(self, params: dict) -> list:
        return self.dim.list_entities(
            realm=params.get("realm"),
            domain=params.get("domain"),
            limit=params.get("limit", 20),
            offset=params.get("offset", 0),
        )

    def _update_entity(self, params: dict) -> dict:
        _wal_log(str(self.dim._base), "update_entity", {
            "entity_id": params["entity_id"],
            "content": params.get("content"),
            "metadata": params.get("metadata"),
            "realm": params.get("realm"),
            "domain": params.get("domain"),
        })
        ok = self.dim.update_entity(
            params["entity_id"],
            content=params.get("content"),
            metadata=params.get("metadata"),
            realm=params.get("realm"),
            domain=params.get("domain"),
        )
        self._invalidate_metadata_cache()
        return {"updated": ok}

    def _delete_entity(self, params: dict) -> dict:
        _wal_log(str(self.dim._base), "delete_entity", {"entity_id": params["entity_id"]})
        result = {"deleted": self.dim.delete_entity(params["entity_id"])}
        self._invalidate_metadata_cache()
        return result

    def _search(self, params: dict) -> list:
        results = self.dim.search(
            params["query"],
            n_results=params.get("n_results", 10),
            realm=params.get("realm"),
            domain=params.get("domain"),
            mode=params.get("mode", "hybrid"),
        )
        out = []
        for r in results:
            meta = (r.metadata or {}).copy()
            if meta.get("source_file"):
                meta["source_file"] = Path(meta["source_file"]).name
            out.append({
                "id": r.id,
                "text": r.text,
                "distance": r.distance,
                "metadata": meta,
                "realm": r.realm,
                "domain": r.domain,
            })
        return out

    def _check_duplicate(self, params: dict) -> dict:
        dup = self.dim.check_duplicate(params["content"], threshold=params.get("threshold", 0.9))
        if dup:
            return dup
        return {"duplicate": False}

    def _kg_add(self, params: dict) -> dict:
        _wal_log(str(self.dim._base), "kg_add", {
            "subject": params["subject"],
            "predicate": params["predicate"],
            "object": params["object"],
            "valid_from": params.get("valid_from"),
            "valid_to": params.get("valid_to"),
            "source": params.get("source", ""),
        })
        fid = self.dim.kg.add(
            params["subject"],
            params["predicate"],
            params["object"],
            valid_from=params.get("valid_from"),
            valid_to=params.get("valid_to"),
            source=params.get("source", ""),
        )
        return {"fact_id": fid}

    def _kg_query(self, params: dict) -> list:
        if params.get("all"):
            return self.dim.kg.query(as_of=params.get("as_of"))
        return self.dim.kg.query(
            entity=params.get("entity"),
            predicate=params.get("predicate"),
            as_of=params.get("as_of"),
            direction=params.get("direction", "both"),
        )

    def _kg_invalidate(self, params: dict) -> dict:
        _wal_log(str(self.dim._base), "kg_invalidate", {
            "subject": params["subject"],
            "predicate": params["predicate"],
            "object": params["object"],
            "ended": params.get("ended"),
        })
        n = self.dim.kg.invalidate(
            params["subject"],
            params["predicate"],
            params["object"],
            ended=params.get("ended"),
        )
        return {"invalidated": n}

    def _kg_stats(self, params: dict) -> dict:
        return self.dim.kg.stats()

    def _record_write(self, params: dict) -> dict:
        _wal_log(str(self.dim._base), "record_write", {
            "agent": params["agent"],
            "entry": params["entry"],
            "topic": params.get("topic", "general"),
            "realm": params.get("realm", ""),
        })
        wing = self.dim.record_write(
            params["agent"],
            params["entry"],
            topic=params.get("topic", "general"),
            realm=params.get("realm", ""),
        )
        self._invalidate_metadata_cache()
        return {"realm": wing}

    def _record_read(self, params: dict) -> list:
        return self.dim.record_read(
            params["agent"],
            last_n=params.get("last_n", 10),
            realm=params.get("realm", ""),
        )

    def _memory_write(self, params: dict) -> Any:
        self._require_layers()
        result = self.memory_stack.write(
            params["agent"],
            params["layer"],
            params["content"],
            topic=params.get("topic"),
            metadata=params.get("metadata"),
        )
        self._invalidate_metadata_cache()
        return result

    def _memory_read(self, params: dict) -> list:
        self._require_layers()
        return self.memory_stack.read(
            params["agent"],
            layer=params.get("layer"),
            last_n=params.get("last_n", 10),
        )

    def _memory_summarize(self, params: dict) -> dict:
        self._require_layers()
        return self.memory_stack.summarize(params["agent"])

    def _mine_file(self, params: dict) -> dict:
        self._require_miner()
        count = miner.mine_file_into_dimension(
            self.dim, params["filepath"], params["realm"], params["domain"],
        )
        self._invalidate_metadata_cache()
        return {"items_mined": count}

    def _mine_text(self, params: dict) -> Any:
        self._require_miner()
        result = miner.mine_text_into_dimension(
            self.dim,
            params["text"],
            params["realm"],
            params["domain"],
            source=params.get("source"),
            chunk=params.get("chunk", True),
        )
        self._invalidate_metadata_cache()
        return result

    def _batch_mine(self, params: dict) -> dict:
        self._require_miner()
        count = miner.batch_mine(
            self.dim,
            params["directory"],
            realm=params.get("realm"),
            pattern=params.get("pattern"),
        )
        self._invalidate_metadata_cache()
        return {"items_mined": count}

    def _aaak_compress(self, params: dict) -> str:
        return aaak_compress(params["text"], max_len=params.get("max_len", 500))

    def _aaak_decompress(self, params: dict) -> str:
        return aaak_decompress(params["text"])

    def _aaak_parse(self, params: dict) -> dict:
        return aaak_parse_entry(params["text"])

    def _rebuild_fts(self, params: dict) -> dict:
        _wal_log(str(self.dim._base), "rebuild_fts", {})
        self.dim.rebuild_fts()
        self._invalidate_metadata_cache()
        return {"rebuilt": True}

    def _kg_timeline(self, params: dict) -> list:
        return self.dim.kg.timeline(entity=params.get("entity"))

    def _create_tunnel(self, params: dict) -> dict:
        tunnel = dim_graph.create_tunnel(
            source_realm=params.get("source_realm", ""),
            source_domain=params.get("source_domain", ""),
            target_realm=params.get("target_realm", ""),
            target_domain=params.get("target_domain", ""),
            label=params.get("label", ""),
            dimension=self.dim,
            source_entity_id=params.get("source_entity_id"),
            target_entity_id=params.get("target_entity_id"),
        )
        return tunnel

    def _delete_tunnel(self, params: dict) -> dict:
        return dim_graph.delete_tunnel(params["tunnel_id"], dimension=self.dim)

    def _list_tunnels(self, params: dict) -> list:
        return dim_graph.list_tunnels(realm=params.get("realm"), dimension=self.dim)

    def _traverse(self, params: dict) -> list:
        return dim_graph.traverse(
            start_domain=params.get("start_domain", ""),
            dimension=self.dim,
            max_hops=params.get("max_hops", 2),
        )

    def _find_tunnels(self, params: dict) -> list:
        return dim_graph.find_tunnels(
            realm_a=params.get("realm_a"),
            realm_b=params.get("realm_b"),
            dimension=self.dim,
        )

    def _follow_tunnels(self, params: dict) -> list:
        return dim_graph.follow_tunnels(
            params["realm"],
            params["domain"],
            dimension=self.dim,
        )

    def _graph_stats(self, params: dict) -> dict:
        return dim_graph.graph_stats(dimension=self.dim)

    def _get_taxonomy(self, params: dict) -> dict:
        return self.dim.get_taxonomy()

    def _get_aaak_spec(self, params: dict) -> str:
        return AAAK_SPEC

    def _memories_filed_away(self, params: dict) -> dict:
        return self.dim.memories_filed_away()

    def _sync(self, params: dict) -> dict:
        _wal_log(str(self.dim._base), "sync", {
            "project_dir": params.get("project_dir"),
            "realm": params.get("realm"),
            "apply": params.get("apply", False),
        })
        project_dirs = [params["project_dir"]] if params.get("project_dir") else None
        report = sync_dimension(
            dimension_path=str(self.dim._base),
            project_dirs=project_dirs,
            realm=params.get("realm"),
            dry_run=not params.get("apply", False),
        )
        self._invalidate_metadata_cache()
        return dict(report)

    def _reconnect(self, params: dict) -> dict:
        ok = self.dim.reconnect()
        return {"reconnected": ok}

    def _hook_settings(self, params: dict) -> dict:
        config = AltMemoryConfig()
        if params:
            return config.set_hook_settings(
                silent_save=params.get("silent_save"),
                desktop_toast=params.get("desktop_toast"),
            )
        return config.get_hook_settings()

    def _get_backend(self, params: dict) -> dict:
        return {"backend": getattr(self.dim, "_backend", "faiss")}

    def _set_backend(self, params: dict) -> dict:
        return self.dim.set_backend(
            backend=params["backend"],
            reindex=params.get("reindex", True),
        )

    def _set_embedder(self, params: dict) -> dict:
        return self.dim.set_embedder(
            model=params["model"],
            device=params.get("device"),
            reindex=params.get("reindex", True),
        )

    def _get_default_embedder(self, params: dict) -> dict:
        from alt_memory.config import AltMemoryConfig
        return {"default_embedder": AltMemoryConfig().default_embedder}

    def _set_default_embedder(self, params: dict) -> dict:
        from alt_memory.config import AltMemoryConfig
        model = AltMemoryConfig().set_default_embedder(params["model"])
        return {"default_embedder": model}

    def _list_agents(self, params: dict) -> list:
        """Discover agent record writers by scanning agent_* realms."""
        agents = []
        for realm in self.dim.list_realms():
            name = realm.get("name", "")
            if name.startswith("agent_"):
                agents.append(name[len("agent_"):])
        return sorted(agents)

    def _get_people_map(self, params: dict) -> dict:
        from alt_memory.config import AltMemoryConfig
        return AltMemoryConfig().people_map

    def _set_people_map(self, params: dict) -> dict:
        from alt_memory.config import AltMemoryConfig
        AltMemoryConfig().save_people_map(params.get("map", {}))
        return {"saved": True}

    def _batch_add_entities(self, params: dict) -> dict:
        entities = params["entities"]
        batch: list[tuple[str, str, str, dict, str, Optional[str]]] = []
        for ent in entities:
            realm = ent["realm"]
            domain = ent["domain"]
            content = ent["content"]
            meta = ent.get("metadata") or {}
            source_file = ent.get("source_file") or ""
            entity_id = ent.get("entity_id")
            batch.append((realm, domain, content, meta, source_file, entity_id))
        ids = self.dim.batch_add_entities(batch)
        self._invalidate_metadata_cache()
        return {"entity_ids": ids, "count": len(ids)}

    def _delete_entities(self, params: dict) -> dict:
        entity_ids = params["entity_ids"]
        count = self.dim.delete_entities(entity_ids)
        self._invalidate_metadata_cache()
        return {"deleted": count}

    def _get_persona(self, params: dict) -> dict:
        return {"persona": self.dim.get_persona()}

    def _set_persona(self, params: dict) -> dict:
        result = self.dim.set_persona(params["name"])
        return result

    def _switch_persona(self, params: dict) -> dict:
        return self.dim.switch_persona(params["name"])

    def _import_entities(self, params: dict) -> dict:
        count = self.dim.import_entities(
            params["entities"],
            overwrite=params.get("overwrite", False),
        )
        self._invalidate_metadata_cache()
        return {"imported": count}

    def _export_collection(self, params: dict) -> list:
        return self.dim.export_collection(
            realm=params.get("realm"),
            domain=params.get("domain"),
        )

    # -- Helpers ----------------------------------------------------------------

    def _require_layers(self):
        if not HAS_LAYERS or self.memory_stack is None:
            raise RuntimeError("MemoryStack not available (alt_memory.layers not installed)")

    def _require_miner(self):
        if not HAS_MINER:
            raise RuntimeError("Miner not available (alt_memory.miner not installed)")


# -- Transports ----------------------------------------------------------------


def _stdio_server(server: MCPServer) -> None:
    """Read JSON-RPC lines from stdin, write response lines to stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        response = server.handle_request(line)
        if response is not None:
            sys.stdout.write(response + "\n")
            sys.stdout.flush()


if HAS_HTTP:

    class _MCPHTTPHandler(BaseHTTPRequestHandler):
        server_instance: MCPServer = None  # type: ignore[assignment]

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._send_json(200, {"status": "ok", "version": VERSION})
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/mcp":
                self.send_response(404)
                self.end_headers()
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"

            response = self.server_instance.handle_request(body)
            if response is None:
                response = json.dumps({"jsonrpc": "2.0", "id": None, "result": None})
            self._send_json(200, response)

        def _send_json(self, status: int, data):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if isinstance(data, str):
                self.wfile.write(data.encode("utf-8"))
            else:
                self.wfile.write(json.dumps(data).encode("utf-8"))

        def log_message(self, fmt, *args):
            logger.debug("HTTP: " + fmt, *args)

    def _sse_server(server: MCPServer, host: str, port: int) -> None:
        _MCPHTTPHandler.server_instance = server
        httpd = HTTPServer((host, port), _MCPHTTPHandler)
        logger.info("MCP SSE server listening on http://%s:%d", host, port)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            httpd.shutdown()


def run_server(dim: Dimension, host: str = "127.0.0.1", port: int = 8316,
               transport: str = "stdio") -> None:
    """Run the MCP server.

    Parameters
    ----------
    dim : Dimension
        Initialized Dimension instance.
    host : str
        Bind address for SSE transport (default ``127.0.0.1``).
    port : int
        Port for SSE transport (default ``8316``).
    transport : str
        ``"stdio"`` (read/write JSON-RPC on stdin/stdout) or
        ``"sse"`` (HTTP server with ``GET /health`` and ``POST /mcp``).
    """
    _maybe_eager_warmup_embedder(dim)
    _start_idle_exit_watchdog()

    server = MCPServer(dim)

    if transport == "stdio":
        _stdio_server(server)
    elif transport == "sse":
        if not HAS_HTTP:
            raise RuntimeError("http.server not available on this platform")
        _sse_server(server, host, port)
    else:
        raise ValueError(f"Unknown transport: {transport!r}")
