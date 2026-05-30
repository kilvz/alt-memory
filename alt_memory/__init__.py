"""Alt Memory v4.5.0 — FAISS-powered memory dimension with hybrid search, entity graph, MCP server, and 56 MCP tools."""

__version__ = "4.5.0"

from alt_memory.dimension import Dimension, SearchResult, mine_lock, MineAlreadyRunning
from alt_memory.backends.embedder import (
    NumpyEmbedder, SentenceTransformerEmbedder, SpacyGloveEmbedder, get_embedder,
)
from alt_memory.backends.faiss_store import FaissStore
from alt_memory.backends.chroma_store import ChromaStore
from alt_memory.backends.knowledge_graph import KnowledgeGraph

from alt_memory.dialect import (
    aaak_compress,
    aaak_decompress,
    aaak_parse_entry,
    aaak_validate,
    aaak_detect_emotions,
    aaak_detect_flags,
    aaak_extract_topics,
    aaak_extract_key_sentence,
    aaak_count_tokens,
    aaak_compression_stats,
    AaakDialect,
)
from alt_memory.layers import MemoryStack
from alt_memory.entity_detector import EntityDetector
from alt_memory.migrate import migrate, migrate_schema, rebuild_faiss, get_dimension_version, set_dimension_version
from alt_memory.entity_registry import EntityRegistry
from alt_memory.miner import (
    mine_file_into_dimension,
    mine_text_into_dimension,
    mine_conversation,
    batch_mine,
    mine_code_file,
    FileMiner,
)
from alt_memory.config import AltMemoryConfig, sanitize_name, sanitize_content
from alt_memory.query_sanitizer import sanitize_query
from alt_memory.normalize import normalize, strip_noise
from alt_memory.exporter import export_dimension
from alt_memory.sync import sync_dimension, SyncReport
from alt_memory.dedup import dedup_dimension, show_stats
from alt_memory.dim_graph import (
    build_graph,
    traverse,
    find_tunnels,
    create_tunnel,
    list_tunnels,
    delete_tunnel,
    follow_tunnels,
    graph_stats,
)
from alt_memory.gateways import compute_gateways_for_realm
from alt_memory.dynamics import (
    initialize_dynamics_fields, potentiate, apply_decay,
)
from alt_memory.general_extractor import extract_memories
from alt_memory.domain_detector_local import detect_domains_local
from alt_memory.fact_checker import check_text
from alt_memory.onboarding import run_onboarding, quick_setup
from alt_memory.project_scanner import (
    ProjectInfo, PersonInfo, scan, find_git_repos,
    to_detected_dict, discover_entities, _dedupe_people,
)
from alt_memory.convo_scanner import scan_claude_projects, is_claude_projects_root

__all__ = [
    "Dimension",
    "SearchResult",
    "mine_lock",
    "MineAlreadyRunning",
    "NumpyEmbedder",
    "SentenceTransformerEmbedder",
    "SpacyGloveEmbedder",
    "get_embedder",
    "FaissStore",
    "ChromaStore",
    "KnowledgeGraph",
    "MemoryStack",
    "EntityDetector",
    "EntityRegistry",
    "migrate",
    "migrate_schema",
    "rebuild_faiss",
    "get_dimension_version",
    "set_dimension_version",
    "aaak_compress",
    "aaak_decompress",
    "aaak_parse_entry",
    "aaak_validate",
    "aaak_detect_emotions",
    "aaak_detect_flags",
    "aaak_extract_topics",
    "aaak_extract_key_sentence",
    "aaak_count_tokens",
    "aaak_compression_stats",
    "AaakDialect",
    "mine_file_into_dimension",
    "mine_text_into_dimension",
    "mine_conversation",
    "batch_mine",
    "mine_code_file",
    "FileMiner",
    "AltMemoryConfig",
    "sanitize_name",
    "sanitize_content",
    "sanitize_query",
    "normalize",
    "strip_noise",
    "export_dimension",
    "sync_dimension",
    "SyncReport",
    "dedup_dimension",
    "show_stats",
    "build_graph",
    "traverse",
    "find_tunnels",
    "create_tunnel",
    "list_tunnels",
    "delete_tunnel",
    "follow_tunnels",
    "graph_stats",
    "compute_gateways_for_realm",
    "initialize_dynamics_fields",
    "potentiate",
    "apply_decay",
    "extract_memories",
    "detect_domains_local",
    "check_text",
    "run_onboarding",
    "quick_setup",
    "ProjectInfo",
    "PersonInfo",
    "scan",
    "find_git_repos",
    "to_detected_dict",
    "discover_entities",
    "_dedupe_people",
    "scan_claude_projects",
    "is_claude_projects_root",
]
