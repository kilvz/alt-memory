"""Alt Memory v4.5.0 — FAISS-powered memory dimension with hybrid search, entity graph, MCP server, and 56 MCP tools."""

__version__ = "4.5.0"

from alt_memory.backends.chroma_store import ChromaStore
from alt_memory.backends.embedder import (
    NumpyEmbedder,
    SentenceTransformerEmbedder,
    SpacyGloveEmbedder,
    get_embedder,
)
from alt_memory.backends.faiss_store import FaissStore
from alt_memory.backends.knowledge_graph import KnowledgeGraph
from alt_memory.config import AltMemoryConfig, sanitize_content, sanitize_name
from alt_memory.convo_scanner import is_claude_projects_root, scan_claude_projects
from alt_memory.dedup import dedup_dimension, show_stats
from alt_memory.dialect import (
    AaakDialect,
    aaak_compress,
    aaak_compression_stats,
    aaak_count_tokens,
    aaak_decompress,
    aaak_detect_emotions,
    aaak_detect_flags,
    aaak_extract_key_sentence,
    aaak_extract_topics,
    aaak_parse_entry,
    aaak_validate,
)
from alt_memory.dim_graph import (
    build_graph,
    create_tunnel,
    delete_tunnel,
    find_tunnels,
    follow_tunnels,
    graph_stats,
    list_tunnels,
    traverse,
)
from alt_memory.dimension import Dimension, MineAlreadyRunning, SearchResult, mine_lock
from alt_memory.domain_detector_local import detect_domains_local
from alt_memory.dynamics import (
    apply_decay,
    initialize_dynamics_fields,
    potentiate,
)
from alt_memory.entity_detector import EntityDetector
from alt_memory.entity_registry import EntityRegistry
from alt_memory.exporter import export_dimension
from alt_memory.fact_checker import check_text
from alt_memory.gateways import compute_gateways_for_realm
from alt_memory.general_extractor import extract_memories
from alt_memory.layers import MemoryStack
from alt_memory.migrate import (
    get_dimension_version,
    migrate,
    migrate_schema,
    rebuild_faiss,
    set_dimension_version,
)
from alt_memory.miner import (
    FileMiner,
    batch_mine,
    mine_code_file,
    mine_conversation,
    mine_file_into_dimension,
    mine_text_into_dimension,
)
from alt_memory.normalize import normalize, strip_noise
from alt_memory.onboarding import quick_setup, run_onboarding
from alt_memory.project_scanner import (
    PersonInfo,
    ProjectInfo,
    _dedupe_people,
    discover_entities,
    find_git_repos,
    scan,
    to_detected_dict,
)
from alt_memory.query_sanitizer import sanitize_query
from alt_memory.sync import SyncReport, sync_dimension

__all__ = [
    "AaakDialect",
    "AltMemoryConfig",
    "ChromaStore",
    "Dimension",
    "EntityDetector",
    "EntityRegistry",
    "FaissStore",
    "FileMiner",
    "KnowledgeGraph",
    "MemoryStack",
    "MineAlreadyRunning",
    "NumpyEmbedder",
    "PersonInfo",
    "ProjectInfo",
    "SearchResult",
    "SentenceTransformerEmbedder",
    "SpacyGloveEmbedder",
    "SyncReport",
    "_dedupe_people",
    "aaak_compress",
    "aaak_compression_stats",
    "aaak_count_tokens",
    "aaak_decompress",
    "aaak_detect_emotions",
    "aaak_detect_flags",
    "aaak_extract_key_sentence",
    "aaak_extract_topics",
    "aaak_parse_entry",
    "aaak_validate",
    "apply_decay",
    "batch_mine",
    "build_graph",
    "check_text",
    "compute_gateways_for_realm",
    "create_tunnel",
    "dedup_dimension",
    "delete_tunnel",
    "detect_domains_local",
    "discover_entities",
    "export_dimension",
    "extract_memories",
    "find_git_repos",
    "find_tunnels",
    "follow_tunnels",
    "get_dimension_version",
    "get_embedder",
    "graph_stats",
    "initialize_dynamics_fields",
    "is_claude_projects_root",
    "list_tunnels",
    "migrate",
    "migrate_schema",
    "mine_code_file",
    "mine_conversation",
    "mine_file_into_dimension",
    "mine_lock",
    "mine_text_into_dimension",
    "normalize",
    "potentiate",
    "quick_setup",
    "rebuild_faiss",
    "run_onboarding",
    "sanitize_content",
    "sanitize_name",
    "sanitize_query",
    "scan",
    "scan_claude_projects",
    "set_dimension_version",
    "show_stats",
    "strip_noise",
    "sync_dimension",
    "to_detected_dict",
    "traverse",
]
