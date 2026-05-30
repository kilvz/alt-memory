"""Agent record memory layers (L0-L3) stored as Dimension entities,
plus dimension-scoped wake-up context (L0 identity + L1 essential story)."""

import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from alt_memory.dimension import Dimension, _sanitize

logger = logging.getLogger(__name__)


class MemoryStack:
    """Agent record memory with hierarchical layers L0-L3, plus dimension-scoped wake-up context.

    Agent-record layers (stored as entities in per-agent realms):
      realm: agent_{sanitized_name}
      domains: 'layer_0_identity', 'layer_1_essential', 'layer_2_working', 'layer_3_archive'

    Dimension wake-up context (generated on-the-fly from dimension data):
      L0: ~/.alt-memory/identity.txt (user-authored identity)
      L1: Top entities from the dimension, scored by importance/weight
      L2: On-demand realm/domain filtered retrieval
      L3: Full semantic search
    """

    LAYER_NAMES = {
        0: "layer_0_identity",
        1: "layer_1_essential",
        2: "layer_2_working",
        3: "layer_3_archive",
    }

    MAX_ENTITIES = 15
    MAX_CHARS = 3200
    MAX_SCAN = 2000

    def __init__(self, dimension: Dimension, identity_path: Optional[str] = None):
        self._dimension = dimension
        self._identity_path = identity_path or os.path.expanduser("~/.alt-memory/identity.txt")
        self._identity_cache: Optional[str] = None

    def _validate_layer(self, layer: int) -> None:
        if layer not in self.LAYER_NAMES:
            raise ValueError(f"Invalid layer {layer!r}. Must be 0-3.")

    def _realm_for(self, agent_name: str) -> str:
        return f"agent_{_sanitize(agent_name)}"

    def write(self, agent_name: str, layer: int, content: str,
              topic: str = "general", metadata: dict = None) -> str:
        """Write content to a memory layer.

        Args:
            agent_name: Name of the agent
            layer: 0-3 (Identity, Essential, Working, Archive)
            content: Text content to store
            topic: Topic label for the entry
            metadata: Additional metadata dict

        Returns: entity ID

        Raises ValueError if layer not in 0-3
        """
        self._validate_layer(layer)
        realm = self._realm_for(agent_name)
        domain = self.LAYER_NAMES[layer]
        meta = dict(metadata or {})
        meta.update({
            "agent": agent_name,
            "topic": topic,
            "type": "layer_entry",
            "layer": layer,
        })
        return self._dimension.add_entity(
            realm=realm, domain=domain, content=content, metadata=meta,
        )

    def read(self, agent_name: str, layer: int, last_n: int = 10) -> list[dict]:
        """Read recent entries from a memory layer.

        Returns list of entity dicts (id, content, metadata, created_at, topic).

        Raises ValueError if layer not in 0-3
        """
        self._validate_layer(layer)
        realm = self._realm_for(agent_name)
        domain = self.LAYER_NAMES[layer]
        entries = self._dimension.list_entities(realm=realm, domain=domain, limit=last_n)
        result = []
        for e in entries:
            meta = e.get("metadata", {})
            result.append({
                "id": e.get("id", ""),
                "content": e.get("content", ""),
                "metadata": meta,
                "created_at": e.get("created_at", ""),
                "topic": meta.get("topic", "general"),
            })
        return result

    def summarize(self, agent_name: str) -> dict:
        """Compress higher layers into lower layers:
        - Summarize L3 (archive) entries into a new L1 (essential) entry
        - Summarize L1 entries into L0 (identity) update
        - Uses naive text summarization: concatenate + truncate

        Returns dict with keys: {l0_updated, l1_count, l3_count}
        """
        realm = self._realm_for(agent_name)

        all_entries = self._dimension.list_entities(realm=realm, limit=1000)
        l3_entries = [e for e in all_entries if e.get("metadata", {}).get("domain") == "layer_3_archive"]
        l1_entries = [e for e in all_entries if e.get("metadata", {}).get("domain") == "layer_1_essential"]

        l3_summary = ""
        if l3_entries:
            snippets = [
                e.get("content", "")[:200] for e in l3_entries if e.get("content", "").strip()
            ]
            l3_summary = "; ".join(snippets)
            if len(l3_summary) > 100000:
                l3_summary = l3_summary[:100000] + "..."

        l1_count = 0
        if l3_summary:
            self._dimension.add_entity(
                realm=realm, domain="layer_1_essential",
                content=l3_summary,
                metadata={
                    "agent": agent_name,
                    "topic": "summarized_archive",
                    "type": "layer_entry",
                    "layer": 1,
                    "source": "summarize_l3",
                },
            )
            l1_count = 1
            l1_entries = self._dimension.list_entities(
                realm=realm, domain="layer_1_essential", limit=1000,
            )

        l0_updated = False
        if l1_entries:
            key_facts = "; ".join(
                e.get("content", "")[:300] for e in l1_entries if e.get("content", "").strip()
            )
            if key_facts:
                self._dimension.add_entity(
                    realm=realm, domain="layer_0_identity",
                    content=key_facts,
                    metadata={
                        "agent": agent_name,
                        "topic": "identity_summary",
                        "type": "layer_entry",
                        "layer": 0,
                        "source": "summarize_l1",
                    },
                )
                l0_updated = True

        return {
            "l0_updated": l0_updated,
            "l1_count": l1_count,
            "l3_count": len(l3_entries),
        }

    def read_all(self, agent_name: str, last_n: int = 5) -> dict[int, list[dict]]:
        """Read recent entries from ALL layers at once.

        Returns dict mapping layer_number -> list of entries
        """
        realm = self._realm_for(agent_name)
        domain_names = set(self.LAYER_NAMES.values())
        all_in_realm = self._dimension.list_entities(realm=realm, limit=1000)
        by_domain: dict[str, list[dict]] = {}
        for e in all_in_realm:
            d = e.get("metadata", {}).get("domain", "")
            if d in domain_names:
                by_domain.setdefault(d, []).append(e)
        result = {}
        for layer in sorted(self.LAYER_NAMES):
            domain = self.LAYER_NAMES[layer]
            entries = by_domain.get(domain, [])
            entries.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            out = []
            for e in entries[:last_n]:
                meta = e.get("metadata", {})
                out.append({
                    "id": e.get("id", ""),
                    "content": e.get("content", ""),
                    "metadata": meta,
                    "created_at": e.get("created_at", ""),
                    "topic": meta.get("topic", "general"),
                })
            result[layer] = out
        return result

    def clear_layer(self, agent_name: str, layer: int) -> bool:
        """Delete all entries in a layer.

        Raises ValueError if layer not in 0-3
        """
        self._validate_layer(layer)
        realm = self._realm_for(agent_name)
        domain = self.LAYER_NAMES[layer]
        for _ in range(100):
            entities = self._dimension.list_entities(realm=realm, domain=domain, limit=1000)
            if not entities:
                return True
            for e in entities:
                eid = e.get("id")
                if eid:
                    self._dimension.delete_entity(eid)
        return True

    def get_latest_identity(self, agent_name: str) -> Optional[dict]:
        """Get the most recent L0 identity entry for an agent.

        Returns None if no identity entry exists.
        """
        entries = self.read(agent_name, 0, last_n=1)
        return entries[0] if entries else None

    def update_identity(self, agent_name: str, facts: dict) -> str:
        """Write a structured identity update (L0).

        facts example: {"role": "developer", "project": "alt-memory", "goal": "port features"}
        Converts dict to formatted text before storing.

        Returns entity ID.
        """
        lines = [f"{k}: {v}" for k, v in facts.items()]
        text = "\n".join(lines)
        return self.write(
            agent_name, 0, content=text, topic="identity_update",
            metadata={"facts": facts},
        )

    # ------------------------------------------------------------------
    # Dimension-scoped wake-up context (L0-L3)
    # ------------------------------------------------------------------

    def _load_l0_identity(self) -> str:
        """Load identity from ~/.alt-memory/identity.txt or return a sensible default."""
        if self._identity_cache is not None:
            return self._identity_cache
        path = self._identity_path
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self._identity_cache = f.read().strip()
                return self._identity_cache
        return "## L0 \u2014 IDENTITY\nNo identity configured. Create ~/.alt-memory/identity.txt"

    def _generate_l1_essential(self, realm: Optional[str] = None) -> str:
        """Auto-generate essential story from top-scored dimension entities."""
        dim = self._dimension
        entities = dim.list_entities(realm=realm, limit=self.MAX_SCAN)

        if not entities:
            return "## L1 \u2014 No memories yet."

        scored = []
        for e in entities:
            meta = e.get("metadata", {}) or {}
            doc = e.get("content", "") or ""
            importance = 3
            for key in ("importance", "weight", "priority", "emotional_weight"):
                val = meta.get(key)
                if val is not None:
                    try:
                        importance = float(val)
                    except (ValueError, TypeError):
                        pass
                    break
            scored.append((importance, meta, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:self.MAX_ENTITIES]

        by_domain = defaultdict(list)
        for imp, meta, doc in top:
            domain = meta.get("domain", "general")
            by_domain[domain].append((imp, meta, doc))

        lines = ["## L1 \u2014 ESSENTIAL STORY"]
        total_len = 0
        for domain, entries in sorted(by_domain.items()):
            domain_line = f"\n[{domain}]"
            lines.append(domain_line)
            total_len += len(domain_line)

            for _imp, meta, doc in entries:
                src = Path(meta.get("source_file", "")).name if meta.get("source_file") else ""
                snippet = doc.strip().replace("\n", " ")
                if len(snippet) > 200:
                    snippet = snippet[:197] + "..."
                entry_line = f"  - {snippet}"
                if src:
                    entry_line += f"  ({src})"
                if total_len + len(entry_line) > self.MAX_CHARS:
                    lines.append("  ... (more in L3 search)")
                    return "\n".join(lines)
                lines.append(entry_line)
                total_len += len(entry_line)

        return "\n".join(lines)

    def wake_up(self, realm: Optional[str] = None) -> str:
        """Generate wake-up context: L0 (identity) + L1 (essential story).

        Typically ~600-900 tokens. Suitable for system prompt injection.

        Args:
            realm: Optional realm filter for L1 (project-specific wake-up).
        """
        parts = [self._load_l0_identity(), "", self._generate_l1_essential(realm=realm)]
        return "\n".join(parts)

    def recall(self, realm: Optional[str] = None, domain: Optional[str] = None,
               limit: int = 10) -> str:
        """On-demand L2 retrieval filtered by realm/domain.

        Returns formatted text with recent entities.
        """
        entities = self._dimension.list_entities(realm=realm, domain=domain, limit=limit)
        if not entities:
            label = f"realm={realm}" if realm else ""
            if domain:
                label += f" domain={domain}" if label else f"domain={domain}"
            return f"No entities found for {label}."

        lines = [f"## L2 \u2014 ON-DEMAND ({len(entities)} entities)"]
        for e in entities:
            meta = e.get("metadata", {}) or {}
            doc = e.get("content", "") or ""
            domain_name = meta.get("domain", "?")
            src = Path(meta.get("source_file", "")).name if meta.get("source_file") else ""
            snippet = doc.strip().replace("\n", " ")
            if len(snippet) > 300:
                snippet = snippet[:297] + "..."
            entry = f"  [{domain_name}] {snippet}"
            if src:
                entry += f"  ({src})"
            lines.append(entry)

        return "\n".join(lines)

    def deep_search(self, query: str, realm: Optional[str] = None,
                    domain: Optional[str] = None, n_results: int = 5) -> str:
        """Deep L3 semantic search.

        Returns formatted text with search results and similarity scores.
        """
        results = self._dimension.search(
            query=query, realm=realm, domain=domain,
            n_results=n_results, mode="hybrid",
        )
        if not results:
            return f'No results found for "{query}".'

        lines = [f'## L3 \u2014 SEARCH RESULTS for "{query}"']
        for i, r in enumerate(results, 1):
            meta = r.metadata or {}
            doc = r.text or ""
            similarity = round(max(0.0, 1.0 - r.distance), 3)
            realm_name = meta.get("realm", "?")
            domain_name = meta.get("domain", "?")
            src = Path(meta.get("source_file", "")).name if meta.get("source_file") else ""

            snippet = doc.strip().replace("\n", " ")
            if len(snippet) > 300:
                snippet = snippet[:297] + "..."

            lines.append(f"  [{i}] {realm_name}/{domain_name} (sim={similarity})")
            lines.append(f"      {snippet}")
            if src:
                lines.append(f"      src: {src}")

        return "\n".join(lines)

    def deep_search_raw(self, query: str, realm: Optional[str] = None,
                        domain: Optional[str] = None, n_results: int = 5) -> list[dict]:
        """Deep L3 semantic search returning raw dicts instead of formatted text."""
        results = self._dimension.search(
            query=query, realm=realm, domain=domain,
            n_results=n_results, mode="hybrid",
        )
        hits = []
        for r in results:
            meta = r.metadata or {}
            hits.append({
                "text": r.text,
                "realm": meta.get("realm", "unknown"),
                "domain": meta.get("domain", "unknown"),
                "source_file": Path(meta.get("source_file", "?")).name,
                "similarity": round(max(0.0, 1.0 - r.distance), 3),
                "metadata": meta,
            })
        return hits

    def layer_status(self) -> dict:
        """Status of all layers — identity file, entity counts, search availability."""
        dim = self._dimension
        result = {
            "dimension_path": str(dim._base) if hasattr(dim, "_base") else "?",
            "L0_identity": {
                "path": self._identity_path,
                "exists": os.path.exists(self._identity_path),
            },
            "L1_essential": {
                "description": "Auto-generated from top dimension entities",
                "max_entities": self.MAX_ENTITIES,
            },
            "L2_on_demand": {
                "description": "Realm/domain filtered retrieval",
            },
            "L3_deep_search": {
                "description": "Full semantic search via FAISS + FTS5",
            },
        }
        try:
            s = dim.status()
            result["total_entities"] = s.get("entities", 0)
        except Exception:
            result["total_entities"] = 0
        return result
