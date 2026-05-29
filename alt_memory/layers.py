"""Agent record memory layers (L0-L3) stored as Dimension entities."""

import logging
from typing import Optional

from alt_memory.dimension import Dimension, _sanitize

logger = logging.getLogger(__name__)


class MemoryStack:
    """Agent record memory with hierarchical layers L0-L3.

    Each layer is stored as a domain within a per-agent realm:
      realm: agent_{sanitized_name}
      domains: 'layer_0_identity', 'layer_1_essential', 'layer_2_working', 'layer_3_archive'
    """

    LAYER_NAMES = {
        0: "layer_0_identity",
        1: "layer_1_essential",
        2: "layer_2_working",
        3: "layer_3_archive",
    }

    def __init__(self, dimension: Dimension):
        self._dimension = dimension

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

        l3_entries = self._dimension.list_entities(
            realm=realm, domain="layer_3_archive", limit=1000,
        )
        l1_entries = self._dimension.list_entities(
            realm=realm, domain="layer_1_essential", limit=1000,
        )

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
        result = {}
        for layer in sorted(self.LAYER_NAMES):
            result[layer] = self.read(agent_name, layer, last_n=last_n)
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
