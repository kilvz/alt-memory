"""Tests for persona: character definition model (Eternal AI-style)."""

import pytest


class TestPersona:
    def test_default_empty(self, fresh_dim):
        assert fresh_dim.get_persona() == {"name": ""}

    def test_create_persona(self, fresh_dim):
        result = fresh_dim.create_persona(
            name="coder",
            system_prompt="You are an expert Python developer.",
            description="Coding expert",
            model="gpt-4",
            framework="opencode",
            metadata={"lang": "python"},
        )
        assert result["name"] == "coder"
        assert result["system_prompt"] == "You are an expert Python developer."
        assert result["description"] == "Coding expert"
        assert result["model"] == "gpt-4"
        assert result["framework"] == "opencode"
        assert result["metadata"] == {"lang": "python"}

        # Should not be active yet
        assert fresh_dim.get_persona()["name"] == ""

    def test_create_duplicate_raises(self, fresh_dim):
        fresh_dim.create_persona(name="coder", system_prompt="test")
        with pytest.raises(ValueError, match="already exists"):
            fresh_dim.create_persona(name="coder", system_prompt="test")

    def test_set_persona(self, fresh_dim):
        result = fresh_dim.set_persona(
            "coder",
            system_prompt="You are an expert Python developer.",
        )
        assert result["name"] == "coder"
        assert result["system_prompt"] == "You are an expert Python developer."
        assert fresh_dim.get_persona()["name"] == "coder"

    def test_set_persona_preserves_existing(self, fresh_dim):
        fresh_dim.set_persona(
            "coder",
            system_prompt="You are an expert Python developer.",
            description="Coding expert",
            model="gpt-4",
        )
        fresh_dim.set_persona("coder", model="claude-3")
        p = fresh_dim.get_persona()
        assert p["name"] == "coder"
        assert p["system_prompt"] == "You are an expert Python developer."
        assert p["description"] == "Coding expert"
        assert p["model"] == "claude-3"

    def test_switch_persona(self, fresh_dim):
        fresh_dim.set_persona("coder", system_prompt="Coder prompt")
        fresh_dim.switch_persona("architect", system_prompt="Architect prompt")
        assert fresh_dim.get_persona()["name"] == "architect"

    def test_set_empty_raises(self, fresh_dim):
        with pytest.raises(ValueError):
            fresh_dim.set_persona("")

    def test_persona_creates_realm(self, fresh_dim):
        fresh_dim.set_persona("tester")
        realms = fresh_dim.list_realms()
        assert any(r["name"] == "persona_tester" for r in realms)

    def test_list_personas(self, fresh_dim):
        fresh_dim.create_persona("alice", system_prompt="Alice prompt", description="Alice")
        fresh_dim.create_persona("bob", system_prompt="Bob prompt", description="Bob")
        fresh_dim.set_persona("alice")
        personas = fresh_dim.list_personas()
        assert len(personas) == 2
        by_name = {p["name"]: p for p in personas}
        assert by_name["alice"]["active"] is True
        assert by_name["bob"]["active"] is False

    def test_delete_persona(self, fresh_dim):
        fresh_dim.create_persona("temp", system_prompt="Temp prompt")
        fresh_dim.set_persona("temp")
        assert fresh_dim.delete_persona("temp") is True
        assert fresh_dim.list_personas() == []
        assert fresh_dim.get_persona()["name"] == ""

    def test_get_persona_character(self, fresh_dim):
        fresh_dim.set_persona(
            "donald",
            system_prompt="Act as if you are Donald Trump.",
        )
        prompt = fresh_dim.get_persona_character()
        assert prompt == "Act as if you are Donald Trump."

        prompt2 = fresh_dim.get_persona_character("donald")
        assert prompt2 == "Act as if you are Donald Trump."

    def test_get_persona_character_not_found(self, fresh_dim):
        assert fresh_dim.get_persona_character() == ""
        assert fresh_dim.get_persona_character("nobody") == ""

    def test_persona_persists_across_reload(self, dim_path):
        from alt_memory.dimension import Dimension
        d1 = Dimension(path=dim_path, backend="faiss")
        d1.init()
        d1.set_persona("persistent", system_prompt="I persist across restarts.")
        d1.close()

        d2 = Dimension(path=dim_path, backend="faiss")
        d2.init()
        assert d2.get_persona()["name"] == "persistent"
        assert d2.get_persona()["system_prompt"] == "I persist across restarts."
        d2.close()

    def test_legacy_upgrade(self, dim_path):
        import json
        from pathlib import Path
        from alt_memory.dimension import Dimension

        # Write legacy format
        legacy = {"persona": "legacy_user"}
        Path(dim_path).joinpath("persona.json").write_text(json.dumps(legacy))

        d = Dimension(path=dim_path, backend="faiss")
        d.init()
        assert d.get_persona()["name"] == "legacy_user"
        assert d.get_persona()["system_prompt"] == ""

        # Verify file was upgraded
        data = json.loads(Path(dim_path).joinpath("persona.json").read_text())
        assert "active" in data
        assert "personas" in data


class TestPersonaMCP:
    def test_get_persona_via_mcp(self, fresh_dim):
        from alt_memory.mcp_server import MCPServer
        server = MCPServer(fresh_dim)
        result = server._get_persona({})
        assert "name" in result

    def test_get_persona_character_via_mcp(self, fresh_dim):
        from alt_memory.mcp_server import MCPServer
        server = MCPServer(fresh_dim)
        fresh_dim.set_persona("test", system_prompt="Test prompt")
        result = server._get_persona_character({})
        assert result["system_prompt"] == "Test prompt"

    def test_set_persona_via_mcp(self, fresh_dim):
        from alt_memory.mcp_server import MCPServer
        server = MCPServer(fresh_dim)
        result = server._set_persona({"name": "mcp_test", "system_prompt": "MCP prompt"})
        assert result["name"] == "mcp_test"
        assert result["system_prompt"] == "MCP prompt"
        assert fresh_dim.get_persona()["name"] == "mcp_test"

    def test_create_persona_via_mcp(self, fresh_dim):
        from alt_memory.mcp_server import MCPServer
        server = MCPServer(fresh_dim)
        result = server._create_persona({
            "name": "created",
            "system_prompt": "Created via MCP",
            "model": "gpt-4",
        })
        assert result["name"] == "created"
        assert result["system_prompt"] == "Created via MCP"

    def test_list_personas_via_mcp(self, fresh_dim):
        from alt_memory.mcp_server import MCPServer
        server = MCPServer(fresh_dim)
        fresh_dim.create_persona("p1", system_prompt="P1")
        fresh_dim.create_persona("p2", system_prompt="P2")
        result = server._list_personas({})
        assert len(result["personas"]) == 2

    def test_delete_persona_via_mcp(self, fresh_dim):
        from alt_memory.mcp_server import MCPServer
        server = MCPServer(fresh_dim)
        fresh_dim.create_persona("temp", system_prompt="Temp")
        result = server._delete_persona({"name": "temp"})
        assert result["deleted"] is True
