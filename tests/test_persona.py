"""Tests for persona: get/set/switch."""

import pytest


class TestPersona:
    def test_default_empty(self, fresh_dim):
        assert fresh_dim.get_persona() == ""

    def test_set_persona(self, fresh_dim):
        result = fresh_dim.set_persona("coder")
        assert result["persona"] == "coder"
        assert fresh_dim.get_persona() == "coder"

    def test_switch_persona(self, fresh_dim):
        fresh_dim.set_persona("coder")
        fresh_dim.switch_persona("architect")
        assert fresh_dim.get_persona() == "architect"

    def test_set_empty_raises(self, fresh_dim):
        with pytest.raises(ValueError):
            fresh_dim.set_persona("")

    def test_persona_creates_realm(self, fresh_dim):
        fresh_dim.set_persona("tester")
        realms = fresh_dim.list_realms()
        assert any(r["name"] == "persona_tester" for r in realms)

    def test_persona_persists_across_reload(self, dim_path):
        from alt_memory.dimension import Dimension
        d1 = Dimension(path=dim_path, backend="faiss")
        d1.init()
        d1.set_persona("persistent")
        d1.close()

        d2 = Dimension(path=dim_path, backend="faiss")
        d2.init()
        assert d2.get_persona() == "persistent"
        d2.close()


class TestPersonaMCP:
    def test_get_persona_via_mcp(self, fresh_dim):
        from alt_memory.mcp_server import MCPServer
        server = MCPServer(fresh_dim)
        result = server._get_persona({})
        assert "persona" in result

    def test_set_persona_via_mcp(self, fresh_dim):
        from alt_memory.mcp_server import MCPServer
        server = MCPServer(fresh_dim)
        result = server._set_persona({"name": "mcp_test"})
        assert result["persona"] == "mcp_test"
        assert fresh_dim.get_persona() == "mcp_test"
