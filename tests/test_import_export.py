"""Tests for import/export round-trip."""



class TestExport:
    def test_export_empty(self, fresh_dim):
        data = fresh_dim.export_collection()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_export_with_data(self, fresh_dim):
        fresh_dim.add_entity("test", "demo", "export me", metadata={"k": "v"})
        data = fresh_dim.export_collection()
        assert len(data) == 1
        assert data[0]["content"] == "export me"
        assert data[0]["realm"] == "test"
        assert data[0]["domain"] == "demo"
        assert data[0]["metadata"]["k"] == "v"

    def test_export_filtered(self, fresh_dim):
        fresh_dim.add_entity("test", "demo", "included")
        fresh_dim.add_entity("test", "other", "excluded")
        data = fresh_dim.export_collection(domain="demo")
        assert len(data) == 1
        assert data[0]["content"] == "included"

    def test_export_includes_ids(self, fresh_dim):
        eid = fresh_dim.add_entity("test", "demo", "id check")
        data = fresh_dim.export_collection()
        assert data[0]["id"] == eid


class TestImport:
    def test_import_round_trip(self, fresh_dim):
        fresh_dim.add_entity("src", "a", "alpha")
        fresh_dim.add_entity("src", "b", "beta")
        exported = fresh_dim.export_collection()

        import shutil
        import tempfile

        from alt_memory.dimension import Dimension
        tmp = tempfile.mkdtemp(prefix="_test_import_")
        d2 = Dimension(path=tmp, backend="faiss")
        d2.init()
        try:
            count = d2.import_entities(exported)
            assert count == 2
            for e in exported:
                entity = d2.get_entity(e["id"])
                assert entity is not None
                assert entity["content"] == e["content"]
        finally:
            d2.close()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_import_empty_list(self, fresh_dim):
        count = fresh_dim.import_entities([])
        assert count == 0

    def test_import_skip_existing(self, fresh_dim):
        eid = fresh_dim.add_entity("test", "demo", "original")
        data = [{"realm": "test", "domain": "demo", "content": "new",
                 "entity_id": eid}]
        count = fresh_dim.import_entities(data, overwrite=False)
        assert count == 0  # skipped because ID exists
        entity = fresh_dim.get_entity(eid)
        assert entity["content"] == "original"  # unchanged

    def test_import_overwrite(self, fresh_dim):
        eid = fresh_dim.add_entity("test", "demo", "original")
        data = [{"realm": "test", "domain": "demo", "content": "overwritten",
                 "entity_id": eid}]
        count = fresh_dim.import_entities(data, overwrite=True)
        assert count == 1
        entity = fresh_dim.get_entity(eid)
        assert entity["content"] == "overwritten"

    def test_import_with_metadata(self, fresh_dim):
        data = [{"realm": "test", "domain": "demo", "content": "meta test",
                 "metadata": {"priority": "high", "tags": ["a", "b"]}}]
        count = fresh_dim.import_entities(data)
        assert count == 1
        entity = fresh_dim.get_entity(data[0].get("entity_id") or (count and list(
            fresh_dim.list_entities()
        )[0]["id"]))
        # Just verify the entity exists
        assert entity is not None


class TestImportViaMCP:
    def test_import_export_via_mcp(self, fresh_dim):
        from alt_memory.mcp_server import MCPServer
        server = MCPServer(fresh_dim)

        # Add data
        fresh_dim.add_entity("mcp", "test", "mcp entity")

        # Export
        exported = server._export_collection({})
        assert len(exported) >= 1

        # Import into same dimension (with overwrite for same IDs)
        result = server._import_entities({"entities": exported, "overwrite": True})
        assert result["imported"] >= 1
