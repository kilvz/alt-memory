"""Tests for Dimension CRUD, batch operations, and search."""

import pytest


class TestRealmDomain:
    def test_create_realm(self, fresh_dim):
        name = fresh_dim.get_or_create_realm("test_realm", "A test realm")
        assert name == "test_realm"
        realms = fresh_dim.list_realms()
        assert any(r["name"] == "test_realm" for r in realms)

    def test_create_domain(self, fresh_dim):
        fresh_dim.get_or_create_realm("r1")
        name = fresh_dim.get_or_create_domain("r1", "d1", "A test domain")
        assert name == "d1"
        domains = fresh_dim.list_domains("r1")
        assert any(d["name"] == "d1" for d in domains)

    def test_delete_realm(self, fresh_dim):
        fresh_dim.get_or_create_realm("delete_me")
        assert fresh_dim.delete_realm("delete_me") is True


class TestEntityCRUD:
    def test_add_entity(self, fresh_dim):
        eid = fresh_dim.add_entity("test", "demo", "hello world")
        assert eid is not None
        assert eid.startswith("doc_")

    def test_get_entity(self, fresh_dim):
        eid = fresh_dim.add_entity("test", "demo", "find me")
        entity = fresh_dim.get_entity(eid)
        assert entity is not None
        assert entity["content"] == "find me"
        assert entity["realm"] == "test"
        assert entity["domain"] == "demo"

    def test_get_entity_not_found(self, fresh_dim):
        assert fresh_dim.get_entity("nonexistent") is None

    def test_update_entity(self, fresh_dim):
        eid = fresh_dim.add_entity("test", "demo", "original")
        updated = fresh_dim.update_entity(eid, content="updated")
        assert updated is True
        entity = fresh_dim.get_entity(eid)
        assert entity["content"] == "updated"

    def test_delete_entity(self, fresh_dim):
        eid = fresh_dim.add_entity("test", "demo", "delete me")
        assert fresh_dim.delete_entity(eid) is True
        assert fresh_dim.get_entity(eid) is None

    def test_delete_nonexistent(self, fresh_dim):
        assert fresh_dim.delete_entity("nonexistent") is False

    def test_list_entities(self, fresh_dim):
        fresh_dim.add_entity("test", "demo", "a")
        fresh_dim.add_entity("test", "demo", "b")
        fresh_dim.add_entity("test", "other", "c")
        entities = fresh_dim.list_entities()
        assert len(entities) == 3

    def test_list_entities_filtered(self, fresh_dim):
        fresh_dim.add_entity("test", "demo", "a")
        fresh_dim.add_entity("test", "other", "b")
        entities = fresh_dim.list_entities(domain="demo")
        assert len(entities) == 1

    def test_list_entities_pagination(self, fresh_dim):
        for i in range(5):
            fresh_dim.add_entity("test", "demo", f"entity {i}")
        page1 = fresh_dim.list_entities(limit=2, offset=0)
        assert len(page1) == 2
        page2 = fresh_dim.list_entities(limit=2, offset=2)
        assert len(page2) == 2


class TestBatchOperations:
    def test_batch_add(self, fresh_dim):
        batch = [
            ("test", "demo", "batch a", {}, "", None),
            ("test", "demo", "batch b", {"x": "y"}, "", None),
            ("test", "other", "batch c", {}, "file.txt", None),
        ]
        ids = fresh_dim.batch_add_entities(batch)
        assert len(ids) == 3
        for eid in ids:
            assert fresh_dim.get_entity(eid) is not None

    def test_delete_entities(self, fresh_dim):
        e1 = fresh_dim.add_entity("test", "demo", "delete me")
        e2 = fresh_dim.add_entity("test", "demo", "delete me too")
        e3 = fresh_dim.add_entity("test", "demo", "keep me")
        count = fresh_dim.delete_entities([e1, e2])
        assert count == 2
        assert fresh_dim.get_entity(e1) is None
        assert fresh_dim.get_entity(e2) is None
        assert fresh_dim.get_entity(e3) is not None


class TestSearch:
    def test_search_finds_results(self, dim_with_data):
        dim, ids = dim_with_data
        results = dim.search("hello world", n_results=5)
        assert len(results) >= 1

    def test_search_hybrid_mode(self, dim_with_data):
        dim, ids = dim_with_data
        results = dim.search("machine learning", n_results=5, mode="hybrid")
        assert len(results) >= 1

    def test_search_keyword_mode(self, dim_with_data):
        dim, ids = dim_with_data
        results = dim.search("python", n_results=5, mode="keyword")
        assert len(results) >= 1

    def test_search_vector_mode(self, dim_with_data):
        dim, ids = dim_with_data
        results = dim.search("brown fox", n_results=5, mode="vector")
        assert len(results) >= 1

    def test_search_with_realm_filter(self, dim_with_data):
        dim, ids = dim_with_data
        results = dim.search("hello", n_results=5, realm="test")
        assert len(results) >= 1

    def test_search_with_domain_filter(self, dim_with_data):
        dim, ids = dim_with_data
        results = dim.search("hello", n_results=5, domain="demo")
        assert len(results) >= 1

    def test_search_empty_query(self, dim_with_data):
        dim, ids = dim_with_data
        results = dim.search("", n_results=5)
        # Empty query falls through to vector search; should still have results
        assert isinstance(results, list)


class TestTaxonomy:
    def test_get_taxonomy(self, fresh_dim):
        fresh_dim.add_entity("tax", "a", "entity a")
        fresh_dim.add_entity("tax", "b", "entity b")
        tax = fresh_dim.get_taxonomy()
        assert "tax" in tax
        assert tax["tax"]["a"] >= 1
        assert tax["tax"]["b"] >= 1


class TestStatus:
    def test_status(self, dim_with_data):
        dim, ids = dim_with_data
        s = dim.status()
        assert s["entities"] >= 10
        assert "embedding" in s


class TestMemoriesFiledAway:
    def test_memories_filed_away(self, dim_with_data):
        dim, ids = dim_with_data
        result = dim.memories_filed_away()
        assert result["total_entities"] >= 10
        assert result["last_saved_at"] is not None
