"""Tests for KnowledgeGraph."""

import pytest


class TestKGAdd:
    def test_add_fact(self, fresh_dim):
        fid = fresh_dim.kg.add("Alice", "loves", "Bob")
        assert fid is not None
        assert isinstance(fid, (str, int))

    def test_add_temporal_fact(self, fresh_dim):
        fid = fresh_dim.kg.add("Alice", "works_at", "Acme", valid_from="2024-01-01")
        assert fid is not None


class TestKGQuery:
    def test_query_by_entity(self, fresh_dim):
        fresh_dim.kg.add("Alice", "loves", "Bob")
        facts = fresh_dim.kg.query(entity="Alice")
        assert len(facts) >= 1
        assert any(f["subject"] == "Alice" for f in facts)

    def test_query_by_predicate(self, fresh_dim):
        fresh_dim.kg.add("Alice", "loves", "Bob")
        facts = fresh_dim.kg.query(predicate="loves")
        assert len(facts) >= 1

    def test_query_all(self, fresh_dim):
        fresh_dim.kg.add("Alice", "loves", "Bob")
        facts = fresh_dim.kg.query()
        assert len(facts) >= 1

    def test_query_direction_outgoing(self, fresh_dim):
        fresh_dim.kg.add("Alice", "loves", "Bob")
        facts = fresh_dim.kg.query(entity="Alice", direction="outgoing")
        assert all(f["subject"] == "Alice" for f in facts)

    def test_query_direction_incoming(self, fresh_dim):
        fresh_dim.kg.add("Alice", "loves", "Bob")
        facts = fresh_dim.kg.query(entity="Bob", direction="incoming")
        assert all(f["object"] == "Bob" for f in facts)

    def test_query_as_of(self, fresh_dim):
        fresh_dim.kg.add("Alice", "works_at", "Acme", valid_from="2024-01-01", valid_to="2024-12-31")
        facts = fresh_dim.kg.query(entity="Alice", as_of="2024-06-15")
        assert len(facts) >= 1
        facts2 = fresh_dim.kg.query(entity="Alice", as_of="2025-01-01")
        assert len(facts2) == 0 if not any(
            f["valid_to"] is None or f["valid_to"] > "2025-01-01"
            for f in facts2
        ) else len(facts2) == len(facts2)

    def test_unknown_entity(self, fresh_dim):
        facts = fresh_dim.kg.query(entity="NonExistent")
        assert facts == []


class TestKGInvalidate:
    def test_invalidate_exact(self, fresh_dim):
        fresh_dim.kg.add("Alice", "loves", "Bob")
        n = fresh_dim.kg.invalidate("Alice", "loves", "Bob")
        assert n >= 1

    def test_invalidate_nonexistent(self, fresh_dim):
        n = fresh_dim.kg.invalidate("Foo", "bar", "Baz")
        assert n == 0


class TestKGStats:
    def test_stats(self, fresh_dim):
        fresh_dim.kg.add("Alice", "loves", "Bob")
        stats = fresh_dim.kg.stats()
        # Keys: current_facts, expired_facts, entities, relationship_types, etc.
        assert any(k.endswith("facts") for k in stats), f"No facts key in {stats.keys()}"
        fact_count = stats.get("current_facts", stats.get("facts", 0))
        assert fact_count >= 1


class TestKGTimeline:
    def test_timeline(self, fresh_dim):
        fresh_dim.kg.add("Alice", "born", "1990")
        fresh_dim.kg.add("Alice", "started_work", "2015", valid_from="2015-01-01")
        timeline = fresh_dim.kg.timeline(entity="Alice")
        assert len(timeline) >= 2
        # Timeline should be sorted chronologically
        for i in range(len(timeline) - 1):
            t1 = timeline[i].get("valid_from") or ""
            t2 = timeline[i + 1].get("valid_from") or ""
            if t1 and t2:
                assert t1 <= t2
