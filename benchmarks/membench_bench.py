#!/usr/bin/env python3
"""
Alt Memory x MemBench Benchmark

MemBench (ACL 2025): https://aclanthology.org/2025.findings-acl.989/
Data: https://github.com/import-myself/Membench

Uses alt-memory's ChromaStore backend + ONNX embedder.
"""

import json
import re
import sys
import time
import argparse
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def _make_embedder():
    from alt_memory.backends.embedder import get_embedder
    return get_embedder(model="minilm")


def _make_store():
    from alt_memory.backends.chroma_store import ChromaStore
    tmp = tempfile.mkdtemp(prefix="alt_membench_")
    store = ChromaStore(tmp, dimension=384)
    return store, tmp


STOP_WORDS = {
    "what", "when", "where", "who", "how", "which",
    "did", "do", "was", "were", "have", "has", "had",
    "is", "are", "the", "a", "an", "my", "me", "i",
    "you", "your", "their", "it", "its", "in", "on",
    "at", "to", "for", "of", "with", "by", "from",
    "ago", "last", "that", "this", "there", "about",
    "get", "got", "give", "gave", "buy", "bought",
    "made", "make", "said", "would", "could", "should",
    "might", "can", "will", "shall", "kind", "type",
    "like", "prefer", "enjoy", "think", "feel",
}

NOT_NAMES = {
    "What", "When", "Where", "Who", "How", "Which",
    "Did", "Do", "Was", "Were", "Have", "Has", "Had",
    "Is", "Are", "The", "My", "Our", "I", "It", "Its",
    "This", "That", "These", "Those",
}


def _kw(text):
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [w for w in words if w not in STOP_WORDS]


def _kw_overlap(query_kws, doc_text):
    if not query_kws:
        return 0.0
    doc_lower = doc_text.lower()
    hits = sum(1 for kw in query_kws if kw in doc_lower)
    return hits / len(query_kws)


def _person_names(text):
    words = re.findall(r"\b[A-Z][a-z]{2,15}\b", text)
    return list(set(w for w in words if w not in NOT_NAMES))


CATEGORY_FILES = {
    "simple": "simple.json",
    "highlevel": "highlevel.json",
    "knowledge_update": "knowledge_update.json",
    "comparative": "comparative.json",
    "conditional": "conditional.json",
    "noisy": "noisy.json",
    "aggregative": "aggregative.json",
    "highlevel_rec": "highlevel_rec.json",
    "lowlevel_rec": "lowlevel_rec.json",
    "RecMultiSession": "RecMultiSession.json",
    "post_processing": "post_processing.json",
}


def load_membench(data_dir: str, categories=None, topic="movie", limit=0):
    data_dir = Path(data_dir)
    if categories is None:
        categories = list(CATEGORY_FILES.keys())

    items = []
    for cat in categories:
        fname = CATEGORY_FILES.get(cat)
        if not fname:
            continue
        fpath = data_dir / fname
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8") as f:
            raw = json.load(f)

        for t, topic_items in raw.items():
            if topic and t not in (topic, "roles", "events"):
                continue
            for item in topic_items:
                turns = item.get("message_list", [])
                qa = item.get("QA", {})
                if not turns or not qa:
                    continue
                items.append({
                    "category": cat,
                    "topic": t,
                    "tid": item.get("tid", 0),
                    "turns": turns,
                    "question": qa.get("question", ""),
                    "choices": qa.get("choices", {}),
                    "ground_truth": qa.get("ground_truth", ""),
                    "answer_text": qa.get("answer", ""),
                    "target_step_ids": qa.get("target_step_id", []),
                })

    if limit > 0:
        items = items[:limit]
    return items


def _turn_text(turn: dict) -> str:
    user = turn.get("user") or turn.get("user_message", "")
    asst = turn.get("assistant") or turn.get("assistant_message", "")
    time_val = turn.get("time", "")
    text = f"[User] {user} [Assistant] {asst}"
    if time_val:
        text = f"[{time_val}] " + text
    return text


def run_membench(
    data_dir, categories=None, topic="movie", top_k=5, limit=0, mode="raw", out_file=None,
):
    items = load_membench(data_dir, categories=categories, topic=topic, limit=limit)
    if not items:
        print(f"No items found in {data_dir}")
        return

    print(f"\n{'=' * 58}")
    print("  Alt Memory (ChromaStore + ONNX) x MemBench")
    print(f"{'=' * 58}")
    print(f"  Data dir:    {data_dir}")
    print(f"  Categories:  {', '.join(categories or ['all'])}")
    print(f"  Topic:       {topic or 'all'}")
    print(f"  Items:       {len(items)}")
    print(f"  Top-k:       {top_k}")
    print(f"  Mode:        {mode}")
    print(f"{'-' * 58}\n")

    embedder = _make_embedder()
    _ = embedder.embed(["warmup"])

    results = []
    by_cat = defaultdict(lambda: {"hit_at_k": 0, "total": 0})
    total_hit = 0
    total_elapsed = 0.0
    tmp_dirs = []
    start_time = datetime.now()

    for item_idx, item in enumerate(items, 1):
        turns = item["turns"]

        session_list = turns
        if turns and isinstance(turns[0], dict):
            session_list = [turns]

        docs = []
        sids = []
        global_indices = []
        global_idx = 0
        for session in session_list:
            if not isinstance(session, list):
                continue
            for turn in session:
                if not isinstance(turn, dict):
                    continue
                sid = turn.get("sid", turn.get("mid"))
                text = _turn_text(turn)
                docs.append(text)
                sids.append(int(sid) if isinstance(sid, (int, float)) else global_idx)
                global_indices.append(global_idx)
                global_idx += 1

        if not docs:
            continue

        store, tmp_dir = _make_store()
        tmp_dirs.append(tmp_dir)

        t0 = time.time()
        n_turns = len(docs)
        embeddings = embedder.embed(docs)
        ids = [f"t{i}" for i in range(n_turns)]
        metadatas = [
            {"sid": sids[i], "global_idx": global_indices[i]}
            for i in range(n_turns)
        ]
        store.add(ids=ids, texts=docs, metadatas=metadatas, embeddings=embeddings)
        t1 = time.time()
        total_elapsed += t1 - t0

        question = item["question"]
        n_retrieve = min(top_k * 3 if mode == "hybrid" else top_k, n_turns)
        if n_retrieve < 1:
            store.close()
            continue

        query_emb = embedder.embed([question])
        query_emb = np.asarray(query_emb, dtype=np.float32)

        q_results = store.search(query_emb, n_results=n_retrieve)
        retrieved_ids, retrieved_docs, dists, retrieved_metas = q_results
        retrieved_sids = [m.get("sid", -1) for m in retrieved_metas]
        retrieved_global = [m.get("global_idx", -1) for m in retrieved_metas]

        if mode == "hybrid":
            names = _person_names(question)
            name_words = {n.lower() for n in names}
            all_kws = _kw(question)
            predicate_kws = [w for w in all_kws if w not in name_words]

            scored = []
            for dist, sid, gidx, doc in zip(
                dists, retrieved_sids, retrieved_global, retrieved_docs
            ):
                pred_overlap = _kw_overlap(predicate_kws, doc)
                fused = dist * (1.0 - 0.50 * pred_overlap)
                scored.append((fused, sid, gidx, doc))
            scored.sort(key=lambda x: x[0])
            retrieved_sids = [x[1] for x in scored[:top_k]]
            retrieved_global = [x[2] for x in scored[:top_k]]
        else:
            retrieved_sids = retrieved_sids[:top_k]
            retrieved_global = retrieved_global[:top_k]

        store.close()

        target_sids = set()
        for step in item["target_step_ids"]:
            if isinstance(step, list) and len(step) >= 1:
                target_sids.add(step[0])

        hit = bool(target_sids & set(retrieved_sids)) or bool(target_sids & set(retrieved_global))
        if hit:
            total_hit += 1
            by_cat[item["category"]]["hit_at_k"] += 1
        by_cat[item["category"]]["total"] += 1

        results.append({
            "category": item["category"],
            "topic": item["topic"],
            "tid": item["tid"],
            "question": question,
            "ground_truth": item["ground_truth"],
            "answer_text": item["answer_text"],
            "target_sids": list(target_sids),
            "retrieved_sids": retrieved_sids,
            "retrieved_global": retrieved_global,
            "hit_at_k": hit,
        })

        if item_idx % 50 == 0:
            running_pct = total_hit / item_idx * 100
            print(f"  [{item_idx:4}/{len(items)}]  running R@{top_k}: {running_pct:.1f}%")

    overall = total_hit / len(items) * 100 if items else 0
    wall_clock = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 58}")
    print(f"  RESULTS - Alt Memory (ChromaStore) on MemBench ({mode} mode, top-{top_k})")
    print(f"{'=' * 58}")
    print(f"\n  Overall R@{top_k}: {overall:.1f}%  ({total_hit}/{len(items)})")
    print(f"  Time:        {wall_clock:.1f}s ({wall_clock / max(len(items), 1):.2f}s per item)")
    print(f"  Embed/store: {total_elapsed:.1f}s\n")
    print("  By category:")
    for cat, v in sorted(by_cat.items()):
        pct = v["hit_at_k"] / v["total"] * 100 if v["total"] else 0
        print(f"    {cat:20} {pct:5.1f}%  ({v['hit_at_k']}/{v['total']})")
    print(f"\n{'=' * 58}\n")

    for d in tmp_dirs:
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass

    if out_file:
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to: {out_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alt Memory x MemBench Benchmark")
    parser.add_argument("data_dir", help="Path to MemBench FirstAgent directory")
    parser.add_argument(
        "--category", default=None, choices=list(CATEGORY_FILES.keys()),
        help="Run a single category (default: all)",
    )
    parser.add_argument(
        "--topic", default="movie",
        help="Topic filter: movie, food, book (default: movie)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval top-k (default: 5)")
    parser.add_argument("--limit", type=int, default=0, help="Limit items (0 = all)")
    parser.add_argument(
        "--mode", choices=["raw", "hybrid"], default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    parser.add_argument("--out", default=None, help="Output JSON file")
    args = parser.parse_args()

    if not args.out:
        cat_tag = f"_{args.category}" if args.category else "_all"
        args.out = (
            f"benchmarks/results_alt_chroma_membench_{args.mode}{cat_tag}_{args.topic}"
            f"_top{args.top_k}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        )

    cats = [args.category] if args.category else None
    run_membench(
        args.data_dir, categories=cats, topic=args.topic,
        top_k=args.top_k, limit=args.limit, mode=args.mode, out_file=args.out,
    )
