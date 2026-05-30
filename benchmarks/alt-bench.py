#!/usr/bin/env python3
"""
alt-bench: permanent benchmark runner for Alt Memory across all 3 datasets.

Usage:
    python benchmarks/alt-bench.py all --limit 10
    python benchmarks/alt-bench.py longmemeval --backend chroma --embedder minilm --limit 10
    python benchmarks/alt-bench.py locomo --backend faiss --embedder sentence --limit 10
    python benchmarks/alt-bench.py membench --backend chroma --embedder minilm --limit 10
    python benchmarks/alt-bench.py matrix --limit 5   # all 6 backend×embedder combinations
"""

import json
import re
import sys
import time
import math
import tempfile
import argparse
import shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Factories ─────────────────────────────────────────────────────────


def _make_embedder(name):
    from alt_memory.backends.embedder import get_embedder
    return get_embedder(model=name)


def _make_store(backend, dim=384):
    from alt_memory.backends.types import DEFAULT_DIM
    tmp = tempfile.mkdtemp(prefix=f"alt_{backend}_")
    if backend == "faiss":
        from alt_memory.backends.faiss_store import FaissStore
        return FaissStore(tmp, dimension=dim), tmp
    elif backend == "chroma":
        from alt_memory.backends.chroma_store import ChromaStore
        return ChromaStore(tmp, dimension=dim), tmp
    raise ValueError(f"Unknown backend: {backend}")


BACKENDS = ["faiss", "chroma"]
EMBEDDERS = ["sentence", "minilm", "numpy"]

# Backend distance semantics:
# FaissStore: IndexFlatIP → inner product (higher=better)
# ChromaStore: ChromaDB IP space → 1-cosine (lower=better)
# Hybrid formula:
#   FaissStore:  fused = dist * (1.0 + 0.5*overlap)  reverse=True
#   ChromaStore: fused = dist * (1.0 - 0.5*overlap)  reverse=False


def _dist_boost(dist, overlap, backend):
    if backend == "faiss":
        return dist * (1.0 + 0.50 * overlap)
    return dist * (1.0 - 0.50 * overlap)


# ── Helpers ───────────────────────────────────────────────────────────


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
    return [w for w in re.findall(r"\b[a-z]{3,}\b", text.lower()) if w not in STOP_WORDS]


def _kw_overlap(qkws, doc):
    if not qkws:
        return 0.0
    return sum(1 for kw in qkws if kw in doc.lower()) / len(qkws)


def _person_names(text):
    return list(set(re.findall(r"\b[A-Z][a-z]{2,15}\b", text)) - NOT_NAMES)


# ── Data Loaders ──────────────────────────────────────────────────────


def load_longmemeval(path, granularity="session", limit=0):
    """LongMemEval: list of {haystack_sessions, haystack_session_ids,
    question, answer_session_ids, ...}"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if limit:
        data = data[:limit]

    items = []
    for entry in data:
        sessions = entry["haystack_sessions"]
        sess_ids = entry["haystack_session_ids"]
        answer_sids = set(entry["answer_session_ids"])

        corpus = []
        corpus_ids = []
        for session, sid in zip(sessions, sess_ids):
            if granularity == "session":
                texts = [t["content"] for t in session if t["role"] == "user"]
                if texts:
                    corpus.append("\n".join(texts))
                    corpus_ids.append(sid)
            else:
                for tnum, turn in enumerate(session):
                    if turn["role"] == "user":
                        corpus.append(turn["content"])
                        corpus_ids.append(f"{sid}_turn_{tnum}")

        items.append({
            "corpus": corpus,
            "corpus_ids": corpus_ids,
            "question": entry["question"],
            "target_ids": answer_sids,
            "question_id": entry.get("question_id", ""),
            "question_type": entry.get("question_type", ""),
        })
    return items


def load_locomo(path, granularity="dialog", limit=0):
    """LoCoMo: list of {conversation, qa, ...}"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if limit:
        data = data[:limit]

    items = []
    for conv_idx, sample in enumerate(data):
        conversation = sample["conversation"]
        qa_pairs = sample["qa"]

        sessions = []
        snum = 1
        while True:
            key = f"session_{snum}"
            if key not in conversation:
                break
            sessions.append({
                "num": snum,
                "dialogs": conversation[key],
                "date": conversation.get(f"session_{snum}_date_time", ""),
            })
            snum += 1

        corpus = []
        corpus_ids = []
        for sess in sessions:
            if granularity == "session":
                texts = [f'{d.get("speaker","?")} said, "{d.get("text","")}"' for d in sess["dialogs"]]
                corpus.append("\n".join(texts))
                corpus_ids.append(f"session_{sess['num']}")
            else:
                for d in sess["dialogs"]:
                    did = d.get("dia_id", f"D{sess['num']}:?")
                    corpus.append(f'{d.get("speaker","?")} said, "{d.get("text","")}"')
                    corpus_ids.append(did)

        for qa in qa_pairs:
            evidence = qa.get("evidence", [])
            if granularity == "dialog":
                targets = set(evidence)
            else:
                targets = set()
                for eid in evidence:
                    m = re.match(r"D(\d+):", eid)
                    if m:
                        targets.add(f"session_{m.group(1)}")

            items.append({
                "corpus": corpus,
                "corpus_ids": corpus_ids,
                "question": qa["question"],
                "target_ids": targets,
                "category": qa.get("category", 0),
                "sample_id": sample.get("sample_id", f"conv-{conv_idx}"),
            })
    return items


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


def load_membench(data_dir, limit=0):
    """MemBench: per-category JSON files, each with conversation items."""
    data_dir = Path(data_dir)
    items = []
    for cat, fname in CATEGORY_FILES.items():
        fpath = data_dir / fname
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8") as f:
            raw = json.load(f)
        for t, topic_items in raw.items():
            if t not in ("movie", "roles", "events"):
                continue
            for item in topic_items:
                turns = item.get("message_list", [])
                qa = item.get("QA", {})
                if not turns or not qa:
                    continue

                # Flatten all sessions into one document list
                session_list = [turns] if (turns and isinstance(turns[0], dict)) else turns
                docs = []
                doc_ids = []
                gidx = 0
                for session in session_list:
                    if not isinstance(session, list):
                        continue
                    for turn in session:
                        if not isinstance(turn, dict):
                            continue
                        sid = turn.get("sid", turn.get("mid", gidx))
                        user = turn.get("user") or turn.get("user_message", "")
                        asst = turn.get("assistant") or turn.get("assistant_message", "")
                        text = f"[User] {user} [Assistant] {asst}"
                        t = turn.get("time", "")
                        if t:
                            text = f"[{t}] {text}"
                        docs.append(text)
                        doc_ids.append(str(int(sid) if isinstance(sid, (int, float)) else gidx))
                        gidx += 1

                target = qa.get("target_step_id", [])
                target_ids = {str(s[0]) for s in target if isinstance(s, list) and len(s) >= 1}

                items.append({
                    "corpus": docs,
                    "corpus_ids": doc_ids,
                    "question": qa.get("question", ""),
                    "target_ids": target_ids,
                    "category": cat,
                })

    if limit:
        items = items[:limit]
    return items


# ── Search ────────────────────────────────────────────────────────────


def search_store(store, question, embedder, backend, top_k, mode, corpus_len):
    n_retrieve = min(top_k * 3 if mode == "hybrid" else top_k, corpus_len)
    if n_retrieve < 1:
        return []

    query_emb = embedder.embed([question])
    import numpy as np
    query_emb = np.asarray(query_emb, dtype=np.float32)
    ids, texts, dists, metas = store.search(query_emb, n_results=n_retrieve)

    if mode == "hybrid" and len(dists) > 0:
        names = _person_names(question)
        name_words = {n.lower() for n in names}
        all_kws = _kw(question)
        pred_kws = [w for w in all_kws if w not in name_words]

        scored = []
        # Extract corpus_id from metadata (field varies by store)
        for d, txt, meta in zip(dists, texts, metas):
            cid = meta.get("corpus_id", meta.get("sid", meta.get("global_idx", "")))
            overlap = _kw_overlap(pred_kws, txt)
            if overlap > 0:
                fused = _dist_boost(d, overlap, backend)
                scored.append((fused, str(cid)))
            else:
                scored.append((d, str(cid)))

        scored.sort(key=lambda x: x[0], reverse=(backend == "faiss"))
        return [x[1] for x in scored[:top_k]]

    # Raw mode: return corpus_ids from metadata
    result = []
    for meta in metas:
        cid = meta.get("corpus_id", meta.get("sid", meta.get("global_idx", "")))
        result.append(str(cid))
    return result[:top_k]


# ── Benchmark Runners ────────────────────────────────────────────────


def run_benchmark(name, items, embedder, backend, top_k=5, mode="hybrid"):
    """Run one benchmark config. Returns {hits, total, results_log, elapsed, embed_time}."""
    hits = 0
    total = 0
    results_log = []
    total_embed = 0.0
    t0_all = time.time()
    tmp_dirs = []

    for item in items:
        corpus = item["corpus"]
        corpus_ids = item["corpus_ids"]
        if not corpus:
            continue
        total += 1

        store, tmp_dir = _make_store(backend)
        tmp_dirs.append(tmp_dir)

        t0 = time.time()
        embeddings = embedder.embed(corpus)
        doc_ids = [f"d{i}" for i in range(len(corpus))]
        metadatas = [{"corpus_id": cid} for cid in corpus_ids]
        store.add(ids=doc_ids, texts=corpus, metadatas=metadatas, embeddings=embeddings)
        total_embed += time.time() - t0

        retrieved_ids = search_store(store, item["question"], embedder, backend,
                                     top_k, mode, len(corpus))
        store.close()

        target_ids = item["target_ids"]
        if not isinstance(target_ids, set):
            target_ids = set(target_ids)
        hit = bool(target_ids & set(retrieved_ids))

        if hit:
            hits += 1
        results_log.append({
            "question": item.get("question", "")[:80],
            "target_ids": list(target_ids),
            "retrieved_ids": retrieved_ids,
            "hit": hit,
        })

    elapsed = time.time() - t0_all
    for d in tmp_dirs:
        shutil.rmtree(d, ignore_errors=True)

    return {"hits": hits, "total": total, "results": results_log,
            "elapsed": elapsed, "embed_time": total_embed}


def print_results(name, config_label, r):
    total = r["total"]
    if total == 0:
        print(f"  {config_label:30}  NO DATA")
        return
    pct = r["hits"] / total * 100
    print(f"  {config_label:30}  R@{5}: {pct:5.1f}%  ({r['hits']}/{total})  "
          f"{r['elapsed']:.1f}s  (embed: {r['embed_time']:.1f}s)")


# ── Data paths (auto-detect) ─────────────────────────────────────────


def _find_data():
    """Auto-detect data paths using common locations."""
    tmp = Path.home() / "AppData" / "Local" / "Temp"
    paths = {
        "longmemeval": tmp / "longmemeval-data" / "longmemeval_s_cleaned.json",
        "locomo": tmp / "locomo_cache" / "locomo10.json",
        "membench": tmp / "membench" / "Membench-main" / "MemData" / "FirstAgent",
    }
    return {k: v for k, v in paths.items() if v.exists()}


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Alt Memory Benchmark Runner")
    ap.add_argument("command", nargs="?", default="all",
                    choices=["all", "matrix", "longmemeval", "locomo", "membench"],
                    help="Run specific benchmark or 'all' / 'matrix'")
    ap.add_argument("--backend", choices=BACKENDS, default="faiss")
    ap.add_argument("--embedder", choices=EMBEDDERS, default="sentence")
    ap.add_argument("--limit", type=int, default=10, help="Items per benchmark")
    ap.add_argument("--top-k", type=int, default=5, help="Top-k retrieval")
    ap.add_argument("--granularity", choices=["session", "turn", "dialog"],
                    default="session",
                    help="Corpus granularity (longmemeval/locomo)")
    ap.add_argument("--data-dir", default=None, help="Override data directory")
    args = ap.parse_args()

    data = _find_data()

    # ── Single benchmark run ──────────────────────────────────────
    if args.command in ("longmemeval", "locomo", "membench"):
        if args.command not in data:
            print(f"Data for '{args.command}' not found at expected paths.")
            print(f"Expected: {[str(v) for k,v in data.items() if k==args.command]}")
            return

        bm = args.command
        print(f"\n{'='*60}")
        print(f"  {bm.upper()}  |  backend={args.backend}  embedder={args.embedder}")
        print(f"{'='*60}")

        embedder = _make_embedder(args.embedder)
        _ = embedder.embed(["warmup"])

        if bm == "longmemeval":
            items = load_longmemeval(data[bm], granularity=args.granularity, limit=args.limit)
        elif bm == "locomo":
            items = load_locomo(data[bm], granularity=args.granularity, limit=args.limit)
        else:
            items = load_membench(data[bm], limit=args.limit)

        print(f"  Questions:   {len(items)}")
        print(f"  Backend:     {args.backend}")
        print(f"  Embedder:    {args.embedder}")
        print(f"  Top-k:       {args.top_k}")
        print()

        r = run_benchmark(bm, items, embedder, args.backend, top_k=args.top_k)
        print_results(bm, f"{args.backend}+{args.embedder}", r)
        print()

    # ── Run all benchmarks with one config ────────────────────────
    elif args.command == "all":
        print(f"\n{'='*60}")
        print(f"  ALL BENCHMARKS  |  backend={args.backend}  embedder={args.embedder}")
        print(f"{'='*60}")

        embedder = _make_embedder(args.embedder)
        _ = embedder.embed(["warmup"])

        for bm in ["longmemeval", "locomo", "membench"]:
            if bm not in data:
                print(f"  SKIP {bm}: data not found")
                continue

            print(f"\n  ── {bm.upper()} ──")
            if bm == "longmemeval":
                items = load_longmemeval(data[bm], granularity=args.granularity, limit=args.limit)
            elif bm == "locomo":
                items = load_locomo(data[bm], granularity=args.granularity, limit=args.limit)
            else:
                items = load_membench(data[bm], limit=args.limit)

            print(f"  Questions: {len(items)}")
            r = run_benchmark(bm, items, embedder, args.backend, top_k=args.top_k)
            print_results(bm, f"{args.backend}+{args.embedder}", r)

        print()

    # ── Matrix: all backend×embedder combos ───────────────────────
    elif args.command == "matrix":
        print(f"\n{'='*60}")
        print(f"  MATRIX  (limit={args.limit})")
        print(f"{'='*60}\n")

        for bm in ["longmemeval", "locomo", "membench"]:
            if bm not in data:
                print(f"  {bm.upper()}: SKIP (no data)\n")
                continue

            print(f"  ── {bm.upper()} ──")

            if bm == "longmemeval":
                items = load_longmemeval(data[bm], granularity=args.granularity, limit=args.limit)
            elif bm == "locomo":
                items = load_locomo(data[bm], granularity=args.granularity, limit=args.limit)
            else:
                items = load_membench(data[bm], limit=args.limit)

            print(f"  Questions: {len(items)}\n")

            for backend in BACKENDS:
                for embed_name in EMBEDDERS:
                    try:
                        embedder = _make_embedder(embed_name)
                        _ = embedder.embed(["warmup"])
                    except Exception as e:
                        print(f"  {backend}+{embed_name:20}  EMBED FAIL: {e}")
                        continue

                    label = f"{backend}+{embed_name}"
                    r = run_benchmark(bm, items, embedder, backend, top_k=args.top_k)
                    print_results(bm, label, r)

            print()

    else:
        ap.print_help()


if __name__ == "__main__":
    main()
