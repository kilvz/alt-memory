#!/usr/bin/env python3
"""
Alt Memory x LongMemEval Benchmark

Uses alt-memory's embedder + raw FAISS (in-memory, no disk I/O)
for pure retrieval recall measurement.
"""

import json
import math
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss
import numpy as np


def _make_alt_embedder():
    from alt_memory.backends.embedder import get_embedder
    ef = get_embedder(model="sentence")
    return ef


def _make_fastembed_embedder():
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    class _FastWrapper:
        dimension = 384
        def embed(self, texts):
            return np.array(list(model.embed(texts)), dtype=np.float32)
    return _FastWrapper()


def _make_chromadb_embedder():
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
    model = ONNXMiniLM_L6_V2()
    class _ChromaWrapper:
        dimension = 384
        def embed(self, texts):
            return np.array(model(texts), dtype=np.float32)
    return _ChromaWrapper()



# ── ChromaDB-native backend (EphemeralClient, fused C++) ────────────


def _run_chromadb_native(data, granularity, ks, metrics, per_type, results_log, start_time):
    """Run using ChromaDB EphemeralClient — delete → create → add → query."""
    import chromadb
    _bench_client = chromadb.EphemeralClient()
    embed_times = []
    search_times = []
    other_times = []
    total_docs = 0

    for i, entry in enumerate(data):
        qid = entry["question_id"]
        qtype = entry["question_type"]
        answer_sids = set(entry["answer_session_ids"])
        sessions = entry["haystack_sessions"]
        session_ids = entry["haystack_session_ids"]

        t0 = time.time()

        corpus = []
        corpus_ids = []
        for session, sess_id in zip(sessions, session_ids):
            if granularity == "session":
                user_turns = [t["content"] for t in session if t["role"] == "user"]
                if user_turns:
                    corpus.append("\n".join(user_turns))
                    corpus_ids.append(sess_id)
            else:
                turn_num = 0
                for turn in session:
                    if turn["role"] == "user":
                        corpus.append(turn["content"])
                        corpus_ids.append(f"{sess_id}_turn_{turn_num}")
                        turn_num += 1

        if not corpus:
            print(f"  [{i + 1:4}/{len(data)}] {qid[:30]:30} SKIP (empty)")
            continue

        try:
            _bench_client.delete_collection("mempal")
        except Exception:
            pass
        col = _bench_client.create_collection("mempal")
        t1 = time.time()

        col.add(
            documents=corpus,
            ids=[f"d_{j}" for j in range(len(corpus))],
            metadatas=[{"corpus_id": cid} for cid in corpus_ids],
        )
        t2 = time.time()
        total_docs += len(corpus)

        results = col.query(query_texts=[entry["question"]], n_results=min(50, len(corpus)))
        t3 = time.time()

        embed_times.append(t2 - t1)
        search_times.append(t3 - t2)
        other_times.append(t1 - t0)

        doc_ids = results.get("ids", [[]])[0] if results else []

        doc_id_to_idx = {}
        for rid in doc_ids:
            j = int(rid.split("_")[1])
            doc_id_to_idx[j] = len(doc_id_to_idx)

        ranked_indices = list(doc_id_to_idx.keys())
        seen = set(ranked_indices)
        for j in range(len(corpus)):
            if j not in seen:
                ranked_indices.append(j)

        if not ranked_indices:
            continue

        session_level_ids = [
            cid.rsplit("_turn_", 1)[0] if "_turn_" in cid else cid
            for cid in corpus_ids
        ]

        for k in ks:
            ra, rl, nd = evaluate_retrieval(ranked_indices, answer_sids, session_level_ids, k)
            metrics[f"recall_any@{k}"].append(ra)
            metrics[f"recall_all@{k}"].append(rl)
            metrics[f"ndcg_any@{k}"].append(nd)

        per_type[qtype]["recall_any@5"].append(metrics["recall_any@5"][-1])
        per_type[qtype]["recall_any@10"].append(metrics["recall_any@10"][-1])

        r5 = int(metrics["recall_any@5"][-1])
        r10 = int(metrics["recall_any@10"][-1])
        status = "HIT" if r5 > 0 else "miss"
        elapsed_q = t3 - t0
        print(f"  [{i + 1:4}/{len(data)}] {qid[:30]:30} R@5={r5} R@10={r10}  {status}  {elapsed_q:.1f}s")

        results_log.append({
            "question_id": qid,
            "question_type": qtype,
            "metrics": {f"recall_any@{k}": metrics[f"recall_any@{k}"][-1] for k in ks},
        })

    elapsed = (datetime.now() - start_time).total_seconds()
    n = len(data)
    return elapsed, n, embed_times, search_times, total_docs, other_times


EMBEDDER_FACTORIES = {
    "alt": _make_alt_embedder,
    "fastembed": _make_fastembed_embedder,
    "chromadb": _make_chromadb_embedder,
}


def dcg(relevances, k):
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        score += rel / math.log2(i + 2)
    return score


def ndcg(rankings, correct_ids, corpus_ids, k):
    relevances = [1.0 if corpus_ids[idx] in correct_ids else 0.0 for idx in rankings[:k]]
    ideal = sorted(relevances, reverse=True)
    idcg = dcg(ideal, k)
    return 0.0 if idcg == 0 else dcg(relevances, k) / idcg


def evaluate_retrieval(rankings, correct_ids, corpus_ids, k):
    top_k_ids = set(corpus_ids[idx] for idx in rankings[:k])
    recall_any = float(any(cid in top_k_ids for cid in correct_ids))
    recall_all = float(all(cid in top_k_ids for cid in correct_ids))
    ndcg_score = ndcg(rankings, correct_ids, corpus_ids, k)
    return recall_any, recall_all, ndcg_score


def run_benchmark(data_file, granularity="session", limit=0, skip=0, out_file=None, backend="alt"):
    with open(data_file, encoding="utf-8") as f:
        data = json.load(f)

    if limit > 0:
        data = data[:limit]
    if skip > 0:
        print(f"  Skipping first {skip} questions")
        data = data[skip:]

    from alt_memory import __version__ as alt_version

    print()
    print("=" * 60)
    print("  Alt Memory x LongMemEval Benchmark")
    print("=" * 60)
    print(f"  Data:        {Path(data_file).name}")
    print(f"  Questions:   {len(data)}")
    print(f"  Granularity: {granularity}")
    print(f"  Version:     alt-memory v{alt_version}")
    print("-" * 60)

    ks = [1, 3, 5, 10, 30, 50]
    metrics = {f"recall_any@{k}": [] for k in ks}
    metrics.update({f"recall_all@{k}": [] for k in ks})
    metrics.update({f"ndcg_any@{k}": [] for k in ks})
    per_type = defaultdict(lambda: defaultdict(list))
    results_log = []
    start_time = datetime.now()

    if backend == "chromadb-native":
        print("  Backend:     chromadb-native (EphemeralClient)")
        print()
        elapsed, n, embed_times, search_times, total_docs, other_times = _run_chromadb_native(
            data, granularity, ks, metrics, per_type, results_log, start_time,
        )
    else:
        print(f"  Backend:     {backend}")
        print("  Warming up embedder...")
        embedder = EMBEDDER_FACTORIES[backend]()
        _ = embedder.embed(["warmup text to pre-load model weights"])
        print(f"  Embedder:    {type(embedder).__name__}")
        print()

        total_docs = 0
        embed_times = []
        search_times = []
        other_times = []

        try:
            for i, entry in enumerate(data):
                qid = entry["question_id"]
                qtype = entry["question_type"]
                answer_sids = set(entry["answer_session_ids"])

                t0 = time.time()

                corpus = []
                corpus_ids = []
                sessions = entry["haystack_sessions"]
                session_ids = entry["haystack_session_ids"]

                for session, sess_id in zip(sessions, session_ids):
                    if granularity == "session":
                        user_turns = [t["content"] for t in session if t["role"] == "user"]
                        if user_turns:
                            corpus.append("\n".join(user_turns))
                            corpus_ids.append(sess_id)
                    else:
                        turn_num = 0
                        for turn in session:
                            if turn["role"] == "user":
                                corpus.append(turn["content"])
                                corpus_ids.append(f"{sess_id}_turn_{turn_num}")
                                turn_num += 1

                if not corpus:
                    print(f"  [{i + 1:4}/{len(data)}] {qid[:30]:30} SKIP (empty)")
                    continue

                t1 = time.time()

                dim = embedder.dimension
                embeddings = embedder.embed(corpus)
                t2 = time.time()

                index = faiss.IndexFlatIP(dim)
                faiss.normalize_L2(embeddings)
                index.add(embeddings)
                total_docs += len(corpus)

                query_vec = embedder.embed([entry["question"]])
                query_vec = np.asarray(query_vec, dtype=np.float32)
                faiss.normalize_L2(query_vec)
                k = min(50, len(corpus))
                distances, indices_arr = index.search(query_vec, k)
                t3 = time.time()

                result_ids = [f"d_{i}" for i in indices_arr[0].tolist() if i != -1]

                embed_times.append(t2 - t1)
                search_times.append(t3 - t2)
                other_times.append(t1 - t0)

                doc_id_to_idx = {}
                for rid in result_ids:
                    j = int(rid.split("_")[1])
                    if j not in doc_id_to_idx:
                        doc_id_to_idx[j] = len(doc_id_to_idx)

                ranked_indices = list(doc_id_to_idx.keys())
                seen_set = set(ranked_indices)
                for j in range(len(corpus)):
                    if j not in seen_set:
                        ranked_indices.append(j)

                if not ranked_indices:
                    continue

                session_level_ids = [
                    cid.rsplit("_turn_", 1)[0] if "_turn_" in cid else cid
                    for cid in corpus_ids
                ]

                for k in ks:
                    ra, rl, nd = evaluate_retrieval(ranked_indices, answer_sids, session_level_ids, k)
                    metrics[f"recall_any@{k}"].append(ra)
                    metrics[f"recall_all@{k}"].append(rl)
                    metrics[f"ndcg_any@{k}"].append(nd)

                per_type[qtype]["recall_any@5"].append(metrics["recall_any@5"][-1])
                per_type[qtype]["recall_any@10"].append(metrics["recall_any@10"][-1])

                r5 = int(metrics["recall_any@5"][-1])
                r10 = int(metrics["recall_any@10"][-1])
                status = "HIT" if r5 > 0 else "miss"
                elapsed_q = t3 - t0
                print(f"  [{i + 1:4}/{len(data)}] {qid[:30]:30} R@5={r5} R@10={r10}  {status}  {elapsed_q:.1f}s")

                results_log.append({
                    "question_id": qid,
                    "question_type": qtype,
                    "metrics": {f"recall_any@{k}": metrics[f"recall_any@{k}"][-1] for k in ks},
                })

        except KeyboardInterrupt:
            print("\n  Interrupted by user.")

    elapsed = (datetime.now() - start_time).total_seconds()
    n = len(data)

    print()
    print("=" * 60)
    print("  RESULTS - Alt Memory (raw mode)")
    print("=" * 60)
    print(f"  Time: {elapsed:.1f}s ({elapsed / max(n, 1):.2f}s per question)")
    print(f"  Documents: {total_docs}")
    if embed_times:
        print(f"  Embedding:  {sum(embed_times):.1f}s total ({sum(embed_times)/len(embed_times):.2f}s/q)")
        print(f"  Search+FAISS: {sum(search_times):.1f}s total ({sum(search_times)/len(search_times):.2f}s/q)")
        print(f"  Overhead:   {sum(other_times):.1f}s total ({sum(other_times)/len(other_times):.2f}s/q)")
    print()

    print("  SESSION-LEVEL METRICS:")
    for k in ks:
        ra = sum(metrics[f"recall_any@{k}"]) / max(len(metrics[f"recall_any@{k}"]), 1)
        nd = sum(metrics[f"ndcg_any@{k}"]) / max(len(metrics[f"ndcg_any@{k}"]), 1)
        print(f"    Recall@{k:2}: {ra:.4f}    NDCG@{k:2}: {nd:.4f}")

    print()
    print("  PER-TYPE BREAKDOWN (recall_any@10):")
    for qtype, vals in sorted(per_type.items()):
        r10 = sum(vals["recall_any@10"]) / max(len(vals["recall_any@10"]), 1)
        nq = len(vals["recall_any@10"])
        print(f"    {qtype:35} R@10={r10:.4f}  (n={nq})")

    print()

    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            for row in results_log:
                f.write(json.dumps(row) + "\n")
        print(f"  Results saved to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alt Memory x LongMemEval Benchmark")
    parser.add_argument("data_file")
    parser.add_argument("--granularity", choices=["session", "turn"], default="session")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--backend", choices=["alt", "fastembed", "chromadb", "chromadb-native"], default="alt",
                        help="Embedding backend (default: alt-memory's sentence-transformers)")
    args = parser.parse_args()

    if not args.out:
        args.out = f"benchmarks/results_alt_{args.granularity}_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"

    run_benchmark(args.data_file, args.granularity, args.limit, args.skip, args.out, args.backend)
