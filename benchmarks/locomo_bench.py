#!/usr/bin/env python3
"""
Alt Memory x LoCoMo Benchmark

Uses alt-memory's embedder + raw FAISS (in-memory, no disk I/O)
for pure retrieval recall measurement on LoCoMo (1,986 QA pairs).

LoCoMo data: https://github.com/snap-research/locomo.git
"""

import json
import re
import sys
import string
import time
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss
import numpy as np


def _make_alt_embedder():
    from alt_memory.backends.embedder import get_embedder
    return get_embedder(model="sentence")


def _compress(texts):
    from alt_memory.dialect import aaak_compress
    return [aaak_compress(t) for t in texts]


def _decompress(text):
    from alt_memory.dialect import aaak_decompress
    return aaak_decompress(text)


CATEGORIES = {
    1: "Single-hop",
    2: "Temporal",
    3: "Temporal-inference",
    4: "Open-domain",
    5: "Adversarial",
}


def normalize_answer(s):
    s = s.replace(",", "")
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    s = " ".join(s.split())
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return s.lower().strip()


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def load_conversation_sessions(conversation, session_summaries=None):
    sessions = []
    session_num = 1
    while True:
        key = f"session_{session_num}"
        date_key = f"session_{session_num}_date_time"
        if key not in conversation:
            break
        dialogs = conversation[key]
        date = conversation.get(date_key, "")
        summary = ""
        if session_summaries:
            summary = session_summaries.get(f"session_{session_num}_summary", "")
        sessions.append({
            "session_num": session_num,
            "date": date,
            "dialogs": dialogs,
            "summary": summary,
        })
        session_num += 1
    return sessions


def build_corpus_from_sessions(sessions, granularity="dialog"):
    corpus = []
    corpus_ids = []
    corpus_timestamps = []

    for sess in sessions:
        if granularity == "session":
            texts = []
            for d in sess["dialogs"]:
                speaker = d.get("speaker", "?")
                text = d.get("text", "")
                texts.append(f'{speaker} said, "{text}"')
            doc = "\n".join(texts)
            corpus.append(doc)
            corpus_ids.append(f"session_{sess['session_num']}")
            corpus_timestamps.append(sess["date"])
        else:
            for d in sess["dialogs"]:
                dia_id = d.get("dia_id", f"D{sess['session_num']}:?")
                speaker = d.get("speaker", "?")
                text = d.get("text", "")
                doc = f'{speaker} said, "{text}"'
                corpus.append(doc)
                corpus_ids.append(dia_id)
                corpus_timestamps.append(sess["date"])

    return corpus, corpus_ids, corpus_timestamps


def compute_retrieval_recall(retrieved_ids, evidence_ids):
    if not evidence_ids:
        return 1.0
    found = sum(1 for eid in evidence_ids if eid in retrieved_ids)
    return found / len(evidence_ids)


def evidence_to_dialog_ids(evidence):
    return set(evidence)


def evidence_to_session_ids(evidence):
    sessions = set()
    for eid in evidence:
        match = re.match(r"D(\d+):", eid)
        if match:
            sessions.add(f"session_{match.group(1)}")
    return sessions


STOP_WORDS = {
    "what", "when", "where", "who", "how", "which",
    "did", "do", "was", "were", "have", "has", "had",
    "is", "are", "the", "a", "an", "my", "me", "i",
    "you", "your", "their", "it", "its", "in", "on",
    "at", "to", "for", "of", "with", "by", "from",
    "ago", "last", "that", "this", "there", "about",
    "get", "got", "give", "gave", "buy", "bought",
    "made", "make", "said",
}

NOT_NAMES = {
    "What", "When", "Where", "Who", "How", "Which",
    "Did", "Do", "Was", "Were", "Have", "Has", "Had",
    "Is", "Are", "The", "My", "Our", "Their",
    "Can", "Could", "Would", "Should", "Will", "Shall",
    "May", "Might", "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "June",
    "July", "August", "September", "October",
    "November", "December",
    "In", "On", "At", "For", "To", "Of", "With", "By",
    "From", "And", "But", "I", "It", "Its", "This",
    "That", "These", "Those", "Previously", "Recently",
    "Also", "Just", "Very", "More", "Said", "Speaker",
    "Person", "Time", "Date", "Year", "Day",
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


def _quoted_phrases(text):
    phrases = []
    for pat in [r"'([^']{3,60})'", r'"([^"]{3,60})"']:
        phrases.extend(re.findall(pat, text))
    return [p.strip() for p in phrases if len(p.strip()) >= 3]


def _quoted_boost(phrases, doc_text):
    if not phrases:
        return 0.0
    doc_lower = doc_text.lower()
    hits = sum(1 for p in phrases if p.lower() in doc_lower)
    return min(hits / len(phrases), 1.0)


def _person_names(text):
    words = re.findall(r"\b[A-Z][a-z]{2,15}\b", text)
    return list(set(w for w in words if w not in NOT_NAMES))


def _name_boost(names, doc_text):
    if not names:
        return 0.0
    doc_lower = doc_text.lower()
    hits = sum(1 for n in names if n.lower() in doc_lower)
    return min(hits / len(names), 1.0)


def run_benchmark(
    data_file,
    top_k=10,
    mode="raw",
    limit=0,
    granularity="dialog",
    out_file=None,
):
    with open(data_file, encoding="utf-8") as f:
        data = json.load(f)

    if limit > 0:
        data = data[:limit]

    print(f"\n{'=' * 60}")
    print("  Alt Memory x LoCoMo Benchmark")
    print(f"{'=' * 60}")
    print(f"  Data:        {Path(data_file).name}")
    print(f"  Conversations: {len(data)}")
    print(f"  Top-k:       {top_k}")
    print(f"  Mode:        {mode}")
    print(f"  Granularity: {granularity}")
    print("-" * 60 + "\n")

    embedder = _make_alt_embedder()
    _ = embedder.embed(["warmup"])
    dim = embedder.dimension

    all_recall = []
    per_category = defaultdict(list)
    results_log = []
    total_qa = 0
    total_conv_time = 0.0

    start_time = datetime.now()

    for conv_idx, sample in enumerate(data):
        sample_id = sample.get("sample_id", f"conv-{conv_idx}")
        conversation = sample["conversation"]
        qa_pairs = sample["qa"]
        session_summaries = sample.get("session_summary", {})
        sessions = load_conversation_sessions(conversation, session_summaries)
        corpus, corpus_ids, _ = build_corpus_from_sessions(sessions, granularity=granularity)

        if mode == "aaak" and corpus:
            corpus = _compress(corpus)

        docs_for_keyword = corpus
        if mode == "aaak":
            docs_for_keyword = [_decompress(d) for d in corpus]

        print(
            f"  [{conv_idx + 1}/{len(data)}] {sample_id}: "
            f"{len(sessions)} sessions, {len(corpus)} docs, {len(qa_pairs)} questions"
        )

        if not corpus:
            continue

        t0 = time.time()
        embeddings = embedder.embed(corpus)
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        t1 = time.time()
        total_conv_time += t1 - t0

        for qa in qa_pairs:
            question = qa["question"]
            answer = qa.get("answer", qa.get("adversarial_answer", ""))
            category = qa["category"]
            evidence = qa.get("evidence", [])

            names = _person_names(question) if mode in ("hybrid", "rooms") else []
            name_words = {n.lower() for n in names}
            all_kws = _kw(question) if mode in ("hybrid", "rooms") else []
            predicate_kws = [w for w in all_kws if w not in name_words]
            quoted = _quoted_phrases(question) if mode in ("hybrid", "rooms") else []

            if mode == "rooms":
                room_scores = []
                for sess in sessions:
                    summary = sess.get("summary", "")
                    overlap = (
                        _kw_overlap(predicate_kws, summary)
                        if (summary and predicate_kws)
                        else 0.0
                    )
                    rid = f"session_{sess['session_num']}" if granularity == "session" else ""
                    room_scores.append((overlap, rid if rid else corpus_ids[sess['session_num'] - 1]))
                room_scores.sort(reverse=True)
                n_rooms = max(top_k, len(sessions) // 3)
                top_room_ids = set(sid for _, sid in room_scores[:n_rooms])

                filtered_indices = [i for i, cid in enumerate(corpus_ids) if cid in top_room_ids]
                if not filtered_indices:
                    filtered_indices = list(range(len(corpus)))
                n_in_rooms = min(top_k * 2, len(filtered_indices))
                filtered_embeddings = embeddings[filtered_indices]
                filtered_corpus_ids = [corpus_ids[i] for i in filtered_indices]
                filtered_corpus = [corpus[i] for i in filtered_indices]
                filtered_docs_for_kw = [docs_for_keyword[i] for i in filtered_indices]

                sub_index = faiss.IndexFlatIP(dim)
                sub_embeddings_t = np.asarray([embeddings[i] for i in filtered_indices], dtype=np.float32)
                faiss.normalize_L2(sub_embeddings_t)
                sub_index.add(sub_embeddings_t)

                query_vec = embedder.embed([question])
                query_vec = np.asarray(query_vec, dtype=np.float32)
                faiss.normalize_L2(query_vec)
                distances, indices_arr = sub_index.search(query_vec, n_in_rooms)
                raw_indices = indices_arr[0].tolist()
                raw_distances = distances[0]

                scored = []
                for i, idx in enumerate(raw_indices):
                    if idx == -1:
                        break
                    doc = filtered_docs_for_kw[idx]
                    dist = raw_distances[i]
                    pred_overlap = _kw_overlap(predicate_kws, doc)
                    fused = dist + 0.30 * pred_overlap
                    q_boost = _quoted_boost(quoted, doc)
                    if q_boost > 0:
                        fused += 0.10 * q_boost
                    n_boost = _name_boost(names, doc)
                    if n_boost > 0:
                        fused += 0.05 * n_boost
                    scored.append((fused, filtered_corpus_ids[idx], doc))
                scored.sort(key=lambda x: x[0], reverse=True)
                retrieved_ids = [x[1] for x in scored[:top_k]]
            else:
                n_retrieve = min(top_k * 3 if mode == "hybrid" else top_k, len(corpus))
                query_vec = embedder.embed([question])
                query_vec = np.asarray(query_vec, dtype=np.float32)
                faiss.normalize_L2(query_vec)
                distances, indices_arr = index.search(query_vec, n_retrieve)
                raw_indices = indices_arr[0].tolist()

                if mode == "hybrid":
                    scored = []
                    for i, idx in enumerate(raw_indices):
                        if idx == -1:
                            break
                        doc = docs_for_keyword[idx]
                        dist = distances[0][i]
                        pred_overlap = _kw_overlap(predicate_kws, doc)
                        fused = dist + 0.30 * pred_overlap
                        q_boost = _quoted_boost(quoted, doc)
                        if q_boost > 0:
                            fused += 0.10 * q_boost
                        n_boost = _name_boost(names, doc)
                        if n_boost > 0:
                            fused += 0.05 * n_boost
                        scored.append((fused, corpus_ids[idx], doc))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    retrieved_ids = [x[1] for x in scored[:top_k]]
                else:
                    retrieved_ids = [corpus_ids[i] for i in raw_indices[:top_k] if i != -1]

            if granularity == "dialog":
                evidence_set = evidence_to_dialog_ids(evidence)
            else:
                evidence_set = evidence_to_session_ids(evidence)

            recall = compute_retrieval_recall(retrieved_ids, evidence_set)
            all_recall.append(recall)
            per_category[category].append(recall)
            total_qa += 1

            results_log.append({
                "sample_id": sample_id,
                "question": question,
                "answer": answer,
                "category": category,
                "evidence": evidence,
                "retrieved_ids": retrieved_ids,
                "recall": recall,
            })

    elapsed = (datetime.now() - start_time).total_seconds()
    avg_recall = sum(all_recall) / len(all_recall) if all_recall else 0

    print(f"\n{'=' * 60}")
    print(f"  RESULTS — Alt Memory ({mode}, {granularity}, top-{top_k})")
    print(f"{'=' * 60}")
    print(f"  Time:        {elapsed:.1f}s ({elapsed / max(total_qa, 1):.2f}s per q)")
    print(f"  Embed/build: {total_conv_time:.1f}s")
    print(f"  Questions:   {total_qa}")
    print(f"  Avg Recall:  {avg_recall:.3f}")

    print("\n  PER-CATEGORY RECALL:")
    for cat in sorted(per_category.keys()):
        vals = per_category[cat]
        avg = sum(vals) / len(vals)
        name = CATEGORIES.get(cat, f"Cat-{cat}")
        print(f"    {name:25} R={avg:.3f}  (n={len(vals)})")

    perfect = sum(1 for r in all_recall if r >= 1.0)
    partial = sum(1 for r in all_recall if 0 < r < 1.0)
    zero = sum(1 for r in all_recall if r == 0)
    print("\n  RECALL DISTRIBUTION:")
    print(f"    Perfect (1.0):  {perfect:4} ({perfect / len(all_recall) * 100:.1f}%)")
    print(f"    Partial (0-1):  {partial:4} ({partial / len(all_recall) * 100:.1f}%)")
    print(f"    Zero (0.0):     {zero:4} ({zero / len(all_recall) * 100:.1f}%)")
    print(f"\n{'=' * 60}\n")

    if out_file:
        with open(out_file, "w") as f:
            json.dump(results_log, f, indent=2)
        print(f"  Results saved to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alt Memory x LoCoMo Benchmark")
    parser.add_argument("data_file", help="Path to locomo10.json")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k retrieval (default: 50)")
    parser.add_argument(
        "--mode", choices=["raw", "aaak", "hybrid", "rooms"], default="raw",
        help="Retrieval mode",
    )
    parser.add_argument(
        "--granularity", choices=["dialog", "session"], default="session",
        help="Corpus granularity",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit to N conversations")
    parser.add_argument("--out", default=None, help="Output JSON file path")
    args = parser.parse_args()

    if not args.out:
        args.out = (
            f"benchmarks/results_alt_locomo_{args.mode}"
            f"_{args.granularity}_top{args.top_k}"
            f"_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        )

    run_benchmark(
        args.data_file, args.top_k, args.mode, args.limit, args.granularity, args.out,
    )
