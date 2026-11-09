"""Evaluation of RAG Law: run the questions, judge relevance, report metrics.

Run from the project folder:
    uv run evaluate.py --stage run      # questions -> run_llm; saves times, tokens, first-search verdicts
    uv run evaluate.py --stage judge    # an LLM labels every (question, verdict) pair
    (then fill the my_label column in eval/results/check.csv with 0 or 1)
    uv run evaluate.py --stage report   # metrics -> eval/results/report.md
Each stage saves its own file, so a stage can be rerun without the earlier ones.
"""
import argparse
import csv
import json
import math
import os
import random
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_core.callbacks import get_usage_metadata_callback
from openai import OpenAI
from qdrant_client import QdrantClient

from agent_lc import run_llm

load_dotenv()

# ---- settings ----
EVAL_DIR = Path("eval")
RESULTS = EVAL_DIR / "results"
QUESTIONS = EVAL_DIR / "questions.txt"   # lines: id|type|question
K = 5                                    # verdicts per question (Hit@5, MRR@5)
JUDGE_MODEL = "gpt-4o-mini"
MAX_JUDGE_CHARS = 350_000                # about 109k tokens; a longer verdict is cut and flagged
SEED = 42
# USD per 1M tokens of the answering model (gpt-4o-mini). Check OpenAI's pricing page before reporting.
PRICE_IN, PRICE_OUT = 0.15, 0.60

RUBRIC = """Kamu menilai apakah sebuah putusan pidana relevan terhadap sebuah pertanyaan hukum.
Putusan RELEVAN (1) jika jenis tindak pidananya sama dengan pertanyaan DAN fakta kunci yang disebut \
pertanyaan (misalnya objek, cara, atau keadaan) tampak pada putusan. Untuk pertanyaan tentang pasal, \
relevan jika putusan memakai pasal itu. Untuk pertanyaan tentang putusan bebas atau lepas, relevan jika \
hasil putusannya bebas atau lepas dan alasannya sesuai. Selain itu TIDAK RELEVAN (0).
Balas hanya JSON: {"reason": "<satu kalimat>", "relevant": 0 atau 1}. Tulis reason lebih dulu."""


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_questions():
    rows = []
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            qid, qtype, question = line.split("|", 2)
            rows.append((int(qid), qtype, question))
    return rows


def get_qdrant():
    return QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"], timeout=120)


# ---------------- stage 1: run ----------------
def stage_run():
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "run.jsonl"
    done = {r["qid"] for r in read_jsonl(path)}  # questions already run are skipped
    todo = [q for q in load_questions() if q[0] not in done]
    if todo and not done:
        run_llm("Apa hukuman untuk pencurian?")  # warm-up call (connections, clients), not recorded
    with open(path, "a", encoding="utf-8") as f:
        for qid, qtype, question in todo:
            with get_usage_metadata_callback() as cb:  # counts the tokens of every model call inside
                t0 = time.perf_counter()
                out = run_llm(question)
                seconds = time.perf_counter() - t0   # what the user waits: from question to returned answer
            usage = {"input_tokens": sum(u["input_tokens"] for u in cb.usage_metadata.values()),
                     "output_tokens": sum(u["output_tokens"] for u in cb.usage_metadata.values())}
            keys = ("doc_id", "case_number", "court", "year", "sub_klasifikasi", "sentence")
            # verdicts are numbered in the order they are found, so the first K are the first search's
            # (this holds when the first search returns K verdicts, which is almost always)
            row = {"qid": qid, "type": qtype, "question": question, "seconds": round(seconds, 2),
                   "usage": usage, "answer": out["answer"],
                   "cited": [d["doc_id"] for d in out["documents"]],
                   "first_search": [{k: d[k] for k in keys} for d in out["retrieved"][:K]]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  question {qid}: {seconds:.1f} s")


# ---------------- stage 2: judge ----------------
def fetch_verdict(qd, doc_id):
    vid = str(uuid.uuid5(uuid.NAMESPACE_URL, "verdict|" + doc_id))  # same id as in embed_index.py
    points = qd.retrieve("verdicts", ids=[vid], with_payload=True)
    return points[0].payload if points else None


def verdict_text(v):
    """The whole verdict as one text: a header line, then every section under its name."""
    head = (f"Nomor: {v['case_number']} | Pengadilan: {v['lembaga_peradilan']} | Tahun: {v['year']} | "
            f"Jenis perkara: {v['sub_klasifikasi']} | Hasil putusan: {v['amar']}")
    body = "\n".join(f"[{name}]\n{text.strip()}" for name, text in v["sections"].items())
    cut = len(body) > MAX_JUDGE_CHARS
    return head + "\n\n" + body[:MAX_JUDGE_CHARS], cut


def judge_call(client, question, text):
    resp = client.chat.completions.create(
        model=JUDGE_MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": RUBRIC},
                  {"role": "user", "content": f"Pertanyaan: {question}\n\nPutusan:\n{text}"}])
    raw = resp.choices[0].message.content or ""
    try:
        data = json.loads(raw)
        relevant = int(data["relevant"])
        assert relevant in (0, 1)
        return {"relevant": relevant, "reason": str(data.get("reason", ""))}, resp.usage.total_tokens
    except Exception:  # unreadable answer: saved as None and retried on the next run
        return {"relevant": None, "reason": "PARSE ERROR: " + raw[:200]}, resp.usage.total_tokens


def latest_labels():
    """Newest label per (question, verdict); later lines override earlier ones."""
    return {(l["qid"], l["doc_id"]): l for l in read_jsonl(RESULTS / "labels.jsonl")}


def write_check_csv(runs):
    """check.csv: one row per pair in random order. Your my_label entries are kept when it is rewritten."""
    path = RESULTS / "check.csv"
    mine = {}
    if path.exists():
        with open(path, encoding="utf-8-sig", newline="") as f:
            mine = {(int(r["qid"]), r["doc_id"]): r["my_label"] for r in csv.DictReader(f)}
    labels, rows = latest_labels(), []
    for r in runs:
        for rank, d in enumerate(r["first_search"], 1):
            l = labels.get((r["qid"], d["doc_id"]))
            if l and l["relevant"] is not None:
                rows.append({"qid": r["qid"], "rank": rank, "doc_id": d["doc_id"], "case_number": d["case_number"],
                             "question": r["question"], "judge_relevant": l["relevant"], "judge_reason": l["reason"],
                             "my_label": mine.get((r["qid"], d["doc_id"]), "")})
    random.Random(SEED).shuffle(rows)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig so Excel reads it correctly
        w = csv.DictWriter(f, fieldnames=["qid", "rank", "doc_id", "case_number", "question",
                                          "judge_relevant", "judge_reason", "my_label"])
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def stage_judge():
    runs = read_jsonl(RESULTS / "run.jsonl")
    done = {k for k, l in latest_labels().items() if l["relevant"] is not None}
    qd, client, tokens = get_qdrant(), OpenAI(max_retries=6), 0
    with open(RESULTS / "labels.jsonl", "a", encoding="utf-8") as f:
        for r in runs:
            for rank, d in enumerate(r["first_search"], 1):
                if (r["qid"], d["doc_id"]) in done:
                    continue
                v = fetch_verdict(qd, d["doc_id"])
                if v is None:
                    print(f"  verdict {d['doc_id']} not found, skipped")
                    continue
                text, cut = verdict_text(v)
                result, used = judge_call(client, r["question"], text)
                tokens += used
                f.write(json.dumps({"qid": r["qid"], "rank": rank, "doc_id": d["doc_id"], **result,
                                    "truncated": cut, "model": JUDGE_MODEL}, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  question {r['qid']} rank {rank}: {result['relevant']}")
    print(f"  judge tokens this run: {tokens}")
    write_check_csv(runs)


# ---------------- stage 3: report ----------------
def wilson(k, n, z=1.96):
    """95% interval for a share k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def stage_report():
    runs, labels = read_jsonl(RESULTS / "run.jsonl"), latest_labels()
    per_q, missing = [], 0
    for r in runs:
        rels = []
        for d in r["first_search"][:K]:
            l = labels.get((r["qid"], d["doc_id"]))
            if l is None or l["relevant"] is None:
                missing += 1  # not judged: counted as not relevant
                rels.append(0)
            else:
                rels.append(l["relevant"])
        first = next((i for i, x in enumerate(rels, 1) if x), None)
        per_q.append({"type": r["type"], "hit": int(first is not None), "rr": 1 / first if first else 0.0})

    n = len(per_q)
    hits = sum(q["hit"] for q in per_q)
    lo, hi = wilson(hits, n)
    secs = [r["seconds"] for r in runs]
    p50, p95 = np.percentile(secs, [50, 95])
    costs = [r["usage"]["input_tokens"] * PRICE_IN / 1e6 + r["usage"]["output_tokens"] * PRICE_OUT / 1e6 for r in runs]

    lines = [f"# Evaluation report ({n} questions)", "",
             f"- Hit@{K}: {hits}/{n} = {hits / n:.2f} (95% interval {lo:.2f} to {hi:.2f})",
             f"- MRR@{K}: {sum(q['rr'] for q in per_q) / n:.3f}",
             f"- Latency (question to answer, measured on this machine): p50 {p50:.1f} s, p95 {p95:.1f} s, mean {np.mean(secs):.1f} s",
             f"- Cost per query (answering model, tokens x price; query embedding excluded): "
             f"${np.mean(costs):.5f}, about ${np.mean(costs) * 100:.3f} per 100 queries",
             f"- Pairs without a judge label (counted as not relevant): {missing}", "",
             "## By question type", "", "| type | n | Hit@5 | MRR@5 |", "|---|---|---|---|"]
    by_type = defaultdict(list)
    for q in per_q:
        by_type[q["type"]].append(q)
    for t, qs in by_type.items():
        lines.append(f"| {t} | {len(qs)} | {sum(q['hit'] for q in qs) / len(qs):.2f} | {sum(q['rr'] for q in qs) / len(qs):.2f} |")

    path = RESULTS / "check.csv"   # agreement between the judge and your own labels
    if path.exists():
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["my_label"].strip() in ("0", "1")]
        agree = sum(r["my_label"].strip() == str(r["judge_relevant"]) for r in rows)
        alo, ahi = wilson(agree, len(rows))
        lines += ["", "## Judge vs your labels", "",
                  f"- Agreement: {agree}/{len(rows)}" + (f" = {agree / len(rows):.2f} (95% interval {alo:.2f} to {ahi:.2f})" if rows else " (no my_label filled in yet)")]
    report = "\n".join(lines) + "\n"
    (RESULTS / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["run", "judge", "report", "all"], required=True)
    stage = ap.parse_args().stage
    if stage in ("run", "all"):
        stage_run()
    if stage in ("judge", "all"):
        stage_judge()
    if stage in ("report", "all"):
        stage_report()