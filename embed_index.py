"""Task 3: embed the chunks and index them into Qdrant.

Reads  internal/chunks.jsonl and the XML files of the selected verdicts
Writes two Qdrant collections:
  chunks    one point per chunk: vector + payload (metadata and chunk text)
  verdicts  one point per verdict: full original text by section, for the viewer

Safe to rerun: points that already exist in Qdrant are skipped, so an
interrupted run costs nothing extra. Uses a plain loop (no asyncio).
"""
import argparse
import json
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models

from build_chunks import read_doc  # reuse the XML reader

# ---- settings ----
EMBED_MODEL = "text-embedding-3-small"
DIM = 1536
EMBED_BATCH = 128      # chunks per OpenAI request (about 120k tokens)
UPSERT_BATCH = 128     # points per Qdrant request
VERDICT_BATCH = 20     # verdict points are large (up to about 300 KB each)
CHUNKS = "chunks"
VERDICTS = "verdicts"


def connect():
    load_dotenv()
    return QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"], timeout=120)


def embed(texts):
    """Return one vector per text. The OpenAI client retries rate limits itself."""
    client = OpenAI(max_retries=6)
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def ensure_collections(qd, recreate):
    if recreate:
        for name in (CHUNKS, VERDICTS):
            if qd.collection_exists(name):
                qd.delete_collection(name)
    if not qd.collection_exists(CHUNKS):
        qd.create_collection(
            CHUNKS,
            # vectors on disk + int8 copy in RAM: fits the 1 GB free tier
            vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE, on_disk=True),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
            ),
            on_disk_payload=True,
        )
        # indexes for later filters (court, case type, year, one verdict)
        for field, kind in [("doc_id", "keyword"), ("sub_klasifikasi", "keyword"),
                            ("amar", "keyword"), ("year", "integer")]:
            qd.create_payload_index(CHUNKS, field, kind)
    if not qd.collection_exists(VERDICTS):
        # verdicts are looked up by id, so they need no vectors
        qd.create_collection(VERDICTS, vectors_config={}, on_disk_payload=True)


def existing_ids(qd, name):
    """All point ids already stored in a collection (no payload, so it is cheap)."""
    ids, offset = set(), None
    while True:
        points, offset = qd.scroll(name, limit=10000, offset=offset,
                                   with_payload=False, with_vectors=False)
        ids.update(str(p.id) for p in points)
        if offset is None:
            return ids


def verdict_id(doc_id):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "verdict|" + doc_id))


def load_chunks(path, limit_docs):
    """Read chunks.jsonl; with --limit keep only the first N documents."""
    chunks, docs = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            d = c["payload"]["doc_id"]
            if d not in docs:
                if limit_docs and len(docs) >= limit_docs:
                    break
                docs.append(d)
            chunks.append(c)
    return chunks, docs


def index_chunks(qd, chunks):
    done = existing_ids(qd, CHUNKS)
    todo = [c for c in chunks if c["id"] not in done]
    print(f"chunks: {len(chunks)} total, {len(done)} already indexed, {len(todo)} to embed")
    start = time.time()
    for i in range(0, len(todo), EMBED_BATCH):
        batch = todo[i:i + EMBED_BATCH]
        vectors = embed([c["text"] for c in batch])
        qd.upsert(CHUNKS, points=[
            # the chunk text is stored too, so the UI can show the matching snippet
            models.PointStruct(id=c["id"], vector=v, payload={**c["payload"], "text": c["text"]})
            for c, v in zip(batch, vectors)
        ])
        n = i + len(batch)
        if (i // EMBED_BATCH) % 10 == 0 or n == len(todo):
            print(f"  embedded {n}/{len(todo)}  ({time.time() - start:.0f}s)")


def index_verdicts(qd, docs, chunks, data_dir):
    done = existing_ids(qd, VERDICTS)
    meta = {}  # one payload per document, taken from its first chunk
    for c in chunks:
        meta.setdefault(c["payload"]["doc_id"], c["payload"])
    todo = [d for d in docs if verdict_id(d) not in done]
    print(f"verdicts: {len(docs)} total, {len(done)} already indexed, {len(todo)} to add")
    for i in range(0, len(todo), VERDICT_BATCH):
        points = []
        for d in todo[i:i + VERDICT_BATCH]:
            _, sections = read_doc(Path(data_dir) / f"{d}.xml")  # original text, nothing removed
            keep = {k: meta[d][k] for k in ("doc_id", "klasifikasi", "sub_klasifikasi", "amar",
                                            "lama_hukuman", "lembaga_peradilan", "provinsi",
                                            "url", "year", "case_number")}
            points.append(models.PointStruct(id=verdict_id(d), vector={},
                                             payload={**keep, "sections": sections}))
        qd.upsert(VERDICTS, points=points)
        print(f"  verdicts {min(i + VERDICT_BATCH, len(todo))}/{len(todo)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="internal/chunks.jsonl")
    ap.add_argument("--data-dir", default="data/indo-law/dataset")
    ap.add_argument("--limit", type=int, default=0, help="only the first N documents")
    ap.add_argument("--recreate", action="store_true", help="delete and rebuild both collections")
    args = ap.parse_args()

    chunks, docs = load_chunks(args.chunks, args.limit)
    qd = connect()
    ensure_collections(qd, args.recreate)
    index_verdicts(qd, docs, chunks, args.data_dir)
    index_chunks(qd, chunks)

    for name in (CHUNKS, VERDICTS):
        print(f"{name}: {qd.count(name, exact=True).count} points")


if __name__ == "__main__":
    main()