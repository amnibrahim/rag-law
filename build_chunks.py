"""Task 2: parse the selected verdicts and build chunks (no API calls).

Reads  internal/subset_ids.txt
Writes internal/chunks.jsonl  (one JSON object per line: id, text, payload)

Each chunk's "text" is what will be embedded (header + body).
The payload is stored next to the vector for the UI, filters and statistics.
"""
import argparse
import json
import re
import statistics
import uuid
from collections import Counter
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from lxml import etree

# ---- settings: edit and rerun ----
EMBED_SECTIONS = [
    "riwayat_perkara", "riwayat_penahanan", "riwayat_dakwaan", "riwayat_tuntutan",
    "fakta", "fakta_hukum", "pertimbangan_hukum", "amar_putusan",
]
CHUNK_CHARS = 3000
OVERLAP_CHARS = 300
MIN_SECTION_CHARS = 500   # shorter sections are merged into the next one (about 150 tokens)
DROP_DUP_LINES = True     # drop exact-duplicate lines within a document
DUP_MIN_LEN = 40          # only lines at least this long count as duplicates
ADD_CASE_CARD = True      # one short summary chunk per document
CARD_AMAR_CHARS = 400
PRICE_PER_M = 0.02        # USD per 1M tokens, text-embedding-3-small
CHARS_PER_TOKEN = 3.22    # measured in the data scan

# Only clear typos and duplicates. Review the rest of the class list before adding more.
LABEL_MERGE = {
    "pemerasaan-dan-pengancaman": "pemerasan-dan-pengancaman",
    "pidsus-pra-peradilan": "pra-peradilan",
    "uang-palsu": "pemalsuan-uang",
}

CASE_NO_RE = re.compile(r"putusan\s+(?:nomor|no|n\s*o)\s+(.+?)\s+demi keadilan")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

parser = etree.XMLParser(recover=True, huge_tree=True)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_CHARS,
    chunk_overlap=OVERLAP_CHARS,
    separators=["\n\n", "\n", " ", ""],
)


def read_doc(path):
    root = etree.parse(str(path), parser).getroot()
    sections = {}
    for child in root:
        if isinstance(child.tag, str):
            sections[child.tag] = sections.get(child.tag, "") + "".join(child.itertext())
    return dict(root.attrib), sections


def drop_dup_lines(sections, seen):
    out, dropped = [], 0
    for name, text in sections:
        kept = []
        for line in text.split("\n"):
            s = " ".join(line.split())
            if not s:
                continue
            if len(s) >= DUP_MIN_LEN:
                if s in seen:
                    dropped += len(s)
                    continue
                seen.add(s)
            kept.append(s)
        out.append((name, "\n".join(kept)))
    return out, dropped


def case_info(kepala):
    m = CASE_NO_RE.search(kepala)
    case_no = " ".join(m.group(1).split()) if m else ""
    if len(case_no) > 60:
        case_no = ""
    y = YEAR_RE.search(case_no) or YEAR_RE.search(kepala)
    return case_no, int(y.group(0)) if y else None


def make_header(attrs, sub, year, section):
    parts = [
        "Putusan " + attrs.get("lembaga_peradilan", "").replace("-", " "),
        attrs.get("klasifikasi", "").replace("-", " "),
        sub.replace("-", " "),
    ]
    if year:
        parts.append(str(year))
    parts.append("bagian: " + section.replace("_", " "))
    return " | ".join(parts)


def chunk_id(doc_id, section, i):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}|{section}|{i}"))


def build_doc(path):
    attrs, sections = read_doc(path)
    doc_id = attrs.get("id", path.stem)
    raw_sub = attrs.get("sub_klasifikasi", "")
    sub = LABEL_MERGE.get(raw_sub, raw_sub)
    case_no, year = case_info(sections.get("kepala_putusan", ""))
    try:
        lama = int(attrs.get("lama_hukuman", ""))
    except ValueError:
        lama = None

    embed = [(n, sections[n]) for n in sections if n in EMBED_SECTIONS]
    chars_before = sum(len(t) for _, t in embed)
    dropped = 0
    if DROP_DUP_LINES:
        embed, dropped = drop_dup_lines(embed, set())
    embed = [(n, t) for n, t in embed if t.strip()]

    base = {
        "doc_id": doc_id,
        "klasifikasi": attrs.get("klasifikasi", ""),
        "sub_klasifikasi": sub,
        "sub_klasifikasi_raw": raw_sub,
        "amar": attrs.get("amar", ""),
        "amar_lainnya": attrs.get("amar_lainnya", ""),
        "lama_hukuman": lama,
        "lembaga_peradilan": attrs.get("lembaga_peradilan", ""),
        "provinsi": attrs.get("provinsi", ""),
        "url": attrs.get("url", ""),
        "year": year,
        "case_number": case_no,
    }

    # merge short sections into the next one
    units, pending, pending_names = [], "", []
    for idx, (name, text) in enumerate(embed):
        text = pending + "\n" + text if pending else text
        names = pending_names + [name]
        is_last = idx == len(embed) - 1
        if len(text) < MIN_SECTION_CHARS and not is_last:
            pending, pending_names = text, names
            continue
        units.append((name, names, text))
        pending, pending_names = "", []

    chunks = []
    if ADD_CASE_CARD:
        amar_text = " ".join(sections.get("amar_putusan", "").split())[:CARD_AMAR_CHARS]
        card = (
            f"Nomor putusan: {case_no or 'tidak diketahui'}. "
            f"Pengadilan: {attrs.get('lembaga_peradilan', '').replace('-', ' ')}. "
            f"Jenis perkara: {attrs.get('klasifikasi', '').replace('-', ' ')}, "
            f"{sub.replace('-', ' ')}. Amar: {attrs.get('amar', '')}.\n{amar_text}"
        )
        header = make_header(attrs, sub, year, "ringkasan")
        chunks.append({
            "id": chunk_id(doc_id, "case_card", 0),
            "text": header + "\n" + card,
            "payload": {**base, "section": "case_card", "section_parts": [], "chunk_index": 0},
        })
    for label, names, text in units:
        header = make_header(attrs, sub, year, label)
        pieces = splitter.split_text(text)
        for i, piece in enumerate(pieces):
            chunks.append({
                "id": chunk_id(doc_id, label, i),
                "text": header + "\n" + piece,
                "payload": {**base, "section": label, "section_parts": names,
                            "chunk_index": i, "n_chunks_in_section": len(pieces)},
            })

    fakta_tail = ""
    if "pertimbangan_hukum" not in sections and "fakta_hukum" not in sections:
        fakta_tail = " ".join(sections.get("fakta", "").split())[-300:]
    return chunks, {
        "chars_before": chars_before, "dropped": dropped, "year": year,
        "case_no": case_no, "relabelled": raw_sub != sub, "fakta_tail": fakta_tail,
        "doc_id": doc_id,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="internal/subset_ids.txt")
    ap.add_argument("--data-dir", default="data/indo-law/dataset")
    ap.add_argument("--out", default="internal/chunks.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="only the first N documents")
    ap.add_argument("--skip-tokens", action="store_true", help="estimate tokens from characters")
    args = ap.parse_args()

    ids = [x.strip() for x in Path(args.ids).read_text(encoding="utf-8").split() if x.strip()]
    if args.limit:
        ids = ids[: args.limit]
    data_dir, out_path = Path(args.data_dir), Path(args.out)
    out_path.parent.mkdir(exist_ok=True)

    enc = None
    if not args.skip_tokens:
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model("text-embedding-3-small")
        except Exception as e:  # tiktoken may need to download its vocabulary
            print("tiktoken unavailable, estimating tokens from characters:", e)

    n_chunks = total_tokens = total_chars = 0
    chunk_chars, per_doc, short_chunks = [], [], 0
    stats = Counter()
    samples, tails, errors = [], [], []
    rng_seen = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for i, doc_id in enumerate(ids, 1):
            try:
                chunks, info = build_doc(data_dir / f"{doc_id}.xml")
            except Exception as e:
                errors.append((doc_id, str(e)))
                continue
            per_doc.append(len(chunks))
            stats["chars_before"] += info["chars_before"]
            stats["dropped"] += info["dropped"]
            stats["year_found"] += info["year"] is not None
            stats["case_no_found"] += bool(info["case_no"])
            stats["relabelled"] += info["relabelled"]
            if info["fakta_tail"] and len(tails) < 3:
                tails.append((info["doc_id"], info["fakta_tail"]))
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
                n = len(c["text"])
                chunk_chars.append(n)
                total_chars += n
                short_chunks += n < 500
                total_tokens += len(enc.encode_ordinary(c["text"])) if enc else 0
                rng_seen += 1
                if len(samples) < 3 and rng_seen % 997 == 1:
                    samples.append(c)
                elif rng_seen % 997 == 1:
                    samples[rng_seen // 997 % 3] = c
            n_chunks += len(chunks)
            if i % 500 == 0:
                print(f"  processed {i}/{len(ids)} documents, {n_chunks} chunks")

    n_docs = len(per_doc)
    if not enc:
        total_tokens = int(total_chars / CHARS_PER_TOKEN)
    print(f"\nDocuments processed: {n_docs} of {len(ids)}, errors: {len(errors)}")
    for d, e in errors[:5]:
        print("  error:", d, e)
    if not n_docs:
        return
    print(f"Chunks: {n_chunks} (per document: median {statistics.median(per_doc):.0f}, "
          f"max {max(per_doc)})")
    q = statistics.quantiles(chunk_chars, n=100)
    print(f"Chunk size in characters (header included): median {q[49]:.0f}, p90 {q[89]:.0f}, "
          f"max {max(chunk_chars)}; under 500 chars: {100 * short_chunks / n_chunks:.1f}%")
    print(f"Tokens to embed: {total_tokens / 1e6:.1f}M"
          f"{'' if enc else ' (estimated from characters)'}, "
          f"cost about ${total_tokens / 1e6 * PRICE_PER_M:.2f}")
    if stats["chars_before"]:
        print(f"Duplicate lines dropped: {100 * stats['dropped'] / stats['chars_before']:.1f}% "
              f"of embedded-section characters")
    print(f"Year found: {100 * stats['year_found'] / n_docs:.1f}%, "
          f"case number found: {100 * stats['case_no_found'] / n_docs:.1f}%, "
          f"labels merged: {stats['relabelled']} documents")
    print(f"Written to {out_path}")

    print("\n== sample chunks (check header, boundaries, readability) ==")
    for c in samples:
        p = c["payload"]
        print(f"\n[{p['doc_id']} | {p['section']} #{p['chunk_index']}]")
        print(c["text"][:350].replace("\n", " / "), "...")
        print("   ...", c["text"][-150:].replace("\n", " / "))
    print("\n== end of fakta for documents without pertimbangan_hukum and fakta_hukum ==")
    for d, t in tails:
        print(f"\n[{d}] ...{t}")


if __name__ == "__main__":
    main()