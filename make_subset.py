import argparse
import csv
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import tiktoken
from lxml import etree

DATA_DIR = Path("data/indo-law/dataset")
INTERNAL = Path("internal")
METRICS_CSV = INTERNAL / "subset_metrics.csv"
IDS_TXT = INTERNAL / "subset_ids.txt"

EMBED_SECTIONS = [
    "riwayat_perkara", "riwayat_penahanan", "riwayat_dakwaan", "riwayat_tuntutan",
    "fakta", "fakta_hukum", "pertimbangan_hukum", "amar_putusan",
]
REQUIRED = ["amar_putusan", "fakta", "riwayat_dakwaan"]

# ---- rules: edit, then rerun (metrics are cached) ----
MAX_PER_CLASS = 300
EXTRA_OUTCOMES = {"bebas": 50, "lepas": 50}  # at least this many of each, if available
EXCLUDE_CLASSES = {"tidak-diketahui"}
MIN_TOKENS = 1000   # guess
MAX_NOISE = 0.15    # guess: share of characters that are not letters, digits or spaces
MAX_DUP = 0.60      # guess: share of text sitting in repeated lines
# TOKEN_CAP = None    # None = 95th percentile of the documents that pass the filters
TOKEN_CAP = 100000  # tokens; about 0.5% of documents are above this
SEED = 42

CHUNK_CHARS, OVERLAP_CHARS, HEADER_TOKENS = 3000, 300, 20
PRICE_PER_M = 0.02
BYTES_PER_VECTOR = 1536 * 4

TEXT_COLS = ("id", "sub_klasifikasi", "amar")
enc = tiktoken.encoding_for_model("text-embedding-3-small")
parser = etree.XMLParser(recover=True, huge_tree=True)
WORD_OR_SPACE = re.compile(r"[\w\s]")


def read_sections(path):
    root = etree.parse(str(path), parser).getroot()
    sections = {}
    for child in root:
        if isinstance(child.tag, str):
            sections[child.tag] = sections.get(child.tag, "") + "".join(child.itertext())
    return dict(root.attrib), sections


def doc_metrics(path):
    attrs, sections = read_sections(path)
    row = {
        "id": path.stem,
        "sub_klasifikasi": attrs.get("sub_klasifikasi", ""),
        "amar": attrs.get("amar", ""),
    }
    stride = CHUNK_CHARS - OVERLAP_CHARS
    texts, tokens, chunks = [], 0, 0
    for s in EMBED_SECTIONS:
        text = sections.get(s, "")
        present = bool(text.strip())
        n_tok = len(enc.encode_ordinary(text)) if present else 0
        row[f"has_{s}"] = int(present)
        row[f"tok_{s}"] = n_tok
        tokens += n_tok
        if present:
            texts.append(text)
            chunks += max(1, math.ceil(len(text) / stride))
    joined = "\n".join(texts)
    lines = [ln.strip() for ln in joined.split("\n") if len(ln.strip()) > 40]
    line_chars = sum(len(ln) for ln in lines)
    unique_chars = sum(len(ln) for ln in set(lines))
    row.update({
        "tokens": tokens,
        "chunks": chunks,
        "noise": len(WORD_OR_SPACE.sub("", joined)) / max(1, len(joined)),
        "dup": 1 - unique_chars / line_chars if line_chars else 0.0,
    })
    return row


def scan_all():
    files = sorted(DATA_DIR.glob("*.xml"))
    rows, errors = [], 0
    for i, path in enumerate(files, 1):
        try:
            rows.append(doc_metrics(path))
        except Exception:
            errors += 1
        if i % 2000 == 0:
            print(f"  scanned {i}/{len(files)}")
    print(f"Scanned {len(files)} files, {errors} errors")
    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def load_metrics():
    rows = []
    with open(METRICS_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                if k not in TEXT_COLS:
                    r[k] = float(v)
            rows.append(r)
    return rows


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def keep_if(label, pool, keep):
    new = [r for r in pool if keep(r)]
    print(f"  {label}: {len(pool) - len(new)} removed, {len(new)} left")
    return new


def spread(values):
    q = statistics.quantiles(values, n=100)
    return f"median {q[49]:.2f}, p90 {q[89]:.2f}, p99 {q[98]:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescan", action="store_true")
    args = ap.parse_args()
    INTERNAL.mkdir(exist_ok=True)

    if args.rescan or not METRICS_CSV.exists():
        rows = scan_all()
    else:
        rows = load_metrics()
        print(f"Loaded cached metrics for {len(rows)} documents (--rescan recomputes)")

    print("\n== filtering ==")
    print(f"  start: {len(rows)}")
    pool = keep_if("missing " + "/".join(REQUIRED), rows,
                   lambda r: all(r[f"has_{s}"] for s in REQUIRED))
    pool = keep_if("excluded class " + ",".join(sorted(EXCLUDE_CLASSES)), pool,
                   lambda r: r["sub_klasifikasi"] not in EXCLUDE_CLASSES)
    admitted = pool

    print("\n  spread over admitted docs (to judge the thresholds):")
    for key in ("tokens", "noise", "dup"):
        print(f"    {key}: {spread([r[key] for r in admitted])}")
    print(f"  under {MIN_TOKENS} tokens: {sum(r['tokens'] < MIN_TOKENS for r in admitted)}, "
          f"noise > {MAX_NOISE}: {sum(r['noise'] > MAX_NOISE for r in admitted)}, "
          f"dup > {MAX_DUP}: {sum(r['dup'] > MAX_DUP for r in admitted)}")

    def is_flagged(r):
        return r["tokens"] < MIN_TOKENS or r["noise"] > MAX_NOISE or r["dup"] > MAX_DUP

    flagged = [r for r in admitted if is_flagged(r)]
    pool = keep_if("quality flags (any of the three)", pool, lambda r: not is_flagged(r))

    tok_list = [r["tokens"] for r in pool]
    for t in (40000, 60000, 100000):
        print(f"  share of docs above {t} tokens: "
              f"{pct(sum(1 for x in tok_list if x > t), len(tok_list))}")
    cap = TOKEN_CAP or statistics.quantiles(tok_list, n=20)[18]
    pool = keep_if(f"over token cap {cap:.0f}", pool, lambda r: r["tokens"] <= cap)

    # ---- sampling ----
    rng = random.Random(SEED)
    by_class = defaultdict(list)
    for r in pool:
        by_class[r["sub_klasifikasi"]].append(r)
    selected = {}
    for cls in sorted(by_class):
        docs = by_class[cls]
        pick = docs if len(docs) <= MAX_PER_CLASS else rng.sample(docs, MAX_PER_CLASS)
        for r in pick:
            selected[r["id"]] = r
    for amar, want in EXTRA_OUTCOMES.items():
        cand = [r for r in pool if r["amar"] == amar]
        have = sum(1 for r in cand if r["id"] in selected)
        extra = [r for r in cand if r["id"] not in selected]
        need = min(want - have, len(extra))
        if need > 0:
            for r in rng.sample(extra, need):
                selected[r["id"]] = r
    sel = list(selected.values())

    total_by_class = Counter(r["sub_klasifikasi"] for r in rows)
    elig_by_class = Counter(r["sub_klasifikasi"] for r in pool)
    sel_by_class = Counter(r["sub_klasifikasi"] for r in sel)
    print("\n== per class: total / eligible / selected ==")
    miss = Counter(r["sub_klasifikasi"] for r in rows
        if not all(r[f"has_{s}"] for s in REQUIRED))
    over = Counter(r["sub_klasifikasi"] for r in admitted if r["tokens"] > cap)

    for cls, tot in total_by_class.most_common():
        # print(f"  {cls}: {tot} / {elig_by_class[cls]} / {sel_by_class[cls]}")
        print(f"  {cls}: {tot} / {elig_by_class[cls]} / {sel_by_class[cls]}"f"  (missing required: {miss[cls]}, over cap: {over[cls]})")
    print("  outcomes in subset:", dict(Counter(r["amar"] for r in sel)))

    tokens = sum(r["tokens"] for r in sel)
    chunks = sum(r["chunks"] for r in sel)
    est = tokens * (1 + OVERLAP_CHARS / CHUNK_CHARS) + chunks * HEADER_TOKENS
    print("\n== subset estimate ==")
    print(f"  docs {len(sel)}, tokens {tokens / 1e6:.1f}M, "
          f"embed tokens incl. overlap and headers {est / 1e6:.1f}M, "
          f"cost ${est / 1e6 * PRICE_PER_M:.2f}")
    print(f"  chunks ~{chunks}, raw vectors ~{chunks * BYTES_PER_VECTOR / 1e9:.2f} GB")

    print("\n== sections in selected docs ==")
    for s in EMBED_SECTIONS:
        have = [r[f"tok_{s}"] for r in sel if r[f"has_{s}"]]
        med = statistics.median(have) if have else 0
        print(f"  {s}: {100 * sum(have) / tokens:.1f}% of tokens, "
              f"present in {pct(len(have), len(sel))}, median {med:.0f} tokens")

    with_p = [r["tok_fakta"] for r in admitted if r["has_pertimbangan_hukum"]] or [0]
    without_p = [r["tok_fakta"] for r in admitted if not r["has_pertimbangan_hukum"]] or [0]
    print("\n== fakta length vs pertimbangan_hukum (admitted docs) ==")
    print(f"  median fakta tokens, pertimbangan_hukum present: {statistics.median(with_p):.0f}")
    print(f"  median fakta tokens, pertimbangan_hukum absent:  {statistics.median(without_p):.0f}")

    print("\n== 10 random flagged documents (read them) ==")
    rng2 = random.Random(SEED + 1)
    for r in rng2.sample(flagged, min(10, len(flagged))):
        reasons = []
        if r["tokens"] < MIN_TOKENS:
            reasons.append(f"tokens={r['tokens']:.0f}")
        if r["noise"] > MAX_NOISE:
            reasons.append(f"noise={r['noise']:.2f}")
        if r["dup"] > MAX_DUP:
            reasons.append(f"dup={r['dup']:.2f}")
        _, sections = read_sections(DATA_DIR / f"{r['id']}.xml")
        text = " ".join(" ".join(sections.get(s, "").split()) for s in EMBED_SECTIONS)
        print(f"  {r['id']} [{', '.join(reasons)}] {text[:200]}")

    IDS_TXT.write_text("\n".join(sorted(selected)) + "\n", encoding="utf-8")
    print(f"\nWrote {len(selected)} ids to {IDS_TXT}")


if __name__ == "__main__":
    main()