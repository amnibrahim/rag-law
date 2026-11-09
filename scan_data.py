import csv
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import tiktoken
from lxml import etree

DATA_DIR = Path("data/indo-law/dataset")
OUT_CSV = Path("internal/scan_results.csv")

EMBED_SECTIONS = {
    "riwayat_dakwaan", "riwayat_tuntutan", "fakta",
    "fakta_hukum", "pertimbangan_hukum", "amar_putusan",
}
CHUNK_CHARS, OVERLAP_CHARS = 3000, 300
HEADER_TOKENS = 20            # rough size of the per-chunk header
PRICE_PER_M = 0.02            # USD per 1M tokens (text-embedding-3-small)
BYTES_PER_VECTOR = 1536 * 4   # float32

enc = tiktoken.encoding_for_model("text-embedding-3-small")
year_re = re.compile(r"\b(?:19|20)\d{2}\b")
parser = etree.XMLParser(recover=True, huge_tree=True)


def scan_file(path):
    root = etree.parse(str(path), parser).getroot()
    if root is None or root.tag != "putusan":
        raise ValueError("root tag is not <putusan>")
    sections = {}
    for child in root:
        if isinstance(child.tag, str):
            sections[child.tag] = sections.get(child.tag, "") + "".join(child.itertext())
    embed = {k: v for k, v in sections.items() if k in EMBED_SECTIONS}
    stride = CHUNK_CHARS - OVERLAP_CHARS
    year = year_re.search(sections.get("kepala_putusan", ""))
    return {
        "attrs": dict(root.attrib),
        "sections": list(sections),
        "chars_embed": sum(len(v) for v in embed.values()),
        "tokens_embed": sum(len(enc.encode_ordinary(v)) for v in embed.values()),
        "chunks_est": sum(max(1, math.ceil(len(v) / stride)) for v in embed.values()),
        "newlines": sum(v.count("\n") for v in sections.values()),
        "long_lines": sum(
            1 for v in embed.values() for line in v.split("\n") if len(line) > CHUNK_CHARS
        ),
        "year": year.group(0) if year else "",
        "head": sections.get("kepala_putusan", "")[:90].replace("\n", " "),
    }


def pct(n, total):
    return f"{100 * n / total:.1f}%"


def main():
    files = sorted(DATA_DIR.glob("*.xml"))
    total_files = len(files)
    if len(sys.argv) > 1:
        files = files[: int(sys.argv[1])]

    rows, errors = [], []
    key_sets, section_count = Counter(), Counter()
    values = {
        k: Counter()
        for k in ("klasifikasi", "status", "amar", "amar_lainnya",
                  "provinsi", "lembaga_peradilan", "sub_klasifikasi")
    }
    lama_values, examples = [], []

    for i, path in enumerate(files, 1):
        try:
            r = scan_file(path)
        except Exception as e:
            errors.append((path.name, str(e)))
            continue
        a = r["attrs"]
        key_sets[tuple(sorted(a))] += 1
        section_count.update(r["sections"])
        for k in values:
            values[k][a.get(k, "<missing>")] += 1
        if a.get("lama_hukuman", "").isdigit():
            lama_values.append(int(a["lama_hukuman"]))
        if len(examples) < 3:
            examples.append((path.name, r["head"], r["year"]))
        rows.append({
            "id": a.get("id", path.stem),
            "klasifikasi": a.get("klasifikasi", ""),
            "sub_klasifikasi": a.get("sub_klasifikasi", ""),
            "provinsi": a.get("provinsi", ""),
            "lembaga_peradilan": a.get("lembaga_peradilan", ""),
            **{k: r[k] for k in ("chars_embed", "tokens_embed", "chunks_est",
                                 "newlines", "long_lines", "year")},
        })
        if i % 2000 == 0:
            print(f"  scanned {i}/{len(files)}")

    n = len(rows)
    print(f"\nFiles found: {total_files}, scanned: {len(files)}, parsed: {n}, errors: {len(errors)}")
    for name, msg in errors[:5]:
        print("  error:", name, msg)

    OUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n== <putusan> attributes ==")
    for keys, c in key_sets.most_common(5):
        print(f"  {c} files ({pct(c, n)}): {', '.join(keys)}")
    for k, c in values.items():
        if k in ("provinsi", "lembaga_peradilan"):
            print(f"  {k}: {len(c)} distinct values")
        else:
            top = 25 if k == "sub_klasifikasi" else 6
            print(f"  {k}: {len(c)} distinct; top: {c.most_common(top)}")
    if lama_values:
        print(f"  lama_hukuman numeric: {pct(len(lama_values), n)}, "
              f"median {statistics.median(lama_values)}, max {max(lama_values)}")

    print("\n== sections present ==")
    for tag, c in section_count.most_common():
        print(f"  {tag}: {pct(c, n)}")

    toks = [r["tokens_embed"] for r in rows]
    chars = sum(r["chars_embed"] for r in rows)
    print("\n== embedded sections: tokens per document ==")
    print(f"  mean {statistics.mean(toks):.0f}, median {statistics.median(toks):.0f}, "
          f"p90 {statistics.quantiles(toks, n=10)[8]:.0f}, max {max(toks)}")
    print(f"  chars per token: {chars / sum(toks):.2f}")

    print("\n== line structure and year ==")
    print(f"  docs with <5 newlines: {pct(sum(1 for r in rows if r['newlines'] < 5), n)}")
    print(f"  docs with a line over {CHUNK_CHARS} chars: "
          f"{pct(sum(1 for r in rows if r['long_lines'] > 0), n)}")
    print(f"  year found in kepala_putusan: {pct(sum(1 for r in rows if r['year']), n)}")
    for ex in examples:
        print("   ", ex)

    mean_tokens = sum(toks) / n
    mean_chunks = sum(r["chunks_est"] for r in rows) / n
    print(f"\n== estimates ({CHUNK_CHARS}-char chunks, {OVERLAP_CHARS} overlap, "
          f"{HEADER_TOKENS}-token header, 1536 dims) ==")
    print("  docs | embed tokens | cost USD | vectors | raw vectors GB")
    for docs in (3000, total_files // 2, total_files):
        t = docs * (mean_tokens * (1 + OVERLAP_CHARS / CHUNK_CHARS) + mean_chunks * HEADER_TOKENS)
        v = docs * mean_chunks
        print(f"  {docs:>5} | {t / 1e6:>7.1f}M | {t / 1e6 * PRICE_PER_M:>8.2f} | "
              f"{v:>7.0f} | {v * BYTES_PER_VECTOR / 1e9:>6.2f}")


if __name__ == "__main__":
    main()