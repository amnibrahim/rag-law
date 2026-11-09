"""Task 4: retrieval smoke test (no LLM, only embeddings + Qdrant).

Runs the test questions against the "chunks" collection and prints the top
verdicts per question, so the results can be read by hand.
"""
import sys

from qdrant_client import models

from embed_index import CHUNKS, connect, embed

TOP_CHUNKS = 40   # chunks fetched per question
TOP_DOCS = 5      # verdicts shown per question

def match(field, value):
    return models.FieldCondition(key=field, match=models.MatchValue(value=value))

CHRONOLOGY = (
    "pada tanggal 3 maret 2023 saya melihat iklan telepon genggam di media sosial dengan harga "
    "rp 4.500.000. penjual bernama budi santoso meminta saya mentransfer uang terlebih dahulu "
    "ke rekening banknya dengan alasan barang akan dikirim melalui jasa ekspedisi. setelah saya "
    "transfer, budi mengirim nomor resi palsu. keesokan harinya nomor penjual tidak aktif dan "
    "akun media sosialnya hilang. saya mencoba menghubungi lagi beberapa kali tetapi tidak ada "
    "jawaban. barang tidak pernah sampai. apa yang bisa dilakukan dan apa hukuman yang pernah "
    "dijatuhkan pada kasus serupa?"
)

# (label, question, filter or None)
QUESTIONS = [
    ("1 charge and sentence", "Berapa hukuman untuk pencurian sepeda motor di pekarangan rumah?", None),
    ("2 article number", "Putusan dengan pasal 378 KUHP tentang penipuan", None),
    ("3 mitigating", "Hal-hal yang meringankan terdakwa penggelapan yang mengembalikan uang korban", None),
    ("4 aggravating", "Hal-hal yang memberatkan pada kasus penganiayaan yang menyebabkan luka berat", None),
    ("5 similar facts", "Terdakwa membawa 5 gram sabu dan mengaku hanya pengguna, bukan pengedar", None),
    ("6 acquittal", "Terdakwa diputus bebas karena unsur pasal tidak terbukti", None),
    ("7 defamation", "Penghinaan lewat unggahan media sosial", None),
    ("8 court and year", "Putusan korupsi di Pengadilan Negeri Bandung tahun 2015",
     models.Filter(must=[match("sub_klasifikasi", "korupsi"), match("year", 2015),
                         match("lembaga_peradilan", "pn-bandung")])),
    ("9 out of scope", "Sengketa wanprestasi utang piutang antara dua perusahaan", None),
    ("10 pasted chronology", CHRONOLOGY, None),
]

def sentence(days):
    """lama_hukuman is in days; show it as years and months (approximate)."""
    if days is None:
        return "-"
    # return f"{days // 365} th {(days % 365) // 30} bln"
    months = round(days / 30.4)
    return f"{months // 12} th {months % 12} bln"

def group_by_doc(points):
    """Keep the best-scoring chunk per verdict, in score order."""
    docs = {}
    for p in points:
        d = p.payload["doc_id"]
        if d not in docs:
            docs[d] = p
    return list(docs.values())

def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    qd = connect()
    qd.create_payload_index(CHUNKS, "lembaga_peradilan", "keyword")  # for the court filter (safe to repeat)
    vectors = embed([q for _, q, _ in QUESTIONS])
    for (label, question, flt), vec in zip(QUESTIONS, vectors):
        res = qd.query_points(CHUNKS, query=vec, query_filter=flt, limit=TOP_CHUNKS, with_payload=True)
        docs = group_by_doc(res.points)[:TOP_DOCS]
        print(f"\n=== {label}: {question[:90]}")
        print(f"    {len(res.points)} chunks -> {len(docs)} verdicts shown")
        for p in docs:
            pl = p.payload
            print(f"  {p.score:.3f} | {pl['case_number'] or '?'} | {pl['lembaga_peradilan']} | {pl['year']} | "
                  f"{pl['sub_klasifikasi']} | {pl['amar']} | {sentence(pl['lama_hukuman'])} | {pl['section']}")
            print("        " + pl["text"].split("\n", 1)[1][:220].replace("\n", " "))

if __name__ == "__main__":
    main()