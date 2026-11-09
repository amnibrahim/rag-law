"""Task 5: the RAG agent (LangChain version, standalone).

run_llm(question, history) -> {"answer": str, "documents": [...], "retrieved": [...]}

create_agent runs one tool, search_verdicts (short queries plus optional filters),
up to about 3 times. The answer cites verdicts as [n]; "documents" holds only the
cited ones, so the list in the UI always matches the text.
Tracing: set LANGSMITH_TRACING=true, LANGSMITH_API_KEY and LANGSMITH_PROJECT in .env.

It covers two model calls (decide to search, then write the answer), 
one embedding call and one Qdrant search

history mechanism:
Turn 1: user asks Q1. The search returns 5 snippets and the model writes answer A1.
Turn 2: the model receives Q1, A1 and user's new Q2. It does not receive Turn 1’s 5 snippets.

Chunks vs verdicts: 
- the database stores chunks, which are pieces of verdicts. 
- A search returns the 40 best-matching chunks. 
- The code then groups them by verdict, keeps the best chunk of each, and keeps the top 5 verdicts. 
- The model sees those 5 snippets, about 500 characters each.

Retreival Mechanism: Parent-document Retrieval:
1. by_doc = {}  # best chunk per verdict (hits arrive sorted by score).
Group the 40 retrieved chunks by verdict: 
2. Qdrant returns them sorted by similarity score, so the first chunk seen for a verdict is its best one; later ones are ignored
setdefault keeps the first). # by_doc.setdefault(doc.metadata["doc_id"], doc)
3. Scores are not summed: a verdict is ranked by its best chunk.
Example, hits in score order (chunkID:docID): 1:A  2:B  3:A  4:C  5:D  6:E  7:F  8:F  9:F  10:F ...
by_doc keeps A(chunk 1), B(2), C(4), D(5), E(6), F(7). [:TOP_DOCS] then keeps A to E.
Chunk 3 is dropped (A is already there), and F is dropped even though it has 4 chunks.

hits = get_store().similarity_search_with_score(query, k=TOP_CHUNKS, filter=build_filter(args))  # Qdrant: 40 best chunks, sorted by score
by_doc = {}
for doc, _ in hits:
    by_doc.setdefault(doc.metadata["doc_id"], doc) # keep the first (best) chunk per verdict
...
for doc_id, doc in list(by_doc.items())[:TOP_DOCS]:  TOP_DOCS = 5: keep the first 5 verdicts
"""
import os
import re
from functools import lru_cache
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.errors import GraphRecursionError
from langsmith import traceable
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

load_dotenv()  # load the keys before any model or client is created

# ---- settings ----
CHUNKS = "chunks"    # Qdrant collection from embed_index.py
MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
TOP_CHUNKS = 40      # chunks fetched per search
TOP_DOCS = 5         # verdicts kept per search
LLM_SNIPPET = 500    # characters of the best chunk shown to the model
UI_SNIPPET = 300     # characters kept for the UI
RECURSION_LIMIT = 9  # model and tool steps: about 3 searches plus the final answer

SYSTEM_PROMPT = """Kamu adalah asisten riset putusan pengadilan pidana Indonesia untuk praktisi hukum \
yang menjawab pertanyaan hukum dari masyarakat. Basis datamu hanya berisi sebagian putusan PIDANA \
(tidak ada perkara perdata) dan bukan seluruh putusan di Indonesia.

Cara kerja:
1. Untuk pertanyaan tentang kasus atau hukuman, panggil search_verdicts sebelum menjawab.
2. Tulis query pendek (3-12 kata) tentang tindak pidana dan perbuatan utamanya, tanpa nama orang, \
tanggal, nomor rekening, atau nominal uang. Jika pengguna menulis kronologi panjang, ringkas menjadi \
2-3 pencarian yang berbeda.
3. Isi amar hanya jika pengguna menanyakan putusan bebas/lepas; isi year dan court hanya jika disebutkan. \
Format court: pn-<kota> huruf kecil, contoh pn-bandung.
4. Skor pencarian tidak menunjukkan relevansi. Nilai sendiri apakah isi hasil benar-benar cocok. \
Jika tidak ada yang cocok (misalnya sengketa perdata), katakan terus terang dan jangan beri kutipan.

Cara menjawab:
- Bahasa Indonesia, kecuali pengguna memakai bahasa lain. Singkat dan langsung.
- Jelaskan apa yang tampak dari putusan-putusan itu: pasal atau tindak pidana, fakta yang mirip, \
hukuman. Jangan menyimpulkan hukum, memprediksi hasil perkara, atau memberi nasihat hukum.
- Setiap kali menyebut sebuah putusan, tulis nomor hasilnya, contoh [2]. Sebut semua putusan yang \
relevan (paling banyak 5 putusan), tetapi jangan masukkan yang tidak cocok. Jangan mengarang nomor. 
- Hanya gunakan hukuman dari putusan yang kamu kutip. Jangan menyebut rentang atau angka dari hasil lain.
- Katakan bahwa hasil hanya contoh dari sebagian putusan, bukan gambaran lengkap."""


class SearchVerdicts(BaseModel):
    """Arguments of the search tool (the model fills these in)."""

    query: str = Field(description="Query pendek tentang tindak pidana dan perbuatan utama, tanpa nama, tanggal, nominal.")
    amar: Optional[Literal["pidana", "bebas", "lepas"]] = Field(None, description="Filter hasil putusan. Hanya untuk pertanyaan tentang bebas/lepas.")
    year: Optional[int] = Field(None, description="Filter tahun putusan, hanya jika disebutkan pengguna.")
    court: Optional[str] = Field(None, description="Filter pengadilan, format pn-<kota> huruf kecil, mis. pn-bandung.")


class FlatQdrantVectorStore(QdrantVectorStore):
    """Our payload is flat (doc_id, year, ... at the top level), so all of it becomes Document.metadata.
    Overrides a private method of langchain-qdrant (checked on version 1.1.0)."""

    @classmethod
    def _document_from_point(cls, scored_point, collection_name, content_payload_key, metadata_payload_key):
        payload = dict(scored_point.payload)
        return Document(page_content=payload.pop(content_payload_key, ""), metadata=payload)


@lru_cache(maxsize=1)
def get_store():
    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"], timeout=120)
    return FlatQdrantVectorStore(
        client=client, collection_name=CHUNKS,
        embedding=OpenAIEmbeddings(model=EMBED_MODEL),
        content_payload_key="text",  # the chunk text is stored under "text"
    )


def format_sentence(amar, days):
    """lama_hukuman is stored in days; show it as years and months (approximate)."""
    if amar in ("bebas", "lepas"):
        return amar
    if not days:
        return "-"
    months = round(days / 30.4)
    return f"{months // 12} th {months % 12} bln"


def build_filter(args):
    must = []
    for field, value in (("amar", args.amar), ("year", args.year), ("lembaga_peradilan", args.court)):
        if value is not None:
            must.append(models.FieldCondition(key=field, match=models.MatchValue(value=value)))
    return models.Filter(must=must) if must else None


def history_messages(history):
    """Earlier turns as plain text. Old [n] citations are removed because numbers restart each turn."""
    msgs = []
    for m in history or []:
        text = re.sub(r"\s*\[\d+\]", "", m["content"])
        msgs.append(HumanMessage(text) if m["role"] == "user" else AIMessage(text))
    return msgs


def make_search_tool(found):
    """Build the tool for one run; `found` collects the verdicts so [n] numbers stay unique."""

    @tool(args_schema=SearchVerdicts, response_format="content_and_artifact")
    def search_verdicts(query, amar=None, year=None, court=None):
        """Cari putusan pidana yang mirip dengan sebuah uraian perbuatan atau pertanyaan hukum."""
        args = SearchVerdicts(query=query, amar=amar, year=year, court=court)
        hits = get_store().similarity_search_with_score(query, k=TOP_CHUNKS, filter=build_filter(args))
        by_doc = {}  # best chunk per verdict (hits arrive sorted by score)
        for doc, _ in hits:
            by_doc.setdefault(doc.metadata["doc_id"], doc)
        lines = []
        for doc_id, doc in list(by_doc.items())[:TOP_DOCS]:
            m = doc.metadata
            snippet = doc.page_content.split("\n", 1)[-1].replace("\n", " ")
            known = next((n for n, e in found.items() if e["doc_id"] == doc_id), None)
            n = known or len(found) + 1  # a verdict keeps its number across searches
            found[n] = {
                "n": n, "doc_id": doc_id, "case_number": m["case_number"],
                "court": m["lembaga_peradilan"], "year": m["year"],
                "sub_klasifikasi": m["sub_klasifikasi"], "amar": m["amar"],
                "sentence": format_sentence(m["amar"], m["lama_hukuman"]),
                "section": m["section"], "snippet": snippet[:UI_SNIPPET], "url": m["url"],
            }
            e = found[n]
            lines.append(f"[{n}] {e['case_number'] or '?'} | {e['court']} | {e['year']} | {e['sub_klasifikasi']} | "
                         f"putusan: {e['sentence']} | bagian: {e['section']}\n{snippet[:LLM_SNIPPET]}")
        text = "\n\n".join(lines) or "Tidak ada hasil untuk pencarian ini."
        return text, list(found.values())  # (text for the model, artifact for us)

    return search_verdicts


@traceable(name="run_llm", run_type="chain")  # one parent trace per question in LangSmith
def run_llm(question, history=None):
    found = {}
    agent = create_agent(ChatOpenAI(model=MODEL, temperature=0),
                         tools=[make_search_tool(found)], system_prompt=SYSTEM_PROMPT)
    try:
        result = agent.invoke({"messages": [*history_messages(history), HumanMessage(question)]},
                              config={"recursion_limit": RECURSION_LIMIT})
        answer = result["messages"][-1].content
    except GraphRecursionError:  # too many search rounds
        answer = "Pencarian terlalu panjang dan dihentikan. Coba pertanyaan yang lebih spesifik."

    cited = []
    for n in map(int, re.findall(r"\[(\d+)\]", answer)):
        if n in found and n not in cited:
            cited.append(n)
    return {"answer": answer,
            "documents": [found[n] for n in cited],  # only what the answer cites
            "retrieved": list(found.values())}       # everything the searches returned


if __name__ == "__main__":
    # quick try from the terminal: uv run agent_lc.py "your question"
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = run_llm(" ".join(sys.argv[1:]) or "Apa hukuman untuk penipuan jual beli online?")
    print(out["answer"], "\n")
    for d in out["documents"]:
    # for d in out["retrieved"]:
        print(f"[{d['n']}] {d['case_number']} | {d['court']} | {d['year']} | {d['sub_klasifikasi']} | "
              f"{d['sentence']} | bagian: {d['section']}\n    {d['snippet']}")
    print(f"({len(out['documents'])} cited of {len(out['retrieved'])} retrieved)")