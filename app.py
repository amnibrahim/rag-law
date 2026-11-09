"""Task 6: Streamlit UI for RAG Law.

Run:  uv run streamlit run app.py
Chat with the agent; every verdict the answer cites gets a card with a button
that opens the original verdict in a wide dialog.
"""
import html
import os
import re
import uuid

from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from agent_lc import run_llm

load_dotenv()

# ---- settings ----
VERDICTS = "verdicts"   # Qdrant collection with the full verdict text (from embed_index.py)
MAX_HISTORY = 12        # last 12 messages (6 turns) are sent to the model
MAX_QUESTIONS = 10      # questions per browser session
AVATARS = {"user": ":material/person:", "assistant": ":material/gavel:"}  # plain icons keep the theme black and white

# st.set_page_config(page_title="RAG Law", page_icon="👩‍⚖️")
st.set_page_config(page_title="RAG Law", page_icon=str(Path(__file__).parent / "assets" / "balance.png"))


@st.cache_resource
def get_qdrant():
    return QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"], timeout=60)


@st.cache_data(show_spinner=False, max_entries=20)
def fetch_verdict(doc_id):
    """Full text of one verdict (point id = uuid5 of the doc id, as built in embed_index.py)."""
    vid = str(uuid.uuid5(uuid.NAMESPACE_URL, "verdict|" + doc_id))
    points = get_qdrant().retrieve(VERDICTS, ids=[vid], with_payload=True)
    return points[0].payload if points else None


def md_escape(text):
    """Stop Streamlit from reading dataset text as markdown, math ($) or :emoji:."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>$~:])", r"\\\1", text)


def pretty_court(slug):
    return " ".join(w.upper() if w in ("pn", "pt", "ma") else w.capitalize() for w in slug.split("-"))


def find_snippet(sections, d):
    """Find the cited snippet in the original text. Returns (section, start, end) or None.
    The snippet is indexed text (whitespace joined, duplicate lines removed), so words are matched
    with any whitespace between them. The last word may be cut off, so it is dropped; if a removed
    duplicate line breaks the match, a shorter start of the snippet is tried."""
    words = d["snippet"].split()[:-1]
    names = sorted(sections, key=lambda n: n != d["section"])  # the cited section first
    for n in sorted({len(words), 20, 10}, reverse=True):
        if not 5 <= n <= len(words):
            continue
        pattern = r"\s+".join(re.escape(w) for w in words[:n])
        for name in names:
            m = re.search(pattern, sections[name])
            if m:
                return name, m.start(), m.end()
    return None


def marked(text, start, end):
    """HTML of `text` with text[start:end] highlighted; line breaks are kept."""
    return ('<div style="white-space: pre-wrap">' + html.escape(text[:start]) + "<mark>"
            + html.escape(text[start:end]) + "</mark>" + html.escape(text[end:]) + "</div>")


@st.dialog("Putusan", width="large")
def show_verdict(d):
    v = fetch_verdict(d["doc_id"])
    if v is None:
        st.warning("Teks putusan tidak ditemukan.")
        return
    st.markdown(f"**{v['case_number'] or 'Nomor tidak diketahui'}** · {pretty_court(v['lembaga_peradilan'])} · {v['year']}")
    st.caption(f"{v['sub_klasifikasi'].replace('-', ' ')} · putusan: {d['sentence']}")
    st.caption("Teks asli dari dataset: huruf kecil dan tanpa tanda baca. Cuplikan yang dikutip ditandai kuning.")
    hit = find_snippet(v["sections"], d)
    if hit:
        name, start, end = hit
        text = v["sections"][name]
        a, b = max(0, start - 400), min(len(text), end + 400)
        st.markdown("**Cuplikan dalam konteks**")
        st.html(marked(text[a:b], start - a, end - a))  # short view, so the match is visible without scrolling
    else:
        st.caption("Cuplikan tidak ditemukan persis di teks asli (teks terindeks sudah dinormalisasi).")
    open_section = hit[0] if hit else ("amar_putusan" if d["section"] == "case_card" else d["section"])
    for name, text in v["sections"].items():
        with st.expander(name.replace("_", " ").capitalize(), expanded=(name == open_section)):
            if hit and name == hit[0]:
                st.html(marked(text, hit[1], hit[2]))
            else:
                st.text(text.strip())


def render_documents(docs, msg_idx):
    """One card per cited verdict, with a button to open the original."""
    st.markdown("**Putusan yang dikutip**")
    for d in docs:
        with st.container(border=True):
            st.markdown(f"**[{d['n']}] {d['case_number'] or 'Nomor tidak diketahui'}** · "
                        f"{pretty_court(d['court'])} · {d['year']}")
            st.caption(f"{d['sub_klasifikasi'].replace('-', ' ')} · putusan: {d['sentence']} · "
                       f"bagian: {d['section'].replace('_', ' ')}")
            st.markdown("> " + md_escape(d["snippet"]) + "…")  # quote of the indexed text, cut at 300 characters
            if st.button("Buka putusan", key=f"open_{msg_idx}_{d['n']}"):
                show_verdict(d)


# ---- sidebar ----
with st.sidebar:
    st.header("RAG Law")
    st.write("Membantu mencari putusan pidana Indonesia yang mirip dengan pertanyaan atau kronologi Anda.")
    if st.button("Hapus percakapan", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"Pertanyaan: {st.session_state.get('asked', 0)}/{MAX_QUESTIONS}")
    # st.caption("Catatan: RAG Law hanya memberikan putusan pidana, bukan perkara perdata.")
    # st.caption("Dataset: indo-law (Nuranti, Yulianti & Husin, 2022, *Computers* 11(6), 88). "
    #            "[github.com/ir-nlp-csui/indo-law](https://github.com/ir-nlp-csui/indo-law)")
    st.divider()
    st.caption("[Gmail](mailto:amn.ibrahim3@gmail.com) | [LinkedIn](https://www.linkedin.com/in/muhammad-ibra/) | [Dataset](https://github.com/ir-nlp-csui/indo-law)")
    
# ---- chat ----
st.title("RAG Law")
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "documents"}], one list per browser session

for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"], avatar=AVATARS[m["role"]]):
        st.markdown(m["content"])
        if m.get("documents"):
            render_documents(m["documents"], i)

if not st.session_state.messages:
    st.caption("Contoh: *\"Apa hukuman untuk pencurian sepeda motor?\"* atau ceritakan kronologi singkat suatu kasus.")

# if question := st.chat_input("Tulis pertanyaan atau kronologi singkat"):
limit_reached = st.session_state.get("asked", 0) >= MAX_QUESTIONS
if limit_reached:
    st.info(f"Batas {MAX_QUESTIONS} pertanyaan untuk sesi ini sudah tercapai.")
if question := st.chat_input("Tulis pertanyaan atau kronologi singkat", disabled=limit_reached):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages][-MAX_HISTORY:]
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            with st.spinner("Mencari putusan..."):
                out = run_llm(question, history)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat mencari ({type(e).__name__}). Coba lagi.")
            st.stop()
    st.session_state.asked = st.session_state.get("asked", 0) + 1
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": out["answer"], "documents": out["documents"]})
    st.rerun()  # redraw everything from session state, including the new cards