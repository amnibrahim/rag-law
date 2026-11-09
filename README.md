# RAG Law (working title)

Research assistant for Indonesian court verdicts. A user types a short legal question or a short
description of a case, and the tool retrieves similar past criminal verdicts (a precedent-style lookup)
so that an answer to a public legal question can be written faster. Every cited verdict can be opened
in full, with the matching passage highlighted.

Portfolio project: LangChain + Qdrant + Streamlit. Status log below, updated as work progresses.

> Dataset: indo-law, Nuranti, Yulianti & Husin (2022), "Predicting the Category and the Length of
> Punishment in Indonesian Courts Based on Previous Court Decision Documents", Computers 11(6), 88.
> Source: https://github.com/ir-nlp-csui/indo-law (22,630 criminal verdicts, XML, 11 annotated sections).
> The raw data is not committed to this repo (`data/` is git-ignored).

Last updated: 2026-11-10

---

## 1. Current status

| # | Task | Status |
|---|---|---|
| 0 | Setup: uv project, `.env`, `.gitignore`, dataset clone, Qdrant Cloud cluster, connection check | Done |
| 1 | Data scan of all 22,630 files (`scan_data.py`) | Done |
| 1b | Subset selection (`make_subset.py`) | Done: 4,987 docs in `internal/subset_ids.txt` |
| 2 | Parser + chunker (`build_chunks.py`) | Done: 130,275 chunks |
| 3 | Embed + index into Qdrant (`embed_index.py`) | Done: 130,275 chunk points + 4,987 verdict points |
| 4 | Retrieval smoke test (`smoke_test.py`, 10 questions) | Done, findings in section 6 |
| 5 | RAG agent (`agent_lc.py`) | Done |
| 6 | Streamlit UI (`app.py`) | Done |
| E | Evaluation (`evaluate.py`, 20 questions) | Run; LLM judge not yet checked against my own labels |
| 7 | Deploy (Streamlit Community Cloud), secrets, per-session query cap | Query cap done; deployment in progress |
| 8 | README polish | To do |

---

## 2. Decisions so far

| Area | Decision | Notes |
|---|---|---|
| Vector DB | Qdrant Cloud, free tier (Sydney, GCP) | Free tier: 1 GB RAM, 4 GB disk. One cluster holds the chunks and the full verdict text (needed by the viewer, because the XML files cannot ship in the repo). Chunk vectors are on disk with int8 quantization kept in RAM. |
| Hosting | Streamlit Community Cloud | Free; secrets via the Streamlit secrets setting. Apps are hosted in the US (per the Streamlit docs), the database is in Sydney. VPS is the fallback. |
| Embeddings | `text-embedding-3-small`, 1536 dims | $0.02 per 1M tokens. |
| Chat model | `gpt-4o-mini`, temperature 0 | |
| RAG pattern | LangChain `create_agent` with one tool, `search_verdicts(query, amar, year, court)` | The model writes its own short queries and optional filters. Results are grouped by verdict. History (last 12 messages) is passed into `run_llm`. |
| Retrieval | `QdrantVectorStore` + `OpenAIEmbeddings` | The stored payload is flat, so a small subclass overrides one private method of `langchain-qdrant` (checked on version 1.1.0) to keep all fields as metadata. |
| Citations | The answer cites verdicts as `[n]`; the UI lists only the cited verdicts | The list in the UI always matches the text. |
| Tracing | LangSmith (`@traceable` on `run_llm`) | Off in the public deployment, because it would send visitors' questions to my account. |
| Frontend | Streamlit, single process, no separate API | Wide `st.dialog` shows the original verdict and highlights the cited passage. |
| Access | No password; 20 questions per browser session | A password limits who can try the demo. The cap resets on page refresh, so a hard monthly spending limit on the OpenAI account is the real protection. |
| Chunking | Per section, about 3,000 characters, 300 overlap | Recursive splitter (newline, space). No sentence or semantic splitting, because the text has no punctuation. |
| Chunk header | Built from the `<putusan>` attributes and the year, e.g. `Putusan pn kudus \| pidana umum \| pemalsuan \| 2014 \| bagian: fakta` | Hyphens replaced by spaces. |
| Payload (stored, not embedded) | doc_id, klasifikasi, sub_klasifikasi (merged label) and raw label, amar, amar_lainnya, lama_hukuman (days), lembaga_peradilan, provinsi, url, year, case_number, section, chunk text | Used for the UI and for filters on amar, year and court. |
| Chunk-size validation | Skipped the full experiment; checked with a 10-question smoke test and a 20-question evaluation | A chunking comparison stays optional. |

### Embedded sections
`riwayat_perkara`, `riwayat_penahanan`, `riwayat_dakwaan`, `riwayat_tuntutan`, `fakta`, `fakta_hukum`,
`pertimbangan_hukum`, `amar_putusan`. Short sections are merged into the next one.
Not embedded: `identitas`, `kepala_putusan`, `penutup`.

### Preprocessing in `build_chunks.py`
- Exact-duplicate lines within a document are dropped before embedding (12.8% of the embedded characters). The viewer shows the original text.
- One short "case card" chunk per verdict (case number, court, case type, outcome, start of `amar_putusan`).
- A small mapping merges near-duplicate `sub_klasifikasi` labels (23 documents relabelled); the raw label stays in the payload.

---

## 3. Dataset findings (measured on all 22,630 files)

- Every file parses, and every `<putusan>` tag has the same 10 attributes: `amar`, `amar_lainnya`, `id`, `klasifikasi`, `lama_hukuman`, `lembaga_peradilan`, `provinsi`, `status`, `sub_klasifikasi`, `url`.
- **Section tags are inconsistent.** Presence: kepala_putusan 100%, amar_putusan 99.9%, riwayat_perkara 98.8%, riwayat_dakwaan 98.5%, penutup 98.1%, fakta 96.5%, riwayat_tuntutan 95.7%, identitas 87.3%, riwayat_penahanan 82.4%, pertimbangan_hukum 47.4%, fakta_hukum 46.9%. When `pertimbangan_hukum` and `fakta_hukum` are missing, the judges' reasoning sits at the end of `fakta` (checked on 3 documents).
- **Criminal cases only.** pidana-khusus 11,901, pidana-umum 10,729. Civil disputes are not in the dataset.
- **Skewed classes.** 65 `sub_klasifikasi` values; narkotika-dan-psikotropika is 9,194 (41%), `tidak-diketahui` 2,067. Statistics describe the indexed subset, not the full dataset.
- **Noisy labels.** In the smoke test, two verdicts labelled `penghinaan` were about motorbike theft and a narcotics article. The label is not used as a filter.
- **Long documents.** Embedded sections: mean 15.7k tokens, median 12.7k, p90 25.7k, max 471,714. 3.22 characters per token.
- **Text format.** Lowercase, no punctuation, with some noise (`_unk_` tokens, broken words). The year is found for 99.8% of the subset and the case number for 96.3%. Case numbers are not unique on their own (`pn bd` appears for two courts), so the identity key is `doc_id`.
- `lama_hukuman` is in days.
- `status` has a single value; `amar_lainnya` has 1,239 values; `provinsi` has 5 values.
- Outcomes: pidana 22,174, bebas 305, lepas 151.

---

## 4. Subset and index

Subset rules (`make_subset.py`, seed 42):

1. Admit a document only if `amar_putusan`, `fakta` and `riwayat_dakwaan` are all present (979 removed).
2. Exclude `sub_klasifikasi = tidak-diketahui` (1,971 further removed).
3. Quality flags (under 1,000 tokens, noisy text, repeated lines): only 2 documents were flagged; both stay excluded.
4. Token cap 100,000 (100 documents removed; 84 of the 403 korupsi documents are above it, so korupsi verdicts in the demo skew shorter).
5. At most 300 random documents per `sub_klasifikasi`; smaller classes are taken in full. The cap limits cost and free-tier disk; it is not class balancing.
6. At least 50 `bebas` and 50 `lepas` verdicts, if available.

Result: 19,578 eligible documents, **4,987 selected** (4,843 pidana, 94 bebas, 50 lepas).

Index (measured):

| Item | Value |
|---|---|
| Chunks | 130,275 (median 2,801 characters, maximum 3,118) |
| Tokens embedded | 96.4M (script estimate), about $1.93 (billing not yet compared) |
| Indexing time | 4,413 s for the full run |
| Qdrant disk | 1.37 GiB of 4 GiB |
| Qdrant RAM | about 396 MiB of 1 GiB (observed at rest) |

---

## 5. Agent behaviour

- Answers in Indonesian by default; cites verdicts as `[n]`, and each cited verdict appears as a card with case number, court, year, case type, sentence, section and a quoted snippet.
- Writes 2-3 short queries about the offence and key acts, without names, dates or amounts (a long chronology is not searched as a whole). This also keeps personal details out of the embedding API.
- Filters: `amar` (pidana, bebas, lepas), `year` and `court`, only when the question asks for them.
- No legal conclusions or predictions. If nothing relevant is found (for example a civil dispute), it says so and cites nothing.
- The model decides relevance itself. The similarity score is not used as a cutoff, because an irrelevant question scored higher (0.583) than a relevant one (0.546) in the smoke test.
- `run_llm` returns `answer`, `documents` (cited verdicts) and `retrieved` (everything the searches returned).

---

## 6. Findings and open items

Smoke test (10 questions, retrieval only):
- Topic-level retrieval works for charge and sentence questions and for questions about mitigating and aggravating factors.
- Generic phrases ("hal-hal yang meringankan") pull in other case types; "tidak terbukti" appears in convictions too, so acquittal questions need the `amar` filter.
- The cited passage is a quote of the indexed text (lowercase, duplicate lines removed), so the dialog shows the original verdict and highlights the matching words.

Open items:
- [x] Reasoning sits at the end of `fakta` when `pertimbangan_hukum` is missing.
- [x] Qdrant free tier holds the chunks and the verdict text.
- [x] Year regex verified (99.8% found).
- [x] `lama_hukuman` is in days.
- [ ] Check the judge labels against my own labels (`eval/results/check.csv`) and report the agreement.
- [ ] Read why the chronology questions scored lowest (3 questions only), then decide whether to change the UI hint or the prompt.
- [ ] Write 10 new questions and use them only for a final measurement after any change, to avoid tuning to the first 20.
- [ ] Set a hard monthly spending limit on the OpenAI account.
- [ ] Check the model prices used for the cost estimate against OpenAI's pricing page.
- [ ] Case-number queries are weak with dense embeddings; exact-match filtering is not built.
- [ ] Deploy and test the public app. Statistics describe the indexed subset.

---

## 7. Evaluation (20 Indonesian questions, `evaluate.py`)

Method: each question goes through the agent. The 5 verdicts of its first search are labelled relevant or not
by `gpt-4o-mini`, which sees the question and the whole verdict. Hit@5 is the share of questions with a
relevant verdict among the 5; MRR@5 is the mean of 1 / rank of the first relevant verdict.
**The judge labels have not yet been checked against my own labels, so these numbers are provisional.**

| Metric | Value |
|---|---|
| Hit@5 | 14/20 = 0.70 (95% interval 0.48 to 0.85) |
| MRR@5 | 0.635 |
| Latency, question to answer (measured from my machine in Taipei) | p50 5.4 s, p95 6.9 s |
| Cost per query (answering model only, prices not yet verified) | about $0.0004, about $0.04 per 100 queries |

By question type (2 to 8 questions each, so only indicative):

| Type | n | Hit@5 | MRR@5 |
|---|---|---|---|
| tindak_pidana | 8 | 0.88 | 0.88 |
| pasal | 3 | 1.00 | 0.83 |
| faktor | 2 | 0.50 | 0.10 |
| bebas_lepas | 2 | 0.50 | 0.50 |
| fakta_serupa | 2 | 0.50 | 0.50 |
| kronologi | 3 | 0.33 | 0.33 |

Limits: small sample, one judge model from the same family as the answering model, labels only for the 5
verdicts that were returned. Faithfulness of the answer text was not measured.

---

## 8. Files and how to run (Windows cmd, from the project folder)

```
rag-law/
├── .env                  # OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY (git-ignored)
├── .gitignore            # includes .env, .venv/, data/, internal/, eval/results/
├── .streamlit/config.toml   # black and white theme
├── assets/balance.png    # tab icon
├── app.py                # Streamlit UI
├── agent_lc.py           # RAG agent: run_llm(question, history)
├── check_setup.py        # checks the OpenAI key and the Qdrant connection
├── scan_data.py          # scans all XML files, prints stats
├── make_subset.py        # applies the subset rules, writes internal/subset_ids.txt
├── build_chunks.py       # parses the selected verdicts, writes internal/chunks.jsonl
├── embed_index.py        # embeds the chunks and indexes both Qdrant collections
├── smoke_test.py         # 10 retrieval questions, prints the top verdicts
├── evaluate.py           # evaluation: run, judge, report
├── eval/questions.txt    # the 20 evaluation questions (id|type|question)
├── data/indo-law/dataset/   # the 22,630 XML files (git-ignored)
└── internal/, eval/results/ # generated files (git-ignored)
```

```cmd
uv run check_setup.py
uv run scan_data.py
uv run make_subset.py
uv run build_chunks.py
uv run embed_index.py
uv run streamlit run app.py
uv run evaluate.py --stage run
uv run evaluate.py --stage judge
uv run evaluate.py --stage report
```

- `build_chunks.py` makes no API calls. `embed_index.py` costs about $2 for the full run and can be resumed after an interruption; add `--limit 50` for a small test.
- `evaluate.py` is run in three stages. Between `judge` and `report`, fill the `my_label` column of `eval/results/check.csv` with 0 or 1.

---

## 9. Change log

- 2026-11-05: Setup, data scan, subset selection done. Decisions recorded above.
- 2026-11-05: Task 2 done: 130,275 chunks, 12.8% duplicate lines dropped, year found 99.8%, case number 96.3%.
- 2026-11-06: Task 3 done: both Qdrant collections indexed. Task 4 smoke test and Task 5 agent done.
- 2026-11-08: Task 6 done: Streamlit UI with verdict dialog and highlighted passage.
- 2026-11-09: Evaluation run on 20 questions (Hit@5 0.70, MRR@5 0.635; judge labels not yet validated). Query cap of 20 per session added.
- 2026-11-10: README brought up to date; deployment started.
