# Agentic document Q&A

Local persistent Chroma + OpenAI embeddings and responses, with a Streamlit interface.
Ingestion/list, agentic Q&A CLI and Streamlit UI are available. The evaluation harness arrives
in the next focused PR.

## Install / configure
Python 3.10+:
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```
Replace the placeholder key in your local `.env` or provide `OPENAI_API_KEY` in the environment.
Never commit/print keys. `.env`, runtime `data/`, and `artifacts/` are ignored. Automated tests
need no key. The key is needed for live ingestion, Q&A and evaluation.

Configuration defaults are in `.env.example`: `EMBEDDING_MODEL=text-embedding-3-small`,
`EMBEDDING_DIMENSIONS=1536`, `ANSWER_MODEL=gpt-4o-mini`, `VECTOR_STORE_PATH=data/chroma`,
`CHUNK_SIZE=1200`, `CHUNK_OVERLAP=200` (characters), `MAX_UPLOAD_MB=20`, `RETRIEVAL_K=5`,
`MAX_RETRIEVAL_CALLS=3`, `OPENAI_TIMEOUT_SECONDS=45`. Use embedding models that support the
dimensions parameter (`text-embedding-3-small` / `text-embedding-3-large`). Use a new store path
when changing embedding model/dimensions; existing indexes reject incompatible configuration.

## Ingest documents
```bash
.venv/bin/python scripts/create_samples.py
.venv/bin/python -m rag_demo.cli ingest artifacts/samples/handbook.txt
.venv/bin/python -m rag_demo.cli ingest artifacts/samples/travel.pdf
.venv/bin/python -m rag_demo.cli ingest artifacts/samples/equipment.docx
.venv/bin/python -m rag_demo.cli list
```
PDF, DOCX (paragraphs and tables), and UTF-8 TXT are supported. PDF page numbers are retained;
DOCX/TXT do not have reliable page numbers. Chunks preserve name, document hash, and chunk ID.
Overlap never crosses PDF page boundaries. Empty, binary, encrypted or unreadable files return
actionable errors. Scanned PDFs require external OCR. Identical bytes are indexed once even
under a different filename; the original stored filename remains authoritative. Embeddings
finish before persistence; storage errors roll back inserted chunks. A file lock serializes
ingestion. Abrupt process termination during multi-batch persistence is not crash-atomic; use
small documents and a fresh index if a process is killed while indexing.

Documents are transmitted to OpenAI for embedding and later answer generation. This is a
single-user local demo, not a multi-tenant authenticated service. Do not upload data you cannot
send to OpenAI. The local index is not encrypted. Telemetry is disabled in Chroma.

## Development / testing
```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy rag_demo scripts app.py
.venv/bin/python -m pytest -q
```
Offline tests use fake embeddings and real temporary Chroma stores; outbound connections are
blocked. Live smoke: run the ingestion/list commands above with a real environment key and
repeat ingestion to verify duplicate status. This is optional and billable.
Baseline evaluation instructions will be added with its slice.

## Agentic Q&A
```bash
.venv/bin/python -m rag_demo.cli ask "What is the reimbursement deadline?"
.venv/bin/python -m rag_demo.cli ask "What is the travel meal cap and equipment allowance?"
```
An OpenAI structured-output planner decides whether retrieval is needed, rewrites/splits search
queries, and may request one follow-up planning round if evidence is missing. Retrieval is capped
by `MAX_RETRIEVAL_CALLS` (default 3, maximum 5). Short greetings/thanks and the word "help"
can use a no-retrieval response. A deterministic conservative social-message guard prevents
planner misclassification from routing factual requests to the canned greeting.
Facts are answered only from evidence. An empty index, unsupported model answer, or invalid/missing
citation yields explicit insufficient-evidence language. Output includes stable chunk IDs, source
names, PDF pages, exact supporting quotes, retrieved context, queries, retrieval call count, end-to-end
latency and available usage tokens (including planning and query embeddings).

Document content and filenames are untrusted data, serialized separately from system instructions.
The model is told to ignore document commands, never reveal secrets, and avoid using history as
evidence. Citation ID/quote matching is deterministic; semantic support of every generated claim
still relies on the model and must be evaluated. These defenses do not prove injection immunity.
CLI questions are independent; the UI provides the previous three turns as bounded context.

## Web interface
```bash
.venv/bin/python -m streamlit run app.py
```
Open the URL printed by Streamlit. Upload one or more documents in the sidebar and press
**Index documents**. Watch processing status and inspect the indexed document/chunk list.
Ask questions in the chat; history includes response latency, retrieval counts and token usage.
Source links jump to uniquely identified citation expanders with filename, PDF page, chunk ID,
exact quote and supporting excerpt. Questions and answers are rendered as text, not executable
HTML/LLM-generated links. Clearing conversation does not remove indexed documents.

The Streamlit uploader transport limit defaults to 20 MB in `.streamlit/config.toml`, matching
`MAX_UPLOAD_MB`. To change it, also set `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` before startup.
If the two limits differ the sidebar explains them; the lower limit applies to UI uploads.

Missing key, configuration, unsupported/empty/unreadable files and provider failures surface
actionable errors. History is in-memory per browser session and is not a persisted conversation.
For recorded UI checks use the `test-app` skill and an isolated ignored store. Streamlit AppTest
checks widget rendering/history offline, but does not replace the required recorded browser run.

Read `AGENTS.md`; use `.agents/skills/test-app/SKILL.md` before every PR. Each UI-touching PR
requires a recorded browser test with the video attached, not just automated widget tests.