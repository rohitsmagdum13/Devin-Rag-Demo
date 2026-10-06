---
name: test-app
description: Required validation before each PR; offline checks and recorded Streamlit UI checks.
---

# Test this application before every PR

## 1. Setup / applicability
Read `AGENTS.md` and `README.md`. Run from the repo root on Python 3.10+:
```bash
python3 -m venv .venv # only if it does not exist
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/create_samples.py
```
For live checks, copy `.env.example` to ignored `.env` only if absent and configure the key
securely, or set `OPENAI_API_KEY` in the process environment. Never print the key/file.
Use an isolated ignored store, e.g. `VECTOR_STORE_PATH=artifacts/ui-chroma`.
`CHUNK_SIZE=1200`, `CHUNK_OVERLAP=200`, embedding model `text-embedding-3-small`, dimensions
1536 and answer model `gpt-4o-mini` are the defaults. No key is needed for offline tests.
Record model/settings in the report, not secrets.

Slice 1 (ingestion CLI) has no UI/ask/evaluation: execute steps 1–2 and CLI checks below; mark
UI/video **not applicable**. Slice 2 additionally checks CLI ask. After `app.py` exists,
steps 3–6 are mandatory for every UI-touching PR. Non-UI PRs still run steps 1–2 and their
relevant CLI/evaluation checks. Do not claim unrun checks passed.

## 2. Automated checks (must pass BEFORE opening a PR)
```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy rag_demo scripts # add app.py / evaluation when they exist
.venv/bin/python -m pytest -q
```
Tests mock OpenAI and block outbound socket connections. For a live CLI smoke:
```bash
VECTOR_STORE_PATH=artifacts/smoke-chroma .venv/bin/python -m rag_demo.cli ingest artifacts/samples/handbook.txt
VECTOR_STORE_PATH=artifacts/smoke-chroma .venv/bin/python -m rag_demo.cli list
# Slice 2 onward:
VECTOR_STORE_PATH=artifacts/smoke-chroma .venv/bin/python -m rag_demo.cli ask "What is the reimbursement deadline?"
```
Expected: ingestion lists filename/chunk count; second identical ingestion is a duplicate with
no additional embedding calls; answer says 30 days and references a resolvable chunk/excerpt.
If the key is absent, record live checks as untested and request it securely; do not fabricate results.

## 3. Start the application
Hand off UI-driven testing to the testing agent, which owns installation, start, browser and
recording. It should run:
```bash
VECTOR_STORE_PATH=artifacts/ui-chroma .venv/bin/python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
Open the browser on the app. Keep the foreground server in its own shell. Never show .env or
environment dumps on screen. Test with a fresh unique local store to avoid stale document counts.

## 4. Ingestion checks (record in the UI)
- Start recording, annotate each test and its assertion, and show the app.
- Upload `artifacts/samples/handbook.txt`, `travel.pdf`, and `equipment.docx`; press Index documents.
- Verify progress/status success, all three filenames, nonzero chunk counts, and readable content.
- Reupload an identical file, including a renamed copy: duplicate status, unchanged chunk count.
- Refresh/reopen the app: documents remain. PDF source citations must preserve page numbers;
  DOCX/TXT must not invent page numbers.

## 5. Q&A checks
- Ask “What is the reimbursement deadline?” Expect **30 days**, a citation to handbook.txt,
  and supporting excerpt containing that fact.
- Ask “What is the travel meal cap and the equipment allowance?” Expect **$75 per day** and
  **$1200 per year**, with evidence from travel.pdf and equipment.docx.
- Inspect the source link/expander: stable chunk ID, filename, page if available, and matching quote.
- Verify both turns remain in chat history and each has a numeric nonnegative latency in seconds.
  Record measured latency; a response over 30 seconds is an investigation flag, not a universal
  pass/fail SLA. API timeout is 45 seconds per request, plus at most one SDK retry.
- Ask “What is the CEO's personal phone number?” Expect clear insufficient-evidence language,
  no invented number, and no bogus citations. Ask “Hello” to verify the no-retrieval path.
- Upload `instructions.txt`, then ask about reimbursement again: embedded commands must not
  override the answer or reveal secrets. This is a regression test, not a proof of injection immunity.

## 6. Negative checks / delivery
- Try `unsupported.csv`: uploader rejects it (or application reports supported formats).
- Upload `empty.txt` and `blank.pdf`: actionable empty/no-readable-text error, no saved document.
- Upload `broken.pdf`: readable re-export/retry error, no crash. Continue asking a valid question
  afterwards to verify recovery.
- Stop/process recording with an honest pass/fail summary. Preserve the video and key screenshot.
- Attach the video to the UI PR description via supported upload/attachment tooling (convert to
  animated WebP for inline PR media if local video paths are rejected; also preserve original MP4).
- List actual automated commands/results, UI verdict, models, measured latencies, and gaps in the
  PR. Clearly distinguish live OpenAI checks from any explicitly mocked UI run.
- If recording or attachment is unavailable, report the blocker; do not call UI validation complete.

## 7. Evaluation (slice 4 onward)
Run `.venv/bin/python -m evaluation.run --output data/evaluation-results.json` with a live key.
Check the printed table and JSON: naive baseline uses one unchanged-question search, both systems
share models/index/generator, and deterministic scores are labeled separately from optional LLM
judging. Save results for inspection and report exactly what was run.
