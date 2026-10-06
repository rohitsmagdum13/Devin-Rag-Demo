# Fixed-corpus comparison

Run from the repository root with a securely configured `OPENAI_API_KEY`:
```bash
.venv/bin/python -m evaluation.run --output data/evaluation-results.json
# Optional repeated samples (billable, 3x the Q&A requests):
.venv/bin/python -m evaluation.run --repeats 3 --output data/evaluation-repeated.json
```
No key is needed for `pytest`; its provider calls are mocked and outbound sockets are blocked.
The runner itself uses **live OpenAI**, not a mock or an LLM judge. It uploads only the committed
synthetic corpus. Defaults are `text-embedding-3-small` / 1536 and `gpt-4o-mini`; environment
settings may change these, but both systems always share the same settings, index and provider.

## Protocol / fairness
- Three versioned TXT documents in `corpus/`: reimbursements, travel, equipment.
- Eight independent questions in `questions.json`: three direct, two multipart, three unknown.
- No conversation history for either system. Alternate system order per question/repeat to
  reduce systematic warm-cache ordering effects. There is no concurrency or hidden warm-up.
- Ingest once, reuse one persistent Chroma index for **both** systems. A corpus/config SHA-256
  fingerprints its namespace under `data/evaluation-indexes/` (override with `--store-root`).
  Repeated runs skip duplicate embeddings. Extra/wrong document identities/counts fail the run.
  The user's UI index at `VECTOR_STORE_PATH` is deliberately not included in this evaluation.
- `NaiveBaseline.ask` embeds the unchanged original question and calls `store.search` exactly
  once. It never calls the planner, rewrites a query or performs a follow-up search. An empty
  result still counts as one retrieval, but skips answer generation.
- `RetrievalAgent.ask` uses its ordinary bounded planner, rewriting and follow-up behavior.
- Both use identical `provider.answer` prompts, model, temperature, retrieval K, evidence payload
  and fail-closed `validated_answer` checks. No settings are tuned per system/question.

This intentionally small smoke corpus fits within default K=5, so both systems may retrieve all
documents. It is **not** a benchmark demonstrating an agentic quality advantage. Equal quality
with higher agentic latency/tokens is a valid result, not a failed evaluation.

## Deterministic scoring (no LLM judging)
Each answerable question has reference facts defined by regex + expected filename.

1. **Correctness:** fraction of expected fact regexes found in a supported answer, 0–1. A missing
   or unsupported answer scores 0. Regexes include units, e.g. `30 days`, `$75 per day`,
   `$1200 per year`; comma formatting and `/day`/`/year` variants are accepted.
2. **Citation accuracy:** valid-citation fraction × reference-fact citation coverage. A citation
   must resolve to the canonical corpus chunk ID, document/name/page/text, with an exact
   nonempty quote. Coverage is the fraction of expected facts appearing in valid quotes from
   the expected filenames. Empty citations score 0 for answerable questions.
3. **Unsupported handling:** for unknown questions, 1 only for the explicit application
   insufficient-evidence response with `supported=false` and no citations; otherwise 0.
   Unknown-question correctness uses the same score. Citation accuracy is 1 if no citations,
   0 otherwise. Unsupported handling is not applicable to answerable questions.

These are deterministic **lexical/reference proxies**, not semantic entailment checks. They can
miss correct paraphrases/spelled-out numbers, overlook contradictions/additional invented facts,
and do not prove injection immunity or complete citation grounding. Inspect saved answers and
quotes manually. No `llm_judging` is implemented; the JSON explicitly says `not_used`.

## Measurements / output
The console prints a comparison table. JSON saves settings (never keys), corpus manifest/hashes,
question-set hash, timestamp, ingestion metrics, execution order, every question/reference,
answer/citation/excerpt/retrieval query, per-case scores/checks, latency and available token usage.

Summary correctness/citation accuracy are unweighted means across all question trials; unsupported
handling averages only unknown questions. Latency is the median end-to-end `ask` duration,
including planning and query embeddings, excluding ingestion/scoring. Search count and embedding,
prompt and completion tokens are means across successful trials; planning tokens are included for
the agent. Missing API usage is recorded as 0 by the provider, not an estimate. Usage counts reported
responses, not SDK retry attempts; failures may consume billable tokens that cannot be recovered.
Ingestion latency/tokens are saved separately and not charged to either comparator.

API errors are saved as failed trials (quality scores 0), printed in the error column, and cause a
nonzero exit after JSON is written. Failed-call token/search counts are unknown, not silently counted
as measured successes. With no successful trials those means are JSON null / `n/a` in the table.
Timing includes failed trials. The default is one sample per question/system;
OpenAI behavior can vary even at temperature zero. `--repeats` accepts 1–10 for additional samples.
Generated stores and results under `data/` stay ignored, not committed.
