# Biomed Evidence Agent

An agentic RAG system that answers clinical-evidence questions from **PubMed** abstracts and
**ClinicalTrials.gov** records. Every claim is cited, checked against the retrieved sources, and
logged with full provenance.

> *"Does semaglutide reduce cardiovascular events in adults with obesity but no diabetes?"*
> → plans PubMed + registry searches, retrieves and ranks passages, checks whether the evidence
> covers the question (and searches again if not), drafts a cited answer, verifies every
> citation, revises once if anything fails, and returns the answer with its sources.

**Stack:** LangGraph · Claude (Anthropic SDK, structured outputs) · Hugging Face
sentence-transformers (BGE) · PyTorch · ChromaDB · FastAPI + SSE · React/TypeScript · Langfuse ·
Docker · GitHub Actions

---

## How it works

```mermaid
flowchart LR
    Q([Question]) --> P[plan<br/><sub>LLM: PubMed + trials queries</sub>]
    P --> R[retrieve<br/><sub>E-utilities + CT.gov v2<br/>chunk → embed → Chroma</sub>]
    R --> A{assess<br/><sub>LLM: evidence sufficient?</sub>}
    A -- gap found --> R
    A -- sufficient --> G[generate<br/><sub>LLM: claims + citations</sub>]
    G --> V{verify<br/><sub>code: citation checks<br/>LLM: claim support</sub>}
    V -- issues --> G
    V -- grounded --> F([answer + sources<br/>+ run record])
```

| Step | What happens | Why it matters |
|---|---|---|
| **plan** | Claude writes PubMed queries (boolean, MeSH) and a ClinicalTrials.gov term. | Librarian-style queries retrieve far better than pasting the question. |
| **retrieve** | Parallel async calls to NCBI E-utilities and the CT.gov v2 API. Records are chunked, embedded locally with `BAAI/bge-small-en-v1.5`, cached in Chroma, and ranked. At most 2 passages per source. | Re-asking reuses embeddings. The per-source cap keeps one long record from crowding out the rest. |
| **assess** | Claude decides whether the passages cover the question. If not, it proposes follow-up queries (bounded by `max_retrieval_rounds`). | Turns one-shot RAG into an agent that can recover from a weak first search. |
| **generate** | Structured output (`summary`, `claims[{text, citations}]`, `evidence_quality`, `limitations`). | Citations are data, not prose, so they can be checked. |
| **verify** | Code checks that every claim is cited and every cited id was actually retrieved. Then Claude judges whether each claim is supported by its cited passages. Failures go back to `generate` as reviewer feedback. | Catches made-up PMIDs and claims that overstate the evidence. |

Every LLM step returns a validated Pydantic model via the Anthropic SDK's `messages.parse`
(constrained JSON decoding), so no output is parsed from free text.

## Engineering for production

- **Evaluation:** [`evals/`](evals) holds 12 questions with known answers from landmark trials
  (SELECT, EMPEROR-Reduced, DAPA-CKD, SURMOUNT-1, CLARITY AD, ASPREE, VITAL, KEYNOTE-024,
  RECOVERY, ORION-10/11, CLEAR Outcomes, TOGETHER). It includes three null-result trials, to
  check that the agent doesn't overstate benefit. Every expected PMID and NCT id was looked up
  through the PubMed API, and the reference facts are taken from each paper's abstract
  conclusion. Metrics:
  - deterministic: source recall, citation validity, citation coverage, grounded rate
  - LLM-as-judge (a separate model): fact recall and finding direction
  - cost and latency for each question
- **Provenance and lineage:** each answer writes `.data/runs/<date>/<run_id>.json` with the code
  version (git SHA), model and prompt versions, every search query, a content hash of every
  passage shown to the model, per-call token usage and request ids, and the final answer with
  any citation issues. Any answer can be audited after the fact.
- **Observability:** optional [Langfuse](https://langfuse.com) tracing. Each question is one
  trace, with a span per graph step and each Claude call nested as a generation (model, effort,
  tokens). The agent's own checks (`grounded`, `citation_issues`, `revisions`) are attached as
  trace scores, and the eval harness adds its metrics to the same traces. Tracing failures are
  logged and never break an answer. See [Tracing with Langfuse](#tracing-with-langfuse).
- **Reliability:** retries with exponential backoff and jitter on 429/5xx from the public
  APIs. A semaphore keeps requests under NCBI's rate limit. Server-side refusal fallback is
  enabled on Claude calls. Truncated or refused structured outputs raise typed errors.
- **Streaming UX:** `POST /api/ask/stream` streams graph progress as Server-Sent Events. The
  React UI shows each step live, highlights the source when you click a citation, and flags any
  claim that failed verification.
- **Tests:** 36 tests run offline (no API keys). The PubMed and CT.gov parsers are tested
  against recorded real API responses. HTTP is mocked with `respx`. The full LangGraph flow
  (revision loop, follow-up retrieval, no-results path, SSE endpoint) runs against a scripted
  LLM.
- **Delivery:** multi-stage Docker image (CPU-only PyTorch, embedding model baked in, non-root
  user, healthcheck). CI runs lint, tests, the frontend build, and the image build on every
  push.

## Quickstart

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), Node 20+, and an
[Anthropic API key](https://console.anthropic.com/).

```bash
git clone https://github.com/AngeloJKwak/biomed-evidence-agent.git
cd biomed-evidence-agent
cp .env.example .env          # add ANTHROPIC_API_KEY
uv sync --all-extras
```

**CLI**

```bash
uv run biomed-agent ask "Does dexamethasone reduce mortality in patients hospitalized with COVID-19?"
```

**Web app**

```bash
uv run biomed-agent serve            # API on :8000
npm --prefix web install
npm --prefix web run dev             # UI on :5173 (proxies /api)
```

**Docker**

```bash
docker compose up --build            # UI + API on :8000
```

The first run downloads the ~130 MB embedding model from Hugging Face.

## Running the evals

Evals make real Claude API calls. Start small:

```bash
uv run python evals/run_eval.py --limit 2
uv run python evals/run_eval.py                     # all 12 questions
uv run python evals/run_eval.py --ids aspree vital  # specific questions
```

Results are written to `evals/results/<timestamp>.{json,md}`.

## Tracing with Langfuse

1. Create a project at [Langfuse](https://langfuse.com) (cloud or self-hosted) and copy its API
   keys from **Settings → API Keys**.
2. Add them to `.env`. `LANGFUSE_BASE_URL` must match your project's region:

   ```bash
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_BASE_URL=https://cloud.langfuse.com      # EU; US is https://us.cloud.langfuse.com
   ```

3. Run anything (CLI, web app, or evals). Tracing turns on automatically when the keys are
   present. Set `LANGFUSE_TRACING_ENABLED=false` to turn it off without removing them.

Each run shows up as an `evidence-agent` trace:

```
evidence-agent (agent)        input: question · output: answer · scores: grounded, citation_issues, revisions
├── plan (chain)
│   └── claude (generation)   model, effort, input/output tokens
├── retrieve (retriever)      queries, sources fetched, passages ranked
├── assess (chain)
│   └── claude (generation)
├── generate (chain)
│   └── claude (generation)
└── verify (evaluator)
    └── claude (generation)
```

The web UI links each answer to its trace, and `evals/run_eval.py` attaches `eval_fact_recall`,
`eval_source_recall`, `eval_citation_validity`, and `eval_finding_correct` scores to each eval
run's trace. That makes it easy to filter Langfuse for low-scoring runs and inspect exactly what
the model saw. The test suite never sends traces, even if keys are exported in your shell.

## Configuration

All settings are environment variables prefixed with `BIOMED_` (see
[`config.py`](src/biomed_agent/config.py) and [`.env.example`](.env.example)). Common ones:

| Variable | Default | Notes |
|---|---|---|
| `BIOMED_LLM_MODEL` | `claude-opus-5-5` | Planning, generation, verification |
| `BIOMED_JUDGE_MODEL` | `claude-sonnet-5-5` | Eval judge only. A different model from the generator, to reduce self-preference bias. |
| `BIOMED_ANSWER_EFFORT` | `high` | `plan`/`verify` efforts are set separately |
| `BIOMED_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Any sentence-transformers model |
| `BIOMED_TOP_K` | `12` | Passages given to the model |
| `BIOMED_MAX_RETRIEVAL_ROUNDS` | `2` | Upper bound on assess → retrieve loops |
| `BIOMED_MAX_REVISIONS` | `1` | Upper bound on verify → generate loops |

## Project layout

```
src/biomed_agent/
  agent/graph.py        LangGraph state machine and EvidenceAgent (stream / ask)
  agent/prompts.py      System prompts and evidence formatting
  sources/              PubMed (E-utilities) and ClinicalTrials.gov v2 clients
  retrieval/            Chunking, embeddings (HF sentence-transformers), Chroma store
  grounding.py          Deterministic citation normalization and checks
  llm.py                Structured-output wrapper over the Anthropic SDK + usage tracking
  provenance.py         Per-run audit records
  tracing.py            Optional Langfuse integration
  api.py, cli.py        FastAPI (JSON + SSE) and command-line entry points
evals/                  Dataset, runner, metrics, LLM judge
web/                    React + TypeScript UI (Vite)
tests/                  Offline test suite with recorded API fixtures
```

## Design decisions

- **Embeddings run locally.** BGE-small (33M parameters) runs fine on CPU and needs no second
  API key. It's also a sentence-transformers model, so swapping in a biomedical model (e.g.
  PubMedBERT-based) is a one-line config change.
- **The vector store is a cache, not a corpus.** Each question pulls fresh records from the live
  APIs, and Chroma avoids re-embedding records already seen. Searches are scoped to the current
  question's sources, so stale results never leak into an answer.
- **Code checks run before the LLM check.** Uncited claims and made-up ids are caught without an
  extra model call. The LLM verifier only judges claims whose citations are real.
- **Loops are bounded.** Retrieval and revision loops have hard limits, so cost and latency per
  question stay predictable.

## Limitations

- Works from **abstracts and registry records**, not full text. Subgroup results and safety
  tables often only appear in the full paper.
- The ranking is purely semantic. It doesn't weight study design (meta-analysis > RCT >
  observational), so evidence quality is left to the model's `evidence_quality` field.
- The eval set is small and limited to well-known trials. It catches regressions but isn't a
  benchmark.

## Troubleshooting

- **`CERTIFICATE_VERIFY_FAILED` on Windows:** antivirus HTTPS scanning (e.g. Avast Web Shield)
  or a corporate proxy re-signs certificates. The app verifies TLS against the OS certificate
  store via [`truststore`](https://github.com/sethmlarson/truststore) by default
  (`BIOMED_USE_OS_TRUST_STORE=true`). For `uv` itself, pass `--native-tls`.

---

*Research and portfolio project. Not medical advice.*
