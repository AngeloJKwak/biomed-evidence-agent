import { useMemo, useRef, useState } from "react";
import { askStream } from "./api";
import type { AnswerResponse, CitationIssue, ProgressEvent, SourceDocument } from "./types";

const EXAMPLES = [
  "Does semaglutide reduce cardiovascular events in adults with obesity but no diabetes?",
  "Should healthy older adults take low-dose aspirin for primary prevention?",
  "Does lecanemab slow cognitive decline in early Alzheimer's disease?",
  "Does dexamethasone reduce mortality in patients hospitalized with COVID-19?",
];

const STEPS: { node: string; label: string }[] = [
  { node: "plan", label: "Plan searches" },
  { node: "retrieve", label: "Retrieve evidence" },
  { node: "assess", label: "Check coverage" },
  { node: "generate", label: "Draft answer" },
  { node: "verify", label: "Verify citations" },
];

type Status = "idle" | "running" | "done" | "error";

export default function App() {
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const [result, setResult] = useState<AnswerResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeSource, setActiveSource] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  async function run(q: string) {
    const text = q.trim();
    if (text.length < 8 || status === "running") return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setQuestion(text);
    setStatus("running");
    setEvents([]);
    setResult(null);
    setError(null);
    setActiveSource(null);
    try {
      await askStream(
        text,
        {
          onProgress: (e) => setEvents((prev) => [...prev, e]),
          onResult: (r) => {
            setResult(r);
            setStatus("done");
          },
          onError: (msg) => {
            setError(msg);
            setStatus("error");
          },
        },
        controller.signal,
      );
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setError((e as Error).message);
        setStatus("error");
      }
    }
  }

  return (
    <div className="page">
      <header className="masthead">
        <div className="brand">
          <span className="brand-mark" aria-hidden>
            ◆
          </span>
          Biomed Evidence Agent
        </div>
        <p className="tagline">
          Answers biomedical questions from PubMed abstracts and ClinicalTrials.gov records. Every
          claim is cited and checked.
        </p>
      </header>

      <form
        className="ask"
        onSubmit={(e) => {
          e.preventDefault();
          run(question);
        }}
      >
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              run(question);
            }
          }}
          placeholder="Ask a clinical evidence question…"
          rows={2}
          maxLength={500}
          aria-label="Question"
        />
        <button type="submit" disabled={status === "running" || question.trim().length < 8}>
          {status === "running" ? "Working…" : "Ask"}
        </button>
      </form>

      {status === "idle" && (
        <div className="examples">
          {EXAMPLES.map((ex) => (
            <button key={ex} className="example" onClick={() => run(ex)}>
              {ex}
            </button>
          ))}
        </div>
      )}

      {status !== "idle" && <Timeline events={events} status={status} />}

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="results">
          <AnswerView
            result={result}
            activeSource={activeSource}
            onCite={(id) => {
              setActiveSource(id);
              document.getElementById(`src-${id}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
            }}
          />
          <SourcesPanel result={result} activeSource={activeSource} onSelect={setActiveSource} />
        </div>
      )}

      <footer className="disclaimer">
        Research demo. Not medical advice. Summaries come from abstracts and registry records, not
        full texts.
      </footer>
    </div>
  );
}

function Timeline({ events, status }: { events: ProgressEvent[]; status: Status }) {
  const seen = new Set<string>(events.map((e) => e.node));
  const last = events.at(-1)?.node;
  const detail = (node: string): string | null => {
    const evs = events.filter((e) => e.node === node);
    const e = evs.at(-1);
    if (!e) return null;
    switch (node) {
      case "plan":
        return `${(e.queries as string[]).length} PubMed quer${(e.queries as string[]).length === 1 ? "y" : "ies"}${e.trials_query ? " + trials" : ""}`;
      case "retrieve":
        return `${e.total_sources} found · ${e.passages} passages${evs.length > 1 ? ` · ${evs.length} rounds` : ""}`;
      case "assess":
        return e.followup ? "gap found, searching again" : "coverage OK";
      case "generate":
        return `${e.claims} claims${evs.length > 1 ? " (revised)" : ""}`;
      case "verify":
        return e.grounded ? "all claims supported" : `${e.issues} issue(s)`;
      default:
        return null;
    }
  };

  return (
    <ol className="timeline" aria-label="Agent progress">
      {STEPS.map(({ node, label }) => {
        const done = seen.has(node) && (status !== "running" || last !== node || node === "verify");
        const active = status === "running" && !seen.has(node) && isNext(node, last);
        return (
          <li key={node} className={done ? "done" : active ? "active" : ""}>
            <span className="dot" aria-hidden />
            <span className="step-label">{label}</span>
            <span className="step-detail">{detail(node)}</span>
          </li>
        );
      })}
    </ol>
  );
}

function isNext(node: string, last: string | undefined): boolean {
  const order = STEPS.map((s) => s.node);
  if (!last) return node === "plan";
  if (last === "revise") return node === "generate";
  if (last === "assess") return node === "generate";
  return order.indexOf(node) === order.indexOf(last) + 1;
}

function AnswerView({
  result,
  activeSource,
  onCite,
}: {
  result: AnswerResponse;
  activeSource: string | null;
  onCite: (id: string) => void;
}) {
  const { answer, issues } = result;
  const issuesByClaim = useMemo(() => {
    const m = new Map<number, CitationIssue[]>();
    for (const i of issues) m.set(i.claim_index, [...(m.get(i.claim_index) ?? []), i]);
    return m;
  }, [issues]);
  const known = new Set(result.sources.map((s) => s.source_id));

  return (
    <section className="answer" aria-label="Answer">
      <div className="answer-head">
        <h2>Answer</h2>
        <span className={`badge ${result.grounded ? "ok" : "warn"}`}>
          {result.grounded ? "✓ Citations verified" : `⚠ ${issues.length} citation issue${issues.length === 1 ? "" : "s"}`}
        </span>
      </div>
      <p className="summary">{answer.summary}</p>

      {answer.claims.length > 0 && (
        <ol className="claims">
          {answer.claims.map((c, i) => {
            const claimIssues = issuesByClaim.get(i);
            return (
              <li key={i} className={claimIssues ? "flagged" : ""}>
                <span>{c.text}</span>{" "}
                {c.citations.map((id) => (
                  <button
                    key={id}
                    className={`cite ${known.has(id) ? "" : "missing"} ${activeSource === id ? "active" : ""}`}
                    onClick={() => onCite(id)}
                    title={known.has(id) ? "Show source" : "Source not in retrieved evidence"}
                  >
                    {id}
                  </button>
                ))}
                {claimIssues?.map((iss, j) => (
                  <div key={j} className="issue">
                    {iss.kind.replace("_", " ")}: {iss.detail}
                  </div>
                ))}
              </li>
            );
          })}
        </ol>
      )}

      <div className="notes">
        <div>
          <h3>Evidence quality</h3>
          <p>{answer.evidence_quality}</p>
        </div>
        <div>
          <h3>Limitations</h3>
          <p>{answer.limitations}</p>
        </div>
      </div>

      <dl className="run-meta">
        <div><dt>Run</dt><dd className="mono">{result.run_id}</dd></div>
        <div><dt>Latency</dt><dd>{result.latency_s.toFixed(1)}s</dd></div>
        <div><dt>LLM calls</dt><dd>{result.usage.llm_calls}</dd></div>
        <div><dt>Tokens in / out</dt><dd>{result.usage.input_tokens.toLocaleString()} / {result.usage.output_tokens.toLocaleString()}</dd></div>
        <div><dt>Retrieval rounds</dt><dd>{result.retrieval_rounds}</dd></div>
        <div><dt>Revisions</dt><dd>{result.revisions}</dd></div>
        {result.trace_url && (
          <div>
            <dt>Trace</dt>
            <dd>
              <a href={result.trace_url} target="_blank" rel="noreferrer">
                Langfuse ↗
              </a>
            </dd>
          </div>
        )}
      </dl>
    </section>
  );
}

function SourcesPanel({
  result,
  activeSource,
  onSelect,
}: {
  result: AnswerResponse;
  activeSource: string | null;
  onSelect: (id: string) => void;
}) {
  const cited = new Set(result.answer.claims.flatMap((c) => c.citations));
  return (
    <aside className="sources" aria-label="Sources">
      <h2>
        Sources <span className="count">{result.sources.length}</span>
      </h2>
      <ul>
        {result.sources.map((s) => (
          <SourceCard
            key={s.source_id}
            source={s}
            cited={cited.has(s.source_id)}
            active={activeSource === s.source_id}
            onSelect={() => onSelect(s.source_id)}
          />
        ))}
      </ul>
    </aside>
  );
}

function SourceCard({
  source,
  cited,
  active,
  onSelect,
}: {
  source: SourceDocument;
  cited: boolean;
  active: boolean;
  onSelect: () => void;
}) {
  const [open, setOpen] = useState(false);
  const meta = [
    source.year,
    source.venue,
    source.metadata.phase,
    source.metadata.status?.toLowerCase().replace(/_/g, " "),
  ].filter(Boolean);
  return (
    <li
      id={`src-${source.source_id}`}
      className={`source ${active ? "active" : ""} ${cited ? "" : "uncited"}`}
      onClick={onSelect}
    >
      <div className="source-top">
        <span className={`type ${source.source_type}`}>
          {source.source_type === "pubmed" ? "PubMed" : "Trial"}
        </span>
        <a href={source.url} target="_blank" rel="noreferrer" className="mono" onClick={(e) => e.stopPropagation()}>
          {source.source_id} ↗
        </a>
        {!cited && <span className="muted">context only</span>}
      </div>
      <div className="source-title">{source.title}</div>
      {meta.length > 0 && <div className="source-meta">{meta.join(" · ")}</div>}
      <button
        className="link"
        onClick={(e) => {
          e.stopPropagation();
          setOpen((o) => !o);
        }}
      >
        {open ? "Hide text" : "Show text"}
      </button>
      {open && <p className="source-text">{source.text}</p>}
    </li>
  );
}
