// Mirrors biomed_agent.schemas (Python). Keep in sync with AnswerResponse.

export type SourceType = "pubmed" | "trial";

export interface SourceDocument {
  source_id: string;
  source_type: SourceType;
  title: string;
  text: string;
  url: string;
  year: number | null;
  venue: string | null;
  metadata: Record<string, string>;
}

export interface Claim {
  text: string;
  citations: string[];
}

export interface DraftAnswer {
  summary: string;
  claims: Claim[];
  evidence_quality: string;
  limitations: string;
}

export interface CitationIssue {
  claim_index: number;
  kind: "unknown_source" | "uncited_claim" | "unsupported";
  detail: string;
}

export interface AnswerResponse {
  run_id: string;
  question: string;
  answer: DraftAnswer;
  sources: SourceDocument[];
  issues: CitationIssue[];
  grounded: boolean;
  retrieval_rounds: number;
  revisions: number;
  latency_s: number;
  usage: {
    input_tokens: number;
    output_tokens: number;
    cache_read_input_tokens: number;
    llm_calls: number;
  };
  trace_id: string | null;
  trace_url: string | null;
}

export type NodeName = "plan" | "retrieve" | "assess" | "generate" | "verify" | "revise";

export interface ProgressEvent {
  node: NodeName;
  [key: string]: unknown;
}
