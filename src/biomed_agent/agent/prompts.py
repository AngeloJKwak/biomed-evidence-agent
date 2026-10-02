"""Prompt templates. Bump `Settings.prompt_version` when these change so runs stay traceable."""

from __future__ import annotations

from biomed_agent.schemas import Chunk, DraftAnswer, SourceDocument

PLANNER_SYSTEM = """\
You plan literature searches for a biomedical evidence assistant. The searches run against
PubMed (E-utilities esearch syntax) and ClinicalTrials.gov (free-text term).

Write PubMed queries that a medical librarian would write: combine the intervention, population,
and outcome concepts with AND, use OR for synonyms, and add MeSH terms or [tiab] tags where they
help. Prefer 1-2 focused queries over many broad ones. Include a ClinicalTrials.gov term only if
registered trials are likely to be relevant (interventions, drugs, devices); otherwise set it
to null.
"""

ASSESSOR_SYSTEM = """\
You decide whether retrieved evidence is enough to answer a biomedical question. Mark it
sufficient when the passages directly address the question's main concepts, even if the evidence
is limited or mixed - limited evidence is something the answer can report. Mark it insufficient
only when a key concept from the question is missing from the passages entirely, and then propose
different follow-up queries that target what is missing.
"""

ANSWER_SYSTEM = """\
You are a biomedical evidence assistant. Answer using only the numbered evidence passages
provided. Every claim must cite the source ids that directly support it, written exactly as they
appear in the passage headers (for example PMID:37952131 or NCT03574597). Do not cite a source
for something it does not state. If the evidence is thin, conflicting, or only from early-phase or
observational studies, say so in evidence_quality and limitations rather than overstating it.
Do not give individualized medical advice.
"""

VERIFIER_SYSTEM = """\
You check whether each claim in a draft answer is supported by the passages it cites. A claim is
supported only if the cited passages state it or it follows directly from them; numbers, effect
directions, and populations must match. Judge each claim independently and give a short reason.
"""


def format_evidence(chunks: list[Chunk], docs: dict[str, SourceDocument]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        doc = docs.get(c.source_id)
        meta = []
        if doc is not None:
            if doc.year:
                meta.append(str(doc.year))
            if doc.venue:
                meta.append(doc.venue)
            meta.extend(f"{k}={v}" for k, v in doc.metadata.items() if k != "doi")
        header = f"[{i}] source_id={c.source_id}" + (f" ({'; '.join(meta)})" if meta else "")
        blocks.append(f"{header}\n{c.text}")
    return "\n\n".join(blocks)


def planner_prompt(question: str) -> str:
    return f"Question: {question}"


def assessor_prompt(question: str, evidence: str, tried_queries: list[str]) -> str:
    tried = "\n".join(f"- {q}" for q in tried_queries) or "- (none)"
    return (
        f"Question: {question}\n\nQueries already run:\n{tried}\n\n"
        f"Retrieved passages:\n{evidence or '(no passages retrieved)'}"
    )


def answer_prompt(question: str, evidence: str, feedback: str | None = None) -> str:
    prompt = f"Evidence passages:\n\n{evidence}\n\nQuestion: {question}"
    if feedback:
        prompt += (
            "\n\nA reviewer found problems with your previous draft. Fix them; drop any claim the "
            f"evidence does not support:\n{feedback}"
        )
    return prompt


def verifier_prompt(draft: DraftAnswer, evidence: str) -> str:
    claims = "\n".join(
        f"{i}. {c.text} [cites: {', '.join(c.citations) or 'none'}]"
        for i, c in enumerate(draft.claims)
    )
    return f"Evidence passages:\n\n{evidence}\n\nClaims to check (0-indexed):\n{claims}"
