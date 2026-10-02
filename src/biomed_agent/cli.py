"""Command-line entry point: `biomed-agent ask "..."` or `biomed-agent serve`."""

from __future__ import annotations

import argparse
import asyncio
import json
import textwrap

from biomed_agent.config import load_env
from biomed_agent.factory import build_agent
from biomed_agent.tracing import flush


async def _ask(question: str, as_json: bool) -> None:
    agent, http = build_agent()
    try:
        async for event in agent.stream(question):
            if event["node"] != "done":
                if not as_json:
                    details = {k: v for k, v in event.items() if k != "node"}
                    print(f"  · {event['node']:<9} {json.dumps(details)}")
                continue
            result = event["result"]
            if as_json:
                print(json.dumps(result, indent=2))
            else:
                _print_answer(result)
    finally:
        await http.aclose()
        flush()


def _print_answer(r: dict) -> None:
    a = r["answer"]
    print("\n" + textwrap.fill(a["summary"], 100) + "\n")
    for i, c in enumerate(a["claims"]):
        print(
            textwrap.fill(
                f"{i + 1}. {c['text']} [{', '.join(c['citations'])}]", 100, subsequent_indent="   "
            )
        )
    print(f"\nEvidence quality: {a['evidence_quality']}")
    print(f"Limitations: {a['limitations']}\n")
    titles = {s["source_id"]: s for s in r["sources"]}
    cited = dict.fromkeys(c for claim in a["claims"] for c in claim["citations"])
    for cid in cited:
        s = titles.get(cid)
        print(f"  {cid}: {s['title'] if s else '(not in retrieved sources)'}")
    status = "grounded" if r["grounded"] else f"{len(r['issues'])} citation issue(s)"
    u = r["usage"]
    print(
        f"\nrun {r['run_id']} · {status} · {r['latency_s']}s · "
        f"{u['llm_calls']} LLM calls · {u['input_tokens']}/{u['output_tokens']} tokens in/out"
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="biomed-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="Answer a question from the command line")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="Print the full JSON response")
    serve = sub.add_parser("serve", help="Run the API server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    load_env()

    if args.cmd == "ask":
        asyncio.run(_ask(args.question, args.json))
    else:
        import uvicorn

        uvicorn.run("biomed_agent.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
