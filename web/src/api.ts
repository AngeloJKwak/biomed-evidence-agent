import type { AnswerResponse, ProgressEvent } from "./types";

interface StreamHandlers {
  onProgress: (event: ProgressEvent) => void;
  onResult: (result: AnswerResponse) => void;
  onError: (message: string) => void;
}

/**
 * POST the question and read the Server-Sent Events response.
 * (EventSource only supports GET, so the SSE frames are parsed by hand.)
 */
export async function askStream(
  question: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const resp = await fetch("/api/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!resp.ok || !resp.body) {
    const detail = await resp.text().catch(() => "");
    handlers.onError(`Request failed (${resp.status}). ${detail}`);
    return;
  }

  const reader = resp.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value.replace(/\r\n/g, "\n");
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      dispatch(frame, handlers);
    }
  }
}

function dispatch(frame: string, handlers: StreamHandlers): void {
  let event = "message";
  const data: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!data.length) return;
  const payload = JSON.parse(data.join("\n"));
  if (event === "progress") handlers.onProgress(payload as ProgressEvent);
  else if (event === "result") handlers.onResult(payload as AnswerResponse);
  else if (event === "error") handlers.onError(payload.detail ?? "Unknown error");
}
