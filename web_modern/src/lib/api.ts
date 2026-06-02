import type { FrameResult, QAEvent, Status, Track } from "./types";

export function parseSSELine(line: string): QAEvent | null {
  if (!line.startsWith("data:")) return null;
  try { return JSON.parse(line.slice(5).trim()) as QAEvent; } catch { return null; }
}
export const wsBase = () => (location.protocol === "https:" ? "wss:" : "ws:") + "//" + location.host;

export async function getStatus(): Promise<Status> {
  const r = await fetch("/api/status"); return r.json();
}
export async function ingest(form: FormData): Promise<{ session_id?: string; error?: string }> {
  const r = await fetch("/api/ingest", { method: "POST", body: form }); return r.json();
}
export async function askQA(
  tracks: Track[], question: string, apiKey: string | null,
  onEvent: (e: QAEvent) => void,
): Promise<void> {
  const r = await fetch("/api/qa", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tracks, question, api_key: apiKey }),
  });
  const reader = r.body!.getReader(); const dec = new TextDecoder(); let buf = "";
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n"); buf = parts.pop() ?? "";
    for (const p of parts) for (const ln of p.split("\n")) { const e = parseSSELine(ln); if (e) onEvent(e); }
  }
}
export type { FrameResult };
