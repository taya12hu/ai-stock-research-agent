const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000";

export interface StartResponse {
  session_id: string;
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Request to ${path} failed (${res.status}): ${detail}`);
  }
  return res.json() as Promise<T>;
}

export function startResearch(question: string): Promise<StartResponse> {
  return postJSON<StartResponse>("/research", { question });
}

export function askFollowUp(sessionId: string, question: string): Promise<StartResponse> {
  return postJSON<StartResponse>(`/research/${sessionId}/ask`, { question });
}

export function streamUrl(sessionId: string): string {
  return `${API_BASE}/research/${sessionId}/stream`;
}
