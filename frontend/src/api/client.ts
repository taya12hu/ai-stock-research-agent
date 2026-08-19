const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000";

export interface StartResponse {
  session_id: string;
}

// Carries the HTTP status so callers can branch on it (e.g. 404 = session not found)
// without parsing the error message string.
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new ApiError(res.status, detail || res.statusText);
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
