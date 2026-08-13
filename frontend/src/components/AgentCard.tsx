import type { AgentName, AgentState, AgentStatus } from "../types";

const LABELS: Record<AgentName, string> = {
  fundamentals: "Fundamentals",
  technical: "Technical",
  news: "News & Sentiment",
};

const STATUS_STYLES: Record<AgentStatus, string> = {
  queued: "bg-slate-100 text-slate-500",
  running: "bg-amber-100 text-amber-700",
  ok: "bg-emerald-100 text-emerald-700",
  failed: "bg-rose-100 text-rose-700",
};

const STATUS_LABELS: Record<AgentStatus, string> = {
  queued: "queued",
  running: "running…",
  ok: "done",
  failed: "failed",
};

export function AgentCard({ agent, state }: { agent: AgentName; state?: AgentState }) {
  const status = state?.status ?? "queued";

  return (
    <div className="rounded-lg border border-slate-200 p-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-slate-700">{LABELS[agent]}</span>
        <span
          className={`rounded-full px-2 py-0.5 text-xs font-medium ${STATUS_STYLES[status]} ${
            status === "running" ? "animate-pulse" : ""
          }`}
        >
          {STATUS_LABELS[status]}
        </span>
      </div>

      {status === "failed" && state?.error && <p className="mt-2 text-xs text-rose-600">{state.error}</p>}

      {status === "ok" && (
        <>
          {state?.summary && <p className="mt-2 text-xs leading-relaxed text-slate-600">{state.summary}</p>}
          {!!state?.findings?.length && (
            <ul className="mt-2 space-y-1.5 border-t border-slate-100 pt-2">
              {state.findings.map((f) => (
                <li key={f.id} className="text-xs leading-relaxed text-slate-500">
                  <span className="font-medium text-slate-700">{f.claim}</span> — {f.evidence}
                  {f.source.url && (
                    <a
                      href={f.source.url}
                      target="_blank"
                      rel="noreferrer"
                      className="ml-1 text-indigo-600 hover:underline"
                    >
                      source
                    </a>
                  )}
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  );
}
