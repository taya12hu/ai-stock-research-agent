import type { AgentName, AgentState, AgentStatus } from "../types";

const LABELS: Record<AgentName, string> = {
  fundamentals: "Fundamentals",
  technical: "Technical",
  news: "News & Sentiment",
};

const STATUS_STYLES: Record<AgentStatus, string> = {
  queued: "bg-slate-800 text-slate-400",
  running: "bg-amber-500/10 text-amber-400 ring-1 ring-inset ring-amber-500/30",
  ok: "bg-emerald-500/10 text-emerald-400 ring-1 ring-inset ring-emerald-500/30",
  failed: "bg-rose-500/10 text-rose-400 ring-1 ring-inset ring-rose-500/30",
};

const STATUS_LABELS: Record<AgentStatus, string> = {
  queued: "queued",
  running: "running…",
  ok: "done",
  failed: "failed",
};

// Backend exceptions occasionally leak raw provider error payloads (JSON blobs, stack
// traces) instead of a short human-readable message. Those aren't useful to a user, so
// swap them for a generic message; the raw text stays available via the tooltip/console
// for debugging.
function friendlyError(raw: string): string {
  const looksRaw = raw.length > 180 || /\{['"]?error['"]?\s*:|Traceback|Error code:/i.test(raw);
  if (looksRaw) {
    console.error("Agent error:", raw);
    return "Something went wrong while analyzing this ticker. Please try again.";
  }
  return raw;
}

export function AgentCard({ agent, state }: { agent: AgentName; state?: AgentState }) {
  const status = state?.status ?? "queued";

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-slate-300">{LABELS[agent]}</span>
        <span
          className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${STATUS_STYLES[status]} ${
            status === "running" ? "animate-pulse" : ""
          }`}
        >
          {STATUS_LABELS[status]}
        </span>
      </div>

      {status === "failed" && state?.error && (
        <p className="mt-2 text-xs text-rose-400" title={state.error}>
          {friendlyError(state.error)}
        </p>
      )}

      {status === "ok" && (
        <>
          {state?.summary && <p className="mt-2 text-xs leading-relaxed text-slate-400">{state.summary}</p>}
          {!!state?.findings?.length && (
            <div className="mt-2 overflow-hidden rounded-md border border-slate-800/80">
              <table className="w-full border-collapse text-left text-[11px]">
                <thead>
                  <tr className="bg-slate-800/60 text-slate-400">
                    <th className="px-2 py-1.5 font-medium">Claim</th>
                    <th className="px-2 py-1.5 font-medium">Evidence</th>
                    <th className="w-14 px-2 py-1.5 font-medium">Src</th>
                  </tr>
                </thead>
                <tbody>
                  {state.findings.map((f) => (
                    <tr key={f.id} className="border-t border-slate-800/80 align-top">
                      <td className="px-2 py-1.5 font-medium text-slate-300">{f.claim}</td>
                      <td className="px-2 py-1.5 text-slate-500">{f.evidence}</td>
                      <td className="px-2 py-1.5">
                        {f.source.url ? (
                          <a
                            href={f.source.url}
                            target="_blank"
                            rel="noreferrer"
                            className="text-indigo-400 hover:underline"
                          >
                            link
                          </a>
                        ) : (
                          <span className="text-slate-600">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
