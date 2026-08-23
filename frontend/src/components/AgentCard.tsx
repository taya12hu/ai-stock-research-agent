import type { AgentName, AgentState, Finding } from "../types";

const LABELS: Record<AgentName, string> = {
  fundamentals: "Fundamentals",
  technical: "Technical",
  news: "News & Sentiment",
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

function SourceLink({ source }: { source: Finding["source"] }) {
  if (!source.url) return <span className="w-3.5 shrink-0" />;
  return (
    <a
      href={source.url}
      target="_blank"
      rel="noreferrer"
      title={source.label}
      aria-label={`Source: ${source.label}`}
      className="mt-0.5 shrink-0 text-ink-600 transition hover:text-blue-400"
    >
      <svg viewBox="0 0 16 16" fill="none" className="h-3.5 w-3.5">
        <path
          d="M6.5 3.5H3.5v9h9v-3M9.5 3.5h3v3M12.5 3.5L7 9"
          stroke="currentColor"
          strokeWidth="1.3"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </a>
  );
}

// Findings are a claim plus its supporting evidence, not tabular data. The previous
// three-column table ("Claim | Evidence | Src") had to fit inside a third-width card, so
// every claim wrapped to four lines and the word "link" repeated down the page.
//
// Two genuinely different kinds of evidence arrive here, and one layout cannot serve
// both. `market_data` evidence is a figure ("RSI (14-day): 78.49") — short, worth
// right-aligning in tabular-nums so values line up down the column. `web` evidence is a
// sentence of article prose, sometimes 200+ characters. Rendering that as a nowrap
// right-hand figure is what crushed the news claims into a one-word-per-line column and
// pushed the snippet off the edge of the card, so it stacks underneath the claim instead
// and drops the mono/tabular treatment, which does nothing for prose.
function FindingRow({ finding }: { finding: Finding }) {
  const isProse = finding.source.type === "web";

  if (isProse) {
    return (
      <div className="border-b border-ink-800/60 py-2.5 last:border-b-0">
        <div className="flex items-start gap-3">
          <p className="min-w-0 flex-1 text-[13px] leading-relaxed text-ink-300">{finding.claim}</p>
          <SourceLink source={finding.source} />
        </div>
        <p className="mt-1 border-l-2 border-ink-800 pl-2.5 text-[12px] leading-relaxed text-ink-500">
          {finding.evidence}
        </p>
      </div>
    );
  }

  // Stacks below `sm`. Side-by-side needs room for both halves, and a figure like
  // "Revenue growth (YoY): 0.7068" cannot shrink — on a phone it would either overflow the
  // card or crush the claim into a one-word column, which is the same failure the news
  // rows had.
  return (
    <div className="flex flex-col gap-0.5 border-b border-ink-800/60 py-2 last:border-b-0 sm:flex-row sm:items-baseline sm:gap-3">
      <span className="min-w-0 flex-1 text-[13px] leading-relaxed text-ink-300">{finding.claim}</span>
      <div className="flex items-baseline gap-2 sm:shrink-0">
        <span className="font-mono text-[12px] tabular-nums text-ink-400">{finding.evidence}</span>
        <SourceLink source={finding.source} />
      </div>
    </div>
  );
}

function Skeleton() {
  return (
    <div className="space-y-2 py-1">
      {[0, 1, 2].map((i) => (
        <div key={i} className="h-3 animate-pulse rounded bg-ink-800/70" style={{ width: `${88 - i * 16}%` }} />
      ))}
    </div>
  );
}

export function AgentCard({ agent, state }: { agent: AgentName; state?: AgentState }) {
  // No `?? "queued"` fallback: this component is only rendered for agents the turn
  // actually dispatched, and every dispatched agent gets an `agent_started` event before
  // any of them finish. An absent state means the event hasn't landed yet, which is
  // running, not pending.
  const status = state?.status ?? "running";

  return (
    <section className="border-t border-ink-800 pt-3 first:border-t-0 first:pt-0">
      <div className="mb-2 flex items-center gap-2">
        <h4 className="text-[11px] font-semibold uppercase tracking-wider text-ink-500">
          {LABELS[agent]}
        </h4>
        {status === "running" && (
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-400" aria-label="running" />
        )}
        {status === "failed" && (
          <span className="text-[11px] font-medium text-rose-400">unavailable</span>
        )}
      </div>

      {status === "running" && <Skeleton />}

      {status === "failed" && state?.error && (
        <p className="text-[13px] leading-relaxed text-rose-400/90" title={state.error}>
          {friendlyError(state.error)}
        </p>
      )}

      {status === "ok" && (
        <>
          {state?.summary && (
            <p className="mb-2 text-[13px] leading-relaxed text-ink-400">{state.summary}</p>
          )}
          {!!state?.findings?.length && (
            <div>
              {state.findings.map((f) => (
                <FindingRow key={f.id} finding={f} />
              ))}
            </div>
          )}
        </>
      )}
    </section>
  );
}
