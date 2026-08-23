import { AgentCard } from "./AgentCard";
import { PANEL } from "../lib/surfaces";
import type { AgentName, TickerAgents } from "../types";

// Which agents to render, in canonical order: exactly the ones this turn dispatched, which
// is exactly the ones that received an `agent_started`.
//
// Not the turn's `aspects`. Those are what the user asked about, and the two differ
// whenever a follow-up re-fetches only the cells that went stale: a turn can legitimately
// have `aspects = [fundamentals, technical, news]` while dispatching one cell. Rendering
// the requested aspects then left two sections with no state at all, which — since there
// is no longer a "queued" status — displayed as agents stuck loading forever.
//
// Deriving from what was dispatched also keeps this in agreement with the progress steps
// above it, which count the same set.
const ORDER: AgentName[] = ["fundamentals", "technical", "news"];

const SHORT_LABELS: Record<AgentName, string> = {
  fundamentals: "fundamentals",
  technical: "technicals",
  news: "news",
};

export function TickerGroup({
  ticker,
  agents,
  collapsed,
}: {
  ticker: string;
  agents: TickerAgents;
  collapsed: boolean;
}) {
  const shown = ORDER.filter((a) => agents[a] !== undefined);
  if (shown.length === 0) return null;

  const findingCount = shown.reduce((n, a) => n + (agents[a]?.findings?.length ?? 0), 0);

  // Expanded, a single section already carries its own heading, so repeating it beside the
  // ticker just says "technical" twice on one line. Collapsed, that heading is hidden
  // inside the fold and this is the only thing describing what's in there.
  const showSubtitle = collapsed || shown.length > 1;
  const subtitle =
    shown.length === ORDER.length ? "full research" : shown.map((a) => SHORT_LABELS[a]).join(", ");

  const header = (
    <div className="flex items-baseline gap-2">
      <span className="font-mono text-sm font-semibold tracking-wide text-ink-100">{ticker}</span>
      {showSubtitle && <span className="text-xs text-ink-500">{subtitle}</span>}
    </div>
  );

  // Once the report exists, this block is supporting evidence rather than the main event,
  // so it folds away instead of pushing the answer below the fold. Open while the run is
  // still live, because then it IS the content.
  if (collapsed) {
    return (
      <details className={`group rounded-xl ${PANEL}`}>
        <summary className="flex cursor-pointer select-none items-center gap-2 px-4 py-2.5 text-ink-400 hover:text-ink-200">
          <svg
            viewBox="0 0 16 16"
            fill="none"
            className="h-3 w-3 shrink-0 transition-transform group-open:rotate-90"
            aria-hidden="true"
          >
            <path d="M6 4l4 4-4 4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          {header}
          <span className="ml-auto text-xs text-ink-500">
            {findingCount} {findingCount === 1 ? "finding" : "findings"}
          </span>
        </summary>
        <div className="space-y-3 border-t border-ink-800/80 px-4 py-3">
          {shown.map((agent) => (
            <AgentCard key={agent} agent={agent} state={agents[agent]} />
          ))}
        </div>
      </details>
    );
  }

  return (
    <div className={`rounded-xl px-4 py-3 ${PANEL}`}>
      <div className="mb-3">{header}</div>
      <div className="space-y-3">
        {shown.map((agent) => (
          <AgentCard key={agent} agent={agent} state={agents[agent]} />
        ))}
      </div>
    </div>
  );
}
