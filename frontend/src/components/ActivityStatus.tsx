import { TickerGroup } from "./TickerGroup";
import type { AgentName, QueryType, TranscriptEntry } from "../types";

const QUERY_TYPE_LABELS: Record<QueryType, string> = {
  single: "Single-stock analysis",
  portfolio: "Portfolio analysis",
  comparison: "Comparison",
};

const AGENTS: AgentName[] = ["fundamentals", "technical", "news"];

function agentsSettled(entry: TranscriptEntry): boolean {
  if (entry.tickers.length === 0) return false;
  return entry.tickers.every((ticker) =>
    AGENTS.every((agent) => {
      const status = entry.agents[ticker]?.[agent]?.status;
      return status === "ok" || status === "failed";
    }),
  );
}

interface Phase {
  key: string;
  label: string;
}

// The single thing the graph is doing for this turn right now, in the order the SSE
// events actually arrive (build_graph.py's routing) — each new phase fully replaces the
// last one on screen, there's no accumulated history of prior phases. Only called before
// this turn has content (see ActivityStatus below), so the phase returned is always still
// in progress.
function currentPhase(entry: TranscriptEntry): Phase {
  if (!entry.classified) return { key: "understand", label: "Reading your question" };

  if (entry.tickers.length > 0) {
    if (agentsSettled(entry)) return { key: "synthesize", label: "Writing the final report" };

    const hasAgentData = Object.keys(entry.agents).length > 0;
    // A follow-up's tickers arrive with no query-type label to show alongside them, so
    // there's nothing distinct to say between "classified" and "researching" — go
    // straight to the research phase rather than manufacturing an empty-looking one.
    if (hasAgentData || !entry.queryType) return { key: "research", label: `Researching ${entry.tickers.join(", ")}` };

    return { key: "setup", label: `${QUERY_TYPE_LABELS[entry.queryType]} — ${entry.tickers.join(", ")}` };
  }

  // Classified with no tickers: a clarification question, an off-topic reply, an
  // unresolvable ticker, or a follow-up answered straight from context.
  return { key: "reply", label: "Preparing your reply" };
}

function Spinner() {
  return <span className="h-3 w-3 shrink-0 animate-spin rounded-full border-2 border-slate-700 border-t-indigo-400" />;
}

// Two different things live here, and they behave differently:
//   - The ticker/agent findings table is real content — once an agent has posted
//     anything, it stays on screen, report or no report. It's not process chrome.
//   - The status line (spinner + "Researching…" / "Writing the final report") is pure
//     process narration — one line that fully replaces itself as SSE events arrive
//     (Claude-style "what am I doing right now"), and it disappears the instant there's
//     an actual result to show instead, because at that point there's nothing left to
//     narrate.
export function ActivityStatus({ entry, active }: { entry: TranscriptEntry; active: boolean }) {
  const contentReady = entry.report !== null || entry.answer !== null;
  const hasResearchData = Object.keys(entry.agents).length > 0;

  const showStatusLine = !contentReady && active;
  const phase = showStatusLine ? currentPhase(entry) : null;

  if (!hasResearchData && !phase) return null;

  // The findings (when there's anything to show yet) come first — that's the actual
  // content accumulating in real time. The status line, when there is one, trails
  // underneath it as a caption, not a header sitting above content that isn't there yet.
  return (
    <div>
      {hasResearchData && (
        <div className="space-y-3">
          {entry.tickers.map((ticker) => (
            <TickerGroup key={ticker} ticker={ticker} agents={entry.agents[ticker] ?? {}} />
          ))}
        </div>
      )}
      {phase && (
        <div className={`flex items-center gap-2 text-sm font-medium text-slate-200 ${hasResearchData ? "mt-2" : ""}`}>
          <Spinner />
          <span>{phase.label}</span>
        </div>
      )}
    </div>
  );
}
