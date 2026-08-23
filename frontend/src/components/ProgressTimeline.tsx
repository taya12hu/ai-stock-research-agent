import type { AgentName, TranscriptEntry } from "../types";

// Progress narration, split so it can be interleaved with the results it produces.
//
// The steps are not rendered as one block. `ResearchSteps` goes above the ticker cards and
// `SynthesisStep` below them, so the reading order is: what I'm about to do, the thing it
// produced, what I'm doing next. Keeping them in a single box meant the live status stayed
// pinned at the top while results pushed the page down, so you had to scroll back up to
// see what was happening — and "Writing your report" appeared *above* the data it was
// written from, which is backwards.
//
// There is no surrounding panel. These are lines in the conversation, not a widget.
//
// Every state is derived from the SSE-fed `entry`. Nothing is timed or animated on a
// schedule, so the display cannot claim a step the graph has not reached.

const AGENT_LABELS: Record<AgentName, string> = {
  fundamentals: "Fundamentals",
  technical: "Technicals",
  news: "News",
};

const ORDER: AgentName[] = ["fundamentals", "technical", "news"];

type StepState = "done" | "active";
type CellState = "done" | "active" | "failed";

interface Cell {
  ticker: string;
  agent: AgentName;
  state: CellState;
}

// The cells this turn actually dispatched, which is exactly what received an
// `agent_started`. Deliberately not `tickers × aspects`: a follow-up re-fetches only the
// cells that went stale, so a turn can legitimately run one agent out of three. Counting
// the requested aspects instead would show "1 of 3" and never reach completion.
function dispatchedCells(entry: TranscriptEntry): Cell[] {
  const cells: Cell[] = [];
  for (const ticker of entry.tickers) {
    const perAgent = entry.agents[ticker];
    if (!perAgent) continue;
    for (const agent of ORDER) {
      const status = perAgent[agent]?.status;
      if (!status) continue;
      cells.push({
        ticker,
        agent,
        state: status === "ok" ? "done" : status === "failed" ? "failed" : "active",
      });
    }
  }
  return cells;
}

export function researchSettled(entry: TranscriptEntry): boolean {
  const cells = dispatchedCells(entry);
  return cells.length > 0 && cells.every((c) => c.state !== "active");
}

export function findingsTotal(entry: TranscriptEntry): number {
  let n = 0;
  for (const ticker of entry.tickers) {
    for (const agent of ORDER) {
      n += entry.agents[ticker]?.[agent]?.findings?.length ?? 0;
    }
  }
  return n;
}

function StepIcon({ state }: { state: StepState }) {
  if (state === "active") {
    return (
      <span className="h-3 w-3 shrink-0 animate-spin rounded-full border-2 border-ink-700 border-t-blue-400" />
    );
  }
  return (
    <svg viewBox="0 0 12 12" className="h-3 w-3 shrink-0 text-blue-400/70" aria-hidden="true">
      <path
        d="M2.5 6.2 L4.8 8.5 L9.5 3.5"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function Step({
  state,
  label,
  detail,
  children,
}: {
  state: StepState;
  label: string;
  detail?: string;
  children?: React.ReactNode;
}) {
  return (
    <div>
      <div className="flex items-center gap-2.5">
        <StepIcon state={state} />
        <span className={`text-[13px] ${state === "active" ? "font-medium text-ink-100" : "text-ink-400"}`}>
          {label}
        </span>
        {detail && <span className="text-[12px] text-ink-500">{detail}</span>}
      </div>
      {/* Indented to clear the icon column, so the detail reads as part of this step. */}
      {children && <div className="ml-[22px] mt-1.5 space-y-1">{children}</div>}
    </div>
  );
}

function AgentPill({ label, state, count }: { label: string; state: CellState; count?: number }) {
  const tone =
    state === "failed"
      ? "border-rose-500/30 bg-rose-500/10 text-rose-300"
      : state === "done"
        ? "border-ink-700/70 bg-ink-800/50 text-ink-300"
        : "border-blue-500/40 bg-blue-500/10 text-blue-200";

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] ${tone}`}>
      {state === "active" && (
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-blue-400" aria-hidden="true" />
      )}
      {label}
      {state === "done" && count !== undefined && <span className="text-ink-500">{count}</span>}
      {state === "failed" && <span>failed</span>}
    </span>
  );
}

// Everything up to and including the research step. Renders above the ticker cards.
export function ResearchSteps({ entry }: { entry: TranscriptEntry }) {
  const cells = dispatchedCells(entry);
  const done = cells.filter((c) => c.state !== "active").length;
  const settled = researchSettled(entry);

  // No scope resolved: a clarification, an off-topic reply, or an answer from context.
  // Narrating "Researching" here would describe work that never happens.
  if (entry.tickers.length === 0) {
    return (
      <div className="space-y-2">
        <Step state={entry.classified ? "done" : "active"} label="Thinking" />
        {entry.classified && <Step state="active" label="Preparing your reply" />}
      </div>
    );
  }

  const shape =
    entry.queryType === "comparison" ? "comparison" : entry.queryType === "portfolio" ? "portfolio" : undefined;

  return (
    <div className="space-y-2">
      <Step state="done" label="Read your question" detail={shape} />

      {/* Before any `agent_started` lands we know the scope but not yet whether anything
          needs fetching — a fully-fresh turn dispatches nothing at all. Stay vague until
          the events say otherwise rather than promising research that may not run. */}
      {cells.length === 0 ? (
        <Step state="active" label={`Checking ${entry.tickers.join(", ")}`} />
      ) : (
        <Step
          state={settled ? "done" : "active"}
          label={`${settled ? "Researched" : "Researching"} ${entry.tickers.join(", ")}`}
          detail={`${done} of ${cells.length}`}
        >
          {entry.tickers.map((ticker) => {
            const mine = cells.filter((c) => c.ticker === ticker);
            if (mine.length === 0) return null;
            return (
              <div key={ticker} className="flex flex-wrap items-center gap-1.5">
                <span className="font-mono text-[11px] font-medium text-ink-400">{ticker}</span>
                {mine.map((c) => (
                  <AgentPill
                    key={c.agent}
                    label={AGENT_LABELS[c.agent]}
                    state={c.state}
                    count={entry.agents[ticker]?.[c.agent]?.findings?.length}
                  />
                ))}
              </div>
            );
          })}
        </Step>
      )}
    </div>
  );
}

// The step that follows the results. Renders below the ticker cards, so it only ever
// appears once the data it is about to write from is on screen.
export function SynthesisStep({ entry }: { entry: TranscriptEntry }) {
  if (!researchSettled(entry)) return null;
  const n = findingsTotal(entry);
  return <Step state="active" label="Writing your report" detail={n > 0 ? `${n} findings` : undefined} />;
}
