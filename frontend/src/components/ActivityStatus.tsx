import { ResearchSteps, SynthesisStep, findingsTotal } from "./ProgressTimeline";
import { TickerGroup } from "./TickerGroup";
import type { TranscriptEntry } from "../types";

// A finished turn keeps one line where the steps were. They mattered while they were
// happening; afterwards they are history, and leaving the full list above every past
// answer turns a three-question scrollback into a wall of ticked-off process.
function CompletedSummary({ entry }: { entry: TranscriptEntry }) {
  const n = findingsTotal(entry);
  return (
    <div className="flex items-center gap-2 text-[12px] text-ink-500">
      <svg viewBox="0 0 12 12" className="h-3 w-3 shrink-0 text-blue-400/60" aria-hidden="true">
        <path
          d="M2.5 6.2 L4.8 8.5 L9.5 3.5"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <span>
        Researched {entry.tickers.join(", ")}
        {n > 0 && ` · ${n} findings`}
      </span>
    </div>
  );
}

// The turn's process and its output, interleaved in the order they actually happen:
//
//   steps so far  ->  the cards those steps produced  ->  the step that comes next
//
// That ordering is the whole point. Rendering the steps as one block meant the live
// status sat at the top while results grew underneath it, so the thing you most wanted to
// see was the thing furthest from where you were looking. Here the active step is always
// the last element of the turn.
//
// Nothing outside `entry` is read, so a turn's progress can never render against another
// turn's state.
export function ActivityStatus({ entry, active }: { entry: TranscriptEntry; active: boolean }) {
  const contentReady = entry.report !== null || entry.answer !== null;
  const hasResearchData = Object.keys(entry.agents).length > 0;
  const showSteps = !contentReady && active;

  if (!hasResearchData && !showSteps) return null;

  const cards = hasResearchData && (
    <div className="space-y-2">
      {entry.tickers.map((ticker) => (
        <TickerGroup
          key={ticker}
          ticker={ticker}
          agents={entry.agents[ticker] ?? {}}
          collapsed={contentReady}
        />
      ))}
    </div>
  );

  return (
    <div className="space-y-3">
      {showSteps && <ResearchSteps entry={entry} />}
      {contentReady && hasResearchData && <CompletedSummary entry={entry} />}
      {cards}
      {showSteps && <SynthesisStep entry={entry} />}
    </div>
  );
}
