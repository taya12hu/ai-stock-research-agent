import { ActivityStatus } from "./ActivityStatus";
import { FinalReport } from "./FinalReport";
import { Markdown } from "./Markdown";
import { StatusBanner } from "./StatusBanner";
import type { ResearchStreamState, TranscriptEntry } from "../types";

function UserBubble({ question }: { question: string }) {
  return (
    <div className="flex justify-end">
      {/* The one saturated surface in the feed. Stays in the backdrop's blue family so
          the user's turn reads as part of the composition rather than a chip pasted on
          top of it. */}
      <div className="max-w-[88%] rounded-2xl rounded-tr-sm bg-gradient-to-br from-blue-600 to-indigo-700 px-4 py-2.5 text-sm text-white shadow-[0_2px_22px_-6px_rgba(37,99,235,0.65)] sm:max-w-[80%]">
        {question}
      </div>
    </div>
  );
}

// Everything for one turn — its live activity, ticker cards, and eventual answer/report —
// renders together, in this order, immediately after that turn's own question. Nothing
// here reads from outside `entry`, so a turn's cards can never appear detached from (or
// stuck above) the message that actually triggered them.
function TranscriptTurn({
  entry,
  isLast,
  running,
}: {
  entry: TranscriptEntry;
  isLast: boolean;
  running: boolean;
}) {
  const active = isLast && running;

  return (
    <div className="space-y-3">
      <UserBubble question={entry.question} />
      {entry.answer && <Markdown>{entry.answer}</Markdown>}

      <StatusBanner notes={entry.notes} error={entry.error} />

      <ActivityStatus entry={entry} active={active} />

      {entry.report && <FinalReport markdown={entry.report} />}
    </div>
  );
}

export function ConversationFeed({ state }: { state: ResearchStreamState }) {
  const running = state.status === "running";

  return (
    <div className="mx-auto max-w-6xl space-y-5 px-4 py-5 sm:px-6 sm:py-8">
      {state.transcript.map((entry, index) => (
        <TranscriptTurn key={entry.id} entry={entry} isLast={index === state.transcript.length - 1} running={running} />
      ))}
    </div>
  );
}
