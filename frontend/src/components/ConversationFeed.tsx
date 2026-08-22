import { ActivityStatus } from "./ActivityStatus";
import { FinalReport } from "./FinalReport";
import { Markdown } from "./Markdown";
import { StatusBanner } from "./StatusBanner";
import type { ResearchStreamState, TranscriptEntry } from "../types";

function UserBubble({ question }: { question: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%] rounded-2xl rounded-tr-sm bg-indigo-600 px-4 py-2.5 text-sm text-white">
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
    <div className="mx-auto max-w-6xl space-y-5 px-6 py-8">
      {state.transcript.map((entry, index) => (
        <TranscriptTurn key={entry.id} entry={entry} isLast={index === state.transcript.length - 1} running={running} />
      ))}
    </div>
  );
}
