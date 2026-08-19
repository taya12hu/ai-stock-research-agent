import { FinalReport } from "./FinalReport";
import { Markdown } from "./Markdown";
import { StatusBanner } from "./StatusBanner";
import { TickerGroup } from "./TickerGroup";
import type { ResearchStreamState, TranscriptEntry } from "../types";

const QUERY_TYPE_LABELS: Record<string, string> = {
  single: "Single-stock analysis",
  portfolio: "Portfolio analysis",
  comparison: "Comparison",
};

function UserBubble({ question }: { question: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%] rounded-2xl rounded-tr-sm bg-indigo-600 px-4 py-2.5 text-sm text-white">
        {question}
      </div>
    </div>
  );
}

// Everything for one turn — its progress, ticker cards, and eventual answer/report —
// renders together, in this order, immediately after that turn's own question. Nothing
// here reads from outside `entry`, so a turn's cards can never appear detached from (or
// stuck above) the message that actually triggered them.
function TranscriptTurn({
  entry,
  isFirst,
  isLast,
  running,
}: {
  entry: TranscriptEntry;
  isFirst: boolean;
  isLast: boolean;
  running: boolean;
}) {
  const active = isLast && running;
  // Still being classified: no tickers resolved yet, and no plain-reply answer either
  // (a clarification question, off-topic/discovery reply, etc. also lands as `answer`
  // with zero tickers — the guard stops the spinner the instant either arrives).
  const awaitingClassification = active && entry.tickers.length === 0 && !entry.answer && !entry.report;
  const researching = active && entry.tickers.length > 0 && !entry.report;

  return (
    <div className="space-y-3">
      <UserBubble question={entry.question} />
      {entry.answer && <Markdown>{entry.answer}</Markdown>}

      <StatusBanner notes={entry.notes} error={entry.error} />

      {awaitingClassification && (
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-indigo-400" />
          {isFirst ? "Classifying request and resolving tickers…" : "Thinking…"}
        </div>
      )}

      {entry.queryType && entry.tickers.length > 0 && (
        <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
          {QUERY_TYPE_LABELS[entry.queryType] ?? entry.queryType}
          <span className="text-slate-400"> — {entry.tickers.join(", ")}</span>
        </p>
      )}

      {entry.tickers.map((ticker) => (
        <TickerGroup key={ticker} ticker={ticker} agents={entry.agents[ticker] ?? {}} />
      ))}

      {researching && (
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-indigo-400" />
          Researching {entry.tickers.join(", ")}…
        </div>
      )}

      {entry.report && <FinalReport markdown={entry.report} />}
    </div>
  );
}

export function ConversationFeed({ state }: { state: ResearchStreamState }) {
  const running = state.status === "running";

  return (
    <div className="mx-auto max-w-6xl space-y-5 px-6 py-8">
      {state.transcript.map((entry, index) => (
        <TranscriptTurn
          key={entry.id}
          entry={entry}
          isFirst={index === 0}
          isLast={index === state.transcript.length - 1}
          running={running}
        />
      ))}
    </div>
  );
}
