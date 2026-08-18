import { FinalReport } from "./FinalReport";
import { StatusBanner } from "./StatusBanner";
import { TickerGroup } from "./TickerGroup";
import type { ResearchStreamState } from "../types";

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

export function ConversationFeed({ state }: { state: ResearchStreamState }) {
  const [firstTurn, ...followUps] = state.transcript;
  const resolvingTickers = state.status === "running" && state.tickers.length === 0;
  const researching = state.status === "running" && state.tickers.length > 0 && !state.finalReport;

  return (
    <div className="mx-auto max-w-6xl space-y-5 px-6 py-8">
      {firstTurn && <UserBubble question={firstTurn.question} />}

      <StatusBanner notes={state.notes} error={state.error} />

      {resolvingTickers && (
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-indigo-400" />
          Classifying request and resolving tickers…
        </div>
      )}

      {state.queryType && (
        <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
          {QUERY_TYPE_LABELS[state.queryType] ?? state.queryType}
          {state.tickers.length > 0 && (
            <span className="text-slate-400"> — {state.tickers.join(", ")}</span>
          )}
        </p>
      )}

      {state.tickers.map((ticker) => (
        <TickerGroup key={ticker} ticker={ticker} agents={state.agents[ticker] ?? {}} />
      ))}

      {researching && (
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-indigo-400" />
          Researching {state.tickers.join(", ")}…
        </div>
      )}

      {state.finalReport && <FinalReport markdown={state.finalReport} />}

      {followUps.map((entry, i) => {
        const isLastFollowUp = i === followUps.length - 1;
        const isPending = !entry.answer && !entry.isReportUpdate;
        return (
          <div key={entry.id} className="space-y-3">
            <UserBubble question={entry.question} />
            {entry.answer && <p className="text-sm leading-relaxed text-slate-300">{entry.answer}</p>}
            {entry.isReportUpdate && !entry.answer && (
              <p className="text-xs italic text-indigo-400">↑ Updated the report above with fresh data</p>
            )}
            {isPending && isLastFollowUp && state.status === "running" && (
              <div className="flex items-center gap-2 text-sm text-slate-500">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-indigo-400" />
                Thinking…
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
