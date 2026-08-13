import { FinalReport } from "./components/FinalReport";
import { FollowUpChat } from "./components/FollowUpChat";
import { QuestionInput } from "./components/QuestionInput";
import { StatusBanner } from "./components/StatusBanner";
import { TickerGroup } from "./components/TickerGroup";
import { useResearchStream } from "./hooks/useResearchStream";

const QUERY_TYPE_LABELS: Record<string, string> = {
  single: "Single-stock analysis",
  portfolio: "Portfolio analysis",
  comparison: "Comparison",
};

export default function App() {
  const { state, start, ask } = useResearchStream();
  const running = state.status === "running";
  const hasStarted = state.status !== "idle";

  return (
    <div className="mx-auto max-w-4xl px-4 py-10">
      <header className="mb-8">
        <h1 className="text-2xl font-bold text-slate-900">AI Stock Research Assistant</h1>
        <p className="mt-1 text-sm text-slate-500">
          Fundamentals, technicals, and news/sentiment — researched independently and combined into
          one cited report. Not investment advice.
        </p>
      </header>

      {!hasStarted && (
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <QuestionInput onSubmit={start} disabled={running} />
          <p className="mt-3 text-xs text-slate-400">
            Try: "Analyze NVIDIA", "Compare NVIDIA and AMD", or "Analyze my portfolio of NVIDIA,
            Apple and Microsoft"
          </p>
        </div>
      )}

      {hasStarted && (
        <div className="space-y-6">
          <StatusBanner notes={state.notes} error={state.error} />

          {state.queryType && (
            <p className="text-sm text-slate-500">
              {QUERY_TYPE_LABELS[state.queryType] ?? state.queryType}
              {state.tickers.length > 0 && ` — ${state.tickers.join(", ")}`}
            </p>
          )}

          {state.tickers.map((ticker) => (
            <TickerGroup key={ticker} ticker={ticker} agents={state.agents[ticker] ?? {}} />
          ))}

          {state.finalReport && <FinalReport markdown={state.finalReport} />}

          {state.status !== "idle" && (
            <FollowUpChat transcript={state.transcript} onAsk={ask} running={running} />
          )}
        </div>
      )}
    </div>
  );
}
