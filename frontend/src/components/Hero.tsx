import { QuestionInput } from "./QuestionInput";

const FEATURES = [
  {
    icon: (
      <path
        d="M4 15l4-5 3 3 5-7 4 5"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    ),
    title: "Single-stock analysis",
    body: "Ask about any company and get fundamentals, technicals, and news researched independently, then merged into one cited report.",
  },
  {
    icon: (
      <path
        d="M4 5h5v5H4V5zm7 0h5v5h-5V5zM4 12h5v5H4v-5zm7 0h5v5h-5v-5z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
    ),
    title: "Portfolio & comparison",
    body: "Roll up a whole portfolio or compare tickers side-by-side — valuation, momentum, and sentiment, laid out in structured tables.",
  },
  {
    icon: (
      <path
        d="M10 2v4M10 14v4M2 10h4M14 10h4M5 5l2.5 2.5M12.5 12.5L15 15M15 5l-2.5 2.5M7.5 12.5L5 15"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    ),
    title: "Live agent progress",
    body: "Watch each specialist agent work in real time, then ask grounded follow-up questions against the same session.",
  },
];

export function Hero({ onSubmit, disabled }: { onSubmit: (q: string) => void; disabled: boolean }) {
  return (
    <div className="mx-auto flex min-h-full max-w-3xl flex-col items-center justify-center px-4 py-16 text-center">
      <span className="mb-6 inline-flex items-center gap-1.5 rounded-full border border-slate-700 bg-slate-900/70 px-3 py-1 text-xs font-medium text-slate-400">
        Multi-agent research · live &amp; cited
      </span>

      <h1 className="bg-gradient-to-b from-white to-slate-400 bg-clip-text text-4xl font-bold leading-tight text-transparent sm:text-5xl">
        EquityLens
      </h1>
      <p className="mt-2 text-sm font-medium text-slate-400 sm:text-base">
        Your AI-powered stock research assistant
      </p>

      <p className="mx-auto mt-4 max-w-xl text-sm leading-relaxed text-slate-400 sm:text-base">
        Specialist agents research fundamentals, technicals, and news/sentiment independently using
        real market data, then synthesize it into one cited report — with the process visible live
        and every claim traceable to a source. Not investment advice.
      </p>

      <div className="relative mt-10 w-full max-w-2xl">
        <div className="pointer-events-none absolute -inset-x-10 -top-10 -bottom-4 -z-10 rounded-full bg-indigo-500/10 blur-3xl" />
        <QuestionInput onSubmit={onSubmit} disabled={disabled} variant="hero" />
        <p className="mt-3 text-xs text-slate-500">
          Try: "Analyze NVIDIA" · "Compare NVIDIA and AMD" · "Analyze my portfolio of NVIDIA, Apple and
          Microsoft"
        </p>
      </div>

      <div className="mt-14 grid w-full gap-4 sm:grid-cols-3">
        {FEATURES.map((f) => (
          <div
            key={f.title}
            className="rounded-xl border border-slate-800 bg-slate-900/50 p-4 text-left transition hover:border-slate-700 hover:bg-slate-900"
          >
            <div className="mb-3 flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-500/10 text-indigo-400">
              <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4">
                {f.icon}
              </svg>
            </div>
            <h3 className="text-sm font-semibold text-slate-200">{f.title}</h3>
            <p className="mt-1.5 text-xs leading-relaxed text-slate-500">{f.body}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
