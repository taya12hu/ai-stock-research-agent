import { Logo } from "./Logo";
import { QuestionInput } from "./QuestionInput";
import { Wordmark } from "./Wordmark";

// Clickable rather than quoted in a "Try:" line — an example the user can run is worth
// more than an example they have to retype, and it removes the preamble sentence too.
const EXAMPLES = [
  "Analyze NVIDIA",
  "Compare NVIDIA and AMD",
  "Analyze my portfolio of NVIDIA, Apple and Microsoft",
];

export function Hero({ onSubmit, disabled }: { onSubmit: (q: string) => void; disabled: boolean }) {
  return (
    <div className="mx-auto flex min-h-full max-w-2xl flex-col items-center justify-center px-4 py-10 text-center sm:py-16">
      {/* The mark gets its own line and a soft halo rather than sitting inline with the
          word: at hero scale an inline lockup makes the icon read as a bullet point. */}
      <div className="relative mb-5">
        <div className="absolute inset-0 -z-10 rounded-2xl bg-blue-500/20 blur-2xl" aria-hidden="true" />
        <Logo className="h-14 w-14 drop-shadow-[0_4px_18px_rgba(37,99,235,0.45)] sm:h-16 sm:w-16" />
      </div>

      <h1 className="leading-none">
        <Wordmark className="text-4xl text-ink-100 sm:text-6xl" />
      </h1>

      <p className="mt-4 max-w-md text-sm leading-relaxed text-ink-400 sm:text-base">
        Stock research from fundamentals, technicals and news. Every claim cited.
      </p>

      <div className="mt-8 w-full">
        <QuestionInput onSubmit={onSubmit} disabled={disabled} variant="hero" />
      </div>

      <div className="mt-4 flex flex-wrap justify-center gap-2">
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => onSubmit(example)}
            disabled={disabled}
            className="rounded-full border border-ink-800 px-3 py-1.5 text-xs text-ink-400 transition hover:border-ink-700 hover:text-ink-200 disabled:opacity-50"
          >
            {example}
          </button>
        ))}
      </div>

      <p className="mt-10 text-xs text-ink-600">
        US-listed stocks only. Not investment advice.
      </p>
    </div>
  );
}
