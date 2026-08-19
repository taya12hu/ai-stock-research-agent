import { isValidElement, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Verdict = "Buy" | "Sell" | "Hold";

const VERDICT_STYLES: Record<Verdict, string> = {
  Buy: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  Hold: "border-amber-500/30 bg-amber-500/10 text-amber-300",
  Sell: "border-rose-500/30 bg-rose-500/10 text-rose-300",
};

// Synthesis prompts (synthesis_single/portfolio/comparison, answer_from_context) are
// instructed to write a verdict as prose — a "Verdict: Buy/Sell/Hold" line, or, for
// comparisons, one "TICKER: Buy/Sell/Hold" line per stock. There's no structured field
// for it (it's LLM-written text like the rest of the report), so it's picked out of the
// rendered markdown by pattern rather than passed down as data.
function flattenText(node: ReactNode): string {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(flattenText).join("");
  if (isValidElement<{ children?: ReactNode }>(node)) return flattenText(node.props.children);
  return "";
}

function titleCaseVerdict(word: string): Verdict {
  return (word[0].toUpperCase() + word.slice(1).toLowerCase()) as Verdict;
}

interface VerdictMatch {
  ticker: string | null;
  verdict: Verdict;
  rationale: string;
}

const PLAIN_VERDICT = /^Verdict:\s*(Buy|Sell|Hold)\b[\s.:—-]*(.*)$/i;
// No upper bound on ticker length: what actually rules out a false positive is the
// character class (all-caps/digits/dots only — no spaces, so it can never match a
// stray phrase) plus the immediate ": Buy/Sell/Hold" right after it, not a length
// number. A cap here doesn't add real protection — it was previously set to 10 chars
// (tuned to short US symbols) and silently broke on "HDFCBANK.NS" (11 chars); it would
// break again on the next longer one. Uncapped is the actually-correct fix, not a
// bigger guess.
const PER_TICKER_VERDICT = /^([A-Z][A-Z0-9.]+):\s*(Buy|Sell|Hold)\b[\s.:—-]*(.*)$/;

function parseVerdictLine(text: string): VerdictMatch | null {
  const trimmed = text.trim();
  const plain = trimmed.match(PLAIN_VERDICT);
  if (plain) return { ticker: null, verdict: titleCaseVerdict(plain[1]), rationale: plain[2].trim() };
  const perTicker = trimmed.match(PER_TICKER_VERDICT);
  if (perTicker) return { ticker: perTicker[1], verdict: titleCaseVerdict(perTicker[2]), rationale: perTicker[3].trim() };
  return null;
}

function VerdictCallout({ match }: { match: VerdictMatch }) {
  return (
    <div className="my-2 flex flex-wrap items-baseline gap-x-2 gap-y-1">
      <span
        className={`shrink-0 rounded-full border px-2.5 py-0.5 text-xs font-bold uppercase tracking-wide ${VERDICT_STYLES[match.verdict]}`}
      >
        {match.ticker ? `${match.ticker} · ${match.verdict}` : match.verdict}
      </span>
      {match.rationale && <span className="text-sm leading-relaxed text-slate-300">{match.rationale}</span>}
    </div>
  );
}

function isVerdictHeading(text: string): boolean {
  return /^verdicts?$/i.test(text.trim());
}

const components = {
  table: ({ children }: { children?: React.ReactNode }) => (
    <div className="my-4 overflow-x-auto rounded-lg border border-slate-800">
      <table className="w-full border-collapse text-sm">{children}</table>
    </div>
  ),
  thead: ({ children }: { children?: React.ReactNode }) => (
    <thead className="bg-slate-800/70">{children}</thead>
  ),
  th: ({ children }: { children?: React.ReactNode }) => (
    <th className="border-b border-slate-800 px-3 py-2 text-left font-semibold text-slate-300">
      {children}
    </th>
  ),
  td: ({ children }: { children?: React.ReactNode }) => (
    <td className="border-b border-slate-800/60 px-3 py-2 align-top text-slate-400">{children}</td>
  ),
  tr: ({ children }: { children?: React.ReactNode }) => <tr className="last:[&>td]:border-b-0">{children}</tr>,
  code: ({ children }: { children?: React.ReactNode }) => (
    <code className="rounded bg-slate-800 px-1 py-0.5 text-[0.85em] text-indigo-300">{children}</code>
  ),
  p: ({ children }: { children?: ReactNode }) => {
    const match = parseVerdictLine(flattenText(children));
    return match ? <VerdictCallout match={match} /> : <p>{children}</p>;
  },
  li: ({ children }: { children?: ReactNode }) => {
    const match = parseVerdictLine(flattenText(children));
    return match ? (
      <li className="list-none pl-0 marker:content-none">
        <VerdictCallout match={match} />
      </li>
    ) : (
      <li>{children}</li>
    );
  },
  h2: ({ children }: { children?: ReactNode }) =>
    isVerdictHeading(flattenText(children)) ? (
      <h2 className="mb-1 mt-5 text-xs font-bold uppercase tracking-widest text-slate-500">{children}</h2>
    ) : (
      <h2>{children}</h2>
    ),
  h3: ({ children }: { children?: ReactNode }) =>
    isVerdictHeading(flattenText(children)) ? (
      <h3 className="mb-1 mt-5 text-xs font-bold uppercase tracking-widest text-slate-500">{children}</h3>
    ) : (
      <h3>{children}</h3>
    ),
};

export function Markdown({ children, className = "" }: { children: string; className?: string }) {
  return (
    <div
      className={`prose prose-sm prose-invert max-w-none prose-headings:font-semibold prose-headings:text-slate-100 prose-p:text-slate-300 prose-strong:text-slate-100 prose-a:text-indigo-400 prose-blockquote:border-slate-700 prose-blockquote:text-slate-400 prose-hr:border-slate-800 prose-li:text-slate-300 ${className}`}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {children}
      </ReactMarkdown>
    </div>
  );
}
