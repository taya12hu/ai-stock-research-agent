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

// Separator between the Buy/Sell/Hold keyword and the rationale: \p{Pd} is the Unicode
// "Dash Punctuation" category — every dash variant (hyphen, en dash, em dash, minus
// sign, ...) in one property, not a hand-picked list of the ones seen so far. A
// hand-picked list is exactly what broke last time (only "—" and "-" were covered; the
// LLM wrote "–" and it fell through, leaving a stray leading dash on the rationale) —
// same bug shape as the ticker-length cap, fixed the same way: stop guessing which
// characters to allow and use the actual rule.
// Comma included alongside the dash category because `normalizePunctuation` below rewrites a
// spaced dash to ", " before this ever runs — so the separator this pattern most often
// meets is now a comma, not a dash. The dash forms stay for reports rendered from
// history, and for the case where the model ignores the style instruction.
const SEPARATOR = String.raw`[\s.:,\p{Pd}]*`;
const PLAIN_VERDICT = new RegExp(String.raw`^Verdict:\s*(Buy|Sell|Hold)\b${SEPARATOR}(.*)$`, "iu");
// No upper bound on ticker length: what actually rules out a false positive is the
// character class (all-caps/digits/dots only — no spaces, so it can never match a
// stray phrase) plus the immediate ": Buy/Sell/Hold" right after it, not a length
// number. A cap here doesn't add real protection — it was previously set to 10 chars
// (tuned to short US symbols) and silently broke on "HDFCBANK.NS" (11 chars); it would
// break again on the next longer one. Uncapped is the actually-correct fix, not a
// bigger guess.
const PER_TICKER_VERDICT = new RegExp(String.raw`^([A-Z][A-Z0-9.]+):\s*(Buy|Sell|Hold)\b${SEPARATOR}(.*)$`, "u");

function parseVerdictLine(text: string): VerdictMatch | null {
  const trimmed = text.trim();
  const plain = trimmed.match(PLAIN_VERDICT);
  if (plain) return { ticker: null, verdict: titleCaseVerdict(plain[1]), rationale: plain[2].trim() };
  const perTicker = trimmed.match(PER_TICKER_VERDICT);
  if (perTicker) return { ticker: perTicker[1], verdict: titleCaseVerdict(perTicker[2]), rationale: perTicker[3].trim() };
  return null;
}

function VerdictCallout({ match }: { match: VerdictMatch }) {
  // The colored pill alone ("HOLD") doesn't say what it IS out of context — the plain
  // "Verdict: Hold" prose it replaces did. A per-ticker line (comparison/portfolio)
  // already self-identifies via the ticker; the single-verdict line has nothing else
  // labeling it, so it needs its own "Verdict" tag or it reads as an unexplained badge.
  return (
    <div className="my-2 flex flex-wrap items-baseline gap-x-2 gap-y-1">
      {!match.ticker && (
        <span className="shrink-0 text-xs font-semibold uppercase tracking-wide text-ink-500">Verdict</span>
      )}
      <span
        className={`shrink-0 rounded-full border px-2.5 py-0.5 text-xs font-bold uppercase tracking-wide ${VERDICT_STYLES[match.verdict]}`}
      >
        {match.ticker ? `${match.ticker} · ${match.verdict}` : match.verdict}
      </span>
      {match.rationale && <span className="text-sm leading-relaxed text-ink-300">{match.rationale}</span>}
    </div>
  );
}

function isVerdictHeading(text: string): boolean {
  return /^verdicts?$/i.test(text.trim());
}

// Every synthesis report appends a deterministic "**Sources**" block after the LLM's
// own text (see _synthesis_shared.py's sources_section) — never inside it, so splitting
// on the first occurrence is safe. Reference material, not something worth pushing the
// report's own conclusion down the page for, so it renders collapsed by default.
const SOURCES_MARKER = "**Sources**";

function splitSources(markdown: string): { body: string; sources: string | null } {
  const idx = markdown.indexOf(SOURCES_MARKER);
  if (idx === -1) return { body: markdown, sources: null };
  return { body: markdown.slice(0, idx).trimEnd(), sources: markdown.slice(idx + SOURCES_MARKER.length).trim() };
}

function countSources(sources: string): number {
  return (sources.match(/^-\s/gm) ?? []).length;
}

const components = {
  table: ({ children }: { children?: React.ReactNode }) => (
    <div className="my-4 overflow-x-auto rounded-lg border border-ink-800">
      <table className="w-full border-collapse text-sm">{children}</table>
    </div>
  ),
  thead: ({ children }: { children?: React.ReactNode }) => (
    <thead className="bg-ink-800/70">{children}</thead>
  ),
  th: ({ children }: { children?: React.ReactNode }) => (
    <th className="border-b border-ink-800 px-3 py-2 text-left font-semibold text-ink-300">
      {children}
    </th>
  ),
  td: ({ children }: { children?: React.ReactNode }) => (
    <td className="border-b border-ink-800/60 px-3 py-2 align-top text-ink-400">{children}</td>
  ),
  tr: ({ children }: { children?: React.ReactNode }) => <tr className="last:[&>td]:border-b-0">{children}</tr>,
  code: ({ children }: { children?: React.ReactNode }) => (
    <code className="rounded bg-ink-800 px-1 py-0.5 text-[0.85em] text-blue-300">{children}</code>
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
      <h2 className="mb-1 mt-5 text-xs font-bold uppercase tracking-widest text-ink-500">{children}</h2>
    ) : (
      <h2>{children}</h2>
    ),
  h3: ({ children }: { children?: ReactNode }) =>
    isVerdictHeading(flattenText(children)) ? (
      <h3 className="mb-1 mt-5 text-xs font-bold uppercase tracking-widest text-ink-500">{children}</h3>
    ) : (
      <h3>{children}</h3>
    ),
};

// Backstop to the PROSE_STYLE instruction in the synthesis prompts. The instruction is
// the real fix (it changes what gets written); this catches the cases where the model
// ignores it, and repairs reports already saved to chat history from before that
// instruction existed.
//
// Dashes: only one with whitespace on BOTH sides is rewritten. That is the
// parenthetical-aside usage, which is what reads as machine-written. A tight dash is
// almost always a range ("2020–2024") or a compound, and turning those into commas would
// corrupt real data.
//
// Brackets: the model occasionally emits citations as 【id】 rather than [id]. Those are
// real citation ids, just wrongly punctuated, so folding them back into ASCII brackets
// renders them correctly instead of leaking full-width brackets into the report.
function normalizePunctuation(markdown: string): string {
  return markdown
    .replace(/ +[—–] +/g, ", ")
    .replace(/【/g, "[")
    .replace(/】/g, "]");
}

export function Markdown({ children, className = "" }: { children: string; className?: string }) {
  const { body, sources } = splitSources(normalizePunctuation(children));

  return (
    <div
      className={`prose prose-sm prose-invert max-w-none prose-headings:font-display prose-headings:font-semibold prose-headings:tracking-tight prose-headings:text-ink-100 prose-p:text-ink-300 prose-strong:text-ink-100 prose-a:text-blue-400 prose-blockquote:border-ink-700 prose-blockquote:text-ink-400 prose-hr:border-ink-800 prose-li:text-ink-300 ${className}`}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {body}
      </ReactMarkdown>

      {sources && (
        <details className="mt-4 rounded-lg border border-ink-800/80">
          <summary className="cursor-pointer select-none rounded-lg px-3 py-2 text-xs font-semibold uppercase tracking-wide text-ink-500 hover:text-ink-300">
            Sources ({countSources(sources)})
          </summary>
          <div className="border-t border-ink-800/80 px-3 py-2">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
              {sources}
            </ReactMarkdown>
          </div>
        </details>
      )}
    </div>
  );
}
