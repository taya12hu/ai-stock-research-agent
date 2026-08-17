import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function FinalReport({ markdown }: { markdown: string }) {
  return (
    <article className="prose prose-sm prose-invert max-w-none rounded-xl border border-slate-800 bg-slate-900/50 p-6 prose-headings:font-semibold prose-headings:text-slate-100 prose-p:text-slate-300 prose-strong:text-slate-100 prose-a:text-indigo-400 prose-blockquote:border-slate-700 prose-blockquote:text-slate-400 prose-hr:border-slate-800 prose-li:text-slate-300">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({ children }) => (
            <div className="my-4 overflow-x-auto rounded-lg border border-slate-800">
              <table className="w-full border-collapse text-sm">{children}</table>
            </div>
          ),
          thead: ({ children }) => <thead className="bg-slate-800/70">{children}</thead>,
          th: ({ children }) => (
            <th className="border-b border-slate-800 px-3 py-2 text-left font-semibold text-slate-300">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="border-b border-slate-800/60 px-3 py-2 align-top text-slate-400">{children}</td>
          ),
          tr: ({ children }) => <tr className="last:[&>td]:border-b-0">{children}</tr>,
          code: ({ children }) => (
            <code className="rounded bg-slate-800 px-1 py-0.5 text-[0.85em] text-indigo-300">{children}</code>
          ),
        }}
      >
        {markdown}
      </ReactMarkdown>
    </article>
  );
}
