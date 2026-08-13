import ReactMarkdown from "react-markdown";

export function FinalReport({ markdown }: { markdown: string }) {
  return (
    <article className="prose prose-sm prose-slate max-w-none rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      <ReactMarkdown>{markdown}</ReactMarkdown>
    </article>
  );
}
