import { Markdown } from "./Markdown";

export function FinalReport({ markdown }: { markdown: string }) {
  return (
    <article className="rounded-xl border border-slate-800 bg-slate-900/50 p-6">
      <Markdown>{markdown}</Markdown>
    </article>
  );
}
