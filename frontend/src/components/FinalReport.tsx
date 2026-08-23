import { PANEL } from "../lib/surfaces";
import { Markdown } from "./Markdown";

export function FinalReport({ markdown }: { markdown: string }) {
  return (
    <article className={`rounded-xl p-5 sm:p-6 ${PANEL}`}>
      <Markdown>{markdown}</Markdown>
    </article>
  );
}
