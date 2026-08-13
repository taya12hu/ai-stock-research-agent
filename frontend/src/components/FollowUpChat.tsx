import { QuestionInput } from "./QuestionInput";
import type { TranscriptEntry } from "../types";

interface Props {
  transcript: TranscriptEntry[];
  onAsk: (question: string) => void;
  running: boolean;
}

export function FollowUpChat({ transcript, onAsk, running }: Props) {
  // The very first turn's answer is the initial report itself, already shown above —
  // only show follow-up turns (turn index >= 1) in the chat log.
  const followUps = transcript.slice(1);

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <h3 className="mb-3 text-sm font-semibold text-slate-700">Follow-up questions</h3>

      {followUps.length > 0 && (
        <ul className="mb-4 space-y-3">
          {followUps.map((entry) => (
            <li key={entry.id} className="space-y-1">
              <p className="text-sm font-medium text-slate-800">{entry.question}</p>
              {entry.answer && <p className="text-sm text-slate-600">{entry.answer}</p>}
              {entry.isReportUpdate && (
                <p className="text-xs italic text-indigo-600">↑ updated the report above</p>
              )}
              {!entry.answer && !entry.isReportUpdate && (
                <p className="text-xs italic text-slate-400">thinking…</p>
              )}
            </li>
          ))}
        </ul>
      )}

      <QuestionInput
        onSubmit={onAsk}
        disabled={running}
        placeholder="Ask a follow-up, e.g. &quot;Any fresh news today?&quot;"
        submitLabel="Send"
      />
    </div>
  );
}
