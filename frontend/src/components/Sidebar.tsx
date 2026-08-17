import type { ChatSummary } from "../lib/history";

interface Props {
  history: ChatSummary[];
  activeSessionId: string | null;
  onNewChat: () => void;
  onSelectChat: (sessionId: string) => void;
  onDeleteChat: (sessionId: string) => void;
}

export function Sidebar({ history, activeSessionId, onNewChat, onSelectChat, onDeleteChat }: Props) {
  return (
    <aside className="flex h-full w-64 shrink-0 flex-col border-r border-slate-800/80 bg-slate-950/60 backdrop-blur">
      <div className="flex items-center gap-2 px-4 py-4">
        <div className="flex h-7 w-7 items-center justify-center rounded-md bg-gradient-to-br from-indigo-500 to-violet-600 text-sm font-bold text-white">
          Σ
        </div>
        <span className="text-sm font-semibold tracking-wide text-slate-200">EquityLens</span>
      </div>

      <div className="px-3">
        <button
          onClick={onNewChat}
          className="flex w-full items-center gap-2 rounded-lg border border-slate-700/80 px-3 py-2 text-sm font-medium text-slate-200 transition hover:border-slate-600 hover:bg-slate-800/60"
        >
          <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4">
            <path d="M10 4v12M4 10h12" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
          </svg>
          New chat
        </button>
      </div>

      <div className="mt-4 flex-1 overflow-y-auto px-3 pb-4">
        <p className="px-1 pb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
          Recent
        </p>
        {history.length === 0 && (
          <p className="px-1 text-xs text-slate-600">Your research sessions will show up here.</p>
        )}
        <ul className="space-y-0.5">
          {history.map((chat) => (
            <li key={chat.sessionId} className="group relative">
              <button
                onClick={() => onSelectChat(chat.sessionId)}
                className={`block w-full truncate rounded-lg px-3 py-2 pr-8 text-left text-sm transition ${
                  chat.sessionId === activeSessionId
                    ? "bg-slate-800 text-slate-100"
                    : "text-slate-400 hover:bg-slate-800/60 hover:text-slate-200"
                }`}
                title={chat.title}
              >
                {chat.title || "New chat"}
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteChat(chat.sessionId);
                }}
                aria-label="Delete chat"
                className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded-md p-1.5 text-slate-500 opacity-0 transition hover:bg-slate-700 hover:text-slate-200 group-hover:opacity-100"
              >
                <svg viewBox="0 0 20 20" fill="none" className="h-3.5 w-3.5">
                  <path
                    d="M5 6h10M8.5 6V4.5h3V6M6 6l.5 9.5A1 1 0 0 0 7.5 16.5h5a1 1 0 0 0 1-1L14 6"
                    stroke="currentColor"
                    strokeWidth="1.4"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="border-t border-slate-800/80 px-4 py-3 text-[11px] text-slate-600">
        Not investment advice. Research only.
      </div>
    </aside>
  );
}
