import { Logo } from "./Logo";
import { Wordmark } from "./Wordmark";
import { CHROME } from "../lib/surfaces";
import type { ChatSummary } from "../lib/history";

interface Props {
  history: ChatSummary[];
  activeSessionId: string | null;
  onNewChat: () => void;
  onSelectChat: (sessionId: string) => void;
  onDeleteChat: (sessionId: string) => void;
  // Drawer state, only meaningful below `md`. Above it the sidebar is a static column and
  // these are ignored, which is why there is one component rather than two: the content
  // and every handler are identical, only the container's positioning differs.
  open: boolean;
  onClose: () => void;
}

export function Sidebar({
  history,
  activeSessionId,
  onNewChat,
  onSelectChat,
  onDeleteChat,
  open,
  onClose,
}: Props) {
  // Selecting a chat or starting a new one should dismiss the drawer — on a phone the
  // sidebar covers the thing you just asked to see. Harmless above `md`, where `onClose`
  // sets a flag nothing reads.
  const select = (sessionId: string) => {
    onSelectChat(sessionId);
    onClose();
  };
  const newChat = () => {
    onNewChat();
    onClose();
  };

  return (
    <aside
      className={`fixed inset-y-0 left-0 z-40 flex h-full w-64 shrink-0 flex-col border-r border-ink-800/60 transition-transform duration-200 md:static md:z-auto md:translate-x-0 ${CHROME} ${
        open ? "translate-x-0" : "-translate-x-full"
      }`}
    >
      <div className="flex items-center gap-2 px-4 py-4">
        <Logo className="h-7 w-7 shrink-0" />
        <Wordmark className="text-[15px] text-ink-100" />
        <button
          onClick={onClose}
          aria-label="Close menu"
          className="ml-auto rounded-md p-1.5 text-ink-400 transition hover:bg-ink-800/70 hover:text-ink-100 md:hidden"
        >
          <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4">
            <path d="M5 5l10 10M15 5L5 15" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
          </svg>
        </button>
      </div>

      <div className="px-3">
        <button
          onClick={newChat}
          className="flex w-full items-center gap-2 rounded-lg border border-ink-700 px-3 py-2 text-sm font-medium text-ink-200 transition hover:border-blue-500/60 hover:bg-ink-800/70 hover:text-ink-100"
        >
          <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4">
            <path d="M10 4v12M4 10h12" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
          </svg>
          New chat
        </button>
      </div>

      <div className="mt-4 flex-1 overflow-y-auto px-3 pb-4">
        <p className="px-1 pb-2 text-[11px] font-semibold uppercase tracking-wider text-ink-500">
          Recent
        </p>
        {history.length === 0 && (
          <p className="px-1 text-xs text-ink-600">Your research sessions will show up here.</p>
        )}
        <ul className="space-y-0.5">
          {history.map((chat) => (
            <li key={chat.sessionId} className="group relative">
              <button
                onClick={() => select(chat.sessionId)}
                // The active row carries an indigo edge rather than only a lighter fill:
                // against a translucent sidebar over a moving gradient, a fill alone
                // shifts with whatever is behind it and stops reading as "selected".
                className={`block w-full truncate rounded-lg border-l-2 px-3 py-2 pr-8 text-left text-sm transition ${
                  chat.sessionId === activeSessionId
                    ? "border-blue-500 bg-ink-800/80 text-ink-100"
                    : "border-transparent text-ink-400 hover:bg-ink-800/50 hover:text-ink-200"
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
                className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded-md p-1.5 text-ink-500 opacity-0 transition hover:bg-rose-500/15 hover:text-rose-300 group-hover:opacity-100"
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

      <div className="border-t border-ink-800/60 px-4 py-3 text-[11px] text-ink-600">
        Not investment advice. Research only.
      </div>
    </aside>
  );
}
