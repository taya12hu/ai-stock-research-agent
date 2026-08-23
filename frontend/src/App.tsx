import { useEffect, useRef, useState } from "react";

import { BackgroundDecor } from "./components/BackgroundDecor";
import { ConversationFeed } from "./components/ConversationFeed";
import { Hero } from "./components/Hero";
import { Logo } from "./components/Logo";
import { QuestionInput } from "./components/QuestionInput";
import { Sidebar } from "./components/Sidebar";
import { Wordmark } from "./components/Wordmark";
import { useResearchStream } from "./hooks/useResearchStream";
import { CHROME } from "./lib/surfaces";

export default function App() {
  const { sessionId, state, history, start, ask, newChat, loadChat, removeChat } = useResearchStream();
  const running = state.status === "running";
  const hasStarted = state.status !== "idle";
  const bottomRef = useRef<HTMLDivElement>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [state]);

  // Escape closes the drawer. Cheap to add and it is the one affordance people reach for
  // when a panel covers the screen; the scrim handles pointer dismissal.
  useEffect(() => {
    if (!sidebarOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSidebarOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [sidebarOpen]);

  return (
    <div className="app-shell flex overflow-hidden bg-transparent">
      <BackgroundDecor />

      {/* Scrim: only ever present below `md`, and only while the drawer is open. It also
          stops taps landing on the conversation underneath. */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-ink-950/70 backdrop-blur-sm md:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      <Sidebar
        history={history}
        activeSessionId={sessionId}
        onNewChat={newChat}
        onSelectChat={loadChat}
        onDeleteChat={removeChat}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        {/* Mobile-only top bar. It carries the menu trigger, which has nowhere else to
            live once the sidebar is off-canvas. */}
        <header
          className={`flex shrink-0 items-center gap-3 border-b border-ink-800/60 px-4 py-3 md:hidden ${CHROME}`}
        >
          <button
            onClick={() => setSidebarOpen(true)}
            aria-label="Open menu"
            aria-expanded={sidebarOpen}
            className="rounded-md p-1.5 text-ink-300 transition hover:bg-ink-800/70 hover:text-ink-100"
          >
            <svg viewBox="0 0 20 20" fill="none" className="h-5 w-5">
              <path d="M3 5.5h14M3 10h14M3 14.5h14" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
            </svg>
          </button>
          {/* Mark and word are one lockup, so they get their own tighter gap than the
              header's spacing between controls. */}
          <div className="flex items-center gap-2">
            <Logo className="h-6 w-6 shrink-0" />
            <Wordmark className="text-sm text-ink-100" />
          </div>
          {hasStarted && (
            <button
              onClick={newChat}
              aria-label="New chat"
              className="ml-auto rounded-md p-1.5 text-ink-300 transition hover:bg-ink-800/70 hover:text-ink-100"
            >
              <svg viewBox="0 0 20 20" fill="none" className="h-5 w-5">
                <path d="M10 4v12M4 10h12" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
              </svg>
            </button>
          )}
        </header>

        {hasStarted ? (
          <>
            <main className="min-h-0 flex-1 overflow-y-auto">
              <ConversationFeed state={state} />
              <div ref={bottomRef} />
            </main>
            {/* Spacing only: no background, no border. The input carries its own surface,
                and wrapping it in a second one made it read as a box inside a box. Nothing
                scrolls underneath it either — this is a flex sibling of <main>, not an
                overlay — so there is nothing for a backdrop to hide.
                pb-[env(safe-area-inset-bottom)] keeps it clear of a phone's home
                indicator, which otherwise sits on top of the send button. */}
            <div className="shrink-0 px-4 pb-[max(0.75rem,env(safe-area-inset-bottom))] pt-2 sm:px-6 sm:pb-4">
              <div className="mx-auto max-w-6xl">
                <QuestionInput
                  onSubmit={ask}
                  disabled={running}
                  placeholder='Ask a follow-up, e.g. "Any fresh news today?"'
                  variant="bar"
                />
              </div>
            </div>
          </>
        ) : (
          <main className="min-h-0 flex-1 overflow-y-auto">
            <Hero onSubmit={start} disabled={running} />
          </main>
        )}
      </div>
    </div>
  );
}
