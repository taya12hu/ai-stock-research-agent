import { useEffect, useRef } from "react";

import { ConversationFeed } from "./components/ConversationFeed";
import { Hero } from "./components/Hero";
import { QuestionInput } from "./components/QuestionInput";
import { Sidebar } from "./components/Sidebar";
import { useResearchStream } from "./hooks/useResearchStream";

export default function App() {
  const { sessionId, state, history, start, ask, newChat, loadChat, removeChat } = useResearchStream();
  const running = state.status === "running";
  const hasStarted = state.status !== "idle";
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [state]);

  return (
    <div className="flex h-screen overflow-hidden bg-transparent">
      <Sidebar
        history={history}
        activeSessionId={sessionId}
        onNewChat={newChat}
        onSelectChat={loadChat}
        onDeleteChat={removeChat}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        {hasStarted ? (
          <>
            <main className="flex-1 overflow-y-auto">
              <ConversationFeed state={state} />
              <div ref={bottomRef} />
            </main>
            <div className="shrink-0 border-t border-slate-800/80 bg-slate-950/90 px-6 py-4 backdrop-blur">
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
          <main className="flex-1 overflow-y-auto">
            <Hero onSubmit={start} disabled={running} />
          </main>
        )}
      </div>
    </div>
  );
}
