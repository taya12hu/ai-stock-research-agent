import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, askFollowUp, startResearch, streamUrl } from "../api/client";
import { deleteChat, listChats, loadChatState, saveChatState, titleFromQuestion, type ChatSummary } from "../lib/history";
import { EVENT_TYPES, type AgentName, type ResearchEvent, type ResearchStreamState, type TickerAgents, type TranscriptEntry } from "../types";

// The request never reached the graph (network failure, or the backend rejected it
// before a run started), so there's no SSE stream to report a `run_failed` event —
// these are the only two failure shapes `start`/`ask` can actually produce, and each
// gets a message a user can act on rather than the raw fetch/HTTP error text.
function describeStartError(err: unknown): string {
  console.error("Failed to start research:", err);
  return "Something went wrong starting this research. Please try again.";
}

function describeAskError(err: unknown): string {
  if (err instanceof ApiError && err.status === 404) {
    return "This chat's research session has expired or could not be found. Start a new chat to continue.";
  }
  console.error("Failed to send follow-up:", err);
  return "Something went wrong sending that message. Please try again.";
}

const initialState: ResearchStreamState = {
  status: "idle",
  transcript: [],
};

function newEntry(id: number, question: string): TranscriptEntry {
  return {
    id, question, answer: null, report: null,
    queryType: null, tickers: [], agents: {}, notes: [], error: null, classified: false,
  };
}

function setAgent(
  agents: Record<string, TickerAgents>,
  ticker: string,
  agent: AgentName,
  patch: TickerAgents[AgentName],
): Record<string, TickerAgents> {
  return {
    ...agents,
    [ticker]: { ...(agents[ticker] ?? {}), [agent]: { ...(agents[ticker]?.[agent]), ...patch } },
  };
}

// Every event that carries turn-scoped data (progress, results, the final answer/report)
// targets the LAST transcript entry — the one currently in flight — rather than a
// shared top-level field. That's what makes a turn's cards render immediately after its
// own question: there's no separate global slot for them to get stuck in.
function updateLastEntry(
  transcript: TranscriptEntry[],
  update: (entry: TranscriptEntry) => TranscriptEntry,
): TranscriptEntry[] {
  if (transcript.length === 0) return transcript;
  const updated = [...transcript];
  const lastIndex = updated.length - 1;
  updated[lastIndex] = update(updated[lastIndex]);
  return updated;
}

function applyEvent(prev: ResearchStreamState, event: ResearchEvent): ResearchStreamState {
  switch (event.type) {
    case "run_started":
      return { ...prev, status: "running" };
    case "router_completed":
      return {
        ...prev,
        transcript: updateLastEntry(prev.transcript, (e) => ({
          ...e,
          classified: true,
          queryType: event.query_type,
          tickers: event.tickers,
          notes: [...e.notes, ...event.notes],
        })),
      };
    case "followup_classified":
      return {
        ...prev,
        // `tickers` matters here too, not just `notes` — this is the only event that
        // tells the frontend which tickers a follow-up is (re)researching, and without
        // setting it the step tracker (and the ticker/agent cards it renders) never
        // appear for a follow-up turn even though agents are actually running.
        transcript: updateLastEntry(prev.transcript, (e) => ({
          ...e, classified: true, tickers: event.tickers, notes: [...e.notes, ...event.notes],
        })),
      };
    case "agent_started":
      return {
        ...prev,
        transcript: updateLastEntry(prev.transcript, (e) => ({
          ...e, agents: setAgent(e.agents, event.ticker, event.agent, { status: "running" }),
        })),
      };
    case "agent_completed":
      return {
        ...prev,
        transcript: updateLastEntry(prev.transcript, (e) => ({
          ...e,
          agents: setAgent(e.agents, event.ticker, event.agent, {
            status: event.status,
            summary: event.summary,
            findings: event.findings,
            error: event.error,
          }),
        })),
      };
    case "report_ready":
      return {
        ...prev,
        transcript: updateLastEntry(prev.transcript, (e) => ({ ...e, report: event.final_report })),
      };
    case "followup_answer_ready":
      return {
        ...prev,
        transcript: updateLastEntry(prev.transcript, (e) => ({ ...e, answer: event.answer })),
      };
    case "run_completed":
      return { ...prev, status: "done" };
    case "run_failed":
      return {
        ...prev,
        status: "error",
        transcript: updateLastEntry(prev.transcript, (e) => ({ ...e, error: event.error })),
      };
    default:
      return prev;
  }
}

export function useResearchStream() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [state, setState] = useState<ResearchStreamState>(initialState);
  const [history, setHistory] = useState<ChatSummary[]>(() => listChats());
  const eventSourceRef = useRef<EventSource | null>(null);
  const nextTranscriptId = useRef(0);
  const titleRef = useRef<string>("");
  // How many of this session's SSE events have been applied so far. `openStream` is
  // called fresh (a brand-new EventSource, not a browser-level reconnect) at the start
  // of every turn, so without this the backend would replay the entire session's event
  // history — including prior turns' answers — on every single follow-up.
  const lastEventIdRef = useRef(0);
  // Set right before loadChat's setState so the effect below can tell "just opened an
  // existing chat" apart from "this session got new activity" — opening a chat must not
  // bump its position in the sidebar, only new messages should.
  const justLoadedRef = useRef(false);

  const closeStream = useCallback(() => {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
  }, []);

  // Persist a snapshot after every state change so the sidebar's chat list stays
  // in sync and a chat can be reopened later without re-hitting the backend.
  useEffect(() => {
    if (!sessionId || state.status === "idle") return;
    if (justLoadedRef.current) {
      justLoadedRef.current = false;
      return;
    }
    saveChatState(sessionId, titleRef.current || "New chat", state, lastEventIdRef.current);
    setHistory(listChats());
  }, [sessionId, state]);

  const openStream = useCallback(
    (sid: string) => {
      closeStream();
      const es = new EventSource(streamUrl(sid, lastEventIdRef.current));
      eventSourceRef.current = es;

      const handle = (raw: MessageEvent<string>) => {
        if (raw.lastEventId) lastEventIdRef.current = Number(raw.lastEventId);
        const event = JSON.parse(raw.data) as ResearchEvent;
        setState((prev) => applyEvent(prev, event));
        if (event.type === "run_completed" || event.type === "run_failed") {
          closeStream();
        }
      };

      for (const type of EVENT_TYPES) {
        es.addEventListener(type, handle as EventListener);
      }
    },
    [closeStream],
  );

  const start = useCallback(
    async (question: string) => {
      nextTranscriptId.current = 0;
      lastEventIdRef.current = 0;
      titleRef.current = titleFromQuestion(question);
      const entry = newEntry(nextTranscriptId.current++, question);
      setState({ status: "running", transcript: [entry] });
      try {
        const { session_id } = await startResearch(question);
        setSessionId(session_id);
        openStream(session_id);
      } catch (err) {
        const message = describeStartError(err);
        setState((prev) => ({
          ...prev,
          status: "error",
          transcript: updateLastEntry(prev.transcript, (e) => ({ ...e, error: message })),
        }));
      }
    },
    [openStream],
  );

  const ask = useCallback(
    async (question: string) => {
      if (!sessionId) return;
      const entry = newEntry(nextTranscriptId.current++, question);
      setState((prev) => ({
        ...prev,
        status: "running",
        transcript: [...prev.transcript, entry],
      }));
      try {
        await askFollowUp(sessionId, question);
        openStream(sessionId);
      } catch (err) {
        const message = describeAskError(err);
        setState((prev) => ({
          ...prev,
          status: "error",
          transcript: updateLastEntry(prev.transcript, (e) => ({ ...e, error: message })),
        }));
      }
    },
    [sessionId, openStream],
  );

  const newChat = useCallback(() => {
    closeStream();
    titleRef.current = "";
    nextTranscriptId.current = 0;
    lastEventIdRef.current = 0;
    setSessionId(null);
    setState(initialState);
  }, [closeStream]);

  const loadChat = useCallback(
    (sid: string) => {
      const saved = loadChatState(sid);
      if (!saved) return;
      closeStream();
      const { state: savedState, lastEventId } = saved;
      titleRef.current = savedState.transcript[0]?.question
        ? titleFromQuestion(savedState.transcript[0].question)
        : "";
      nextTranscriptId.current = savedState.transcript.length;
      lastEventIdRef.current = lastEventId;
      justLoadedRef.current = true;
      setSessionId(sid);
      setState({
        status: savedState.status === "running" ? "done" : savedState.status,
        // Defensive defaults for a chat saved before per-turn fields existed (this
        // session's own history, or an older one from before this change) — at runtime
        // an old chat's JSON can genuinely lack these keys despite what the type
        // claims, so a missing field just means that turn renders without cards,
        // never a crash.
        transcript: savedState.transcript.map((e) => ({
          id: e.id,
          question: e.question,
          answer: e.answer,
          report: e.report,
          queryType: e.queryType ?? null,
          tickers: e.tickers ?? [],
          agents: e.agents ?? {},
          notes: e.notes ?? [],
          error: e.error ?? null,
          // A chat saved before this field existed lacks it in its JSON; treat it as
          // classified since every prior-saved entry that got this far necessarily was.
          classified: e.classified ?? true,
        })),
      });
    },
    [closeStream],
  );

  const removeChat = useCallback(
    (sid: string) => {
      deleteChat(sid);
      setHistory(listChats());
      if (sid === sessionId) newChat();
    },
    [sessionId, newChat],
  );

  return { sessionId, state, history, start, ask, newChat, loadChat, removeChat };
}
