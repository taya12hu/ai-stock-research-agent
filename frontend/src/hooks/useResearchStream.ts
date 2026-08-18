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
  queryType: null,
  tickers: [],
  notes: [],
  agents: {},
  finalReport: null,
  error: null,
  transcript: [],
};

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

function attachToLastEntry(
  transcript: TranscriptEntry[],
  patch: Partial<Pick<TranscriptEntry, "answer" | "isReportUpdate">>,
): TranscriptEntry[] {
  if (transcript.length === 0) return transcript;
  const updated = [...transcript];
  const lastIndex = updated.length - 1;
  updated[lastIndex] = { ...updated[lastIndex], ...patch };
  return updated;
}

function applyEvent(prev: ResearchStreamState, event: ResearchEvent): ResearchStreamState {
  switch (event.type) {
    case "run_started":
      return { ...prev, status: "running", error: null };
    case "router_completed":
      return {
        ...prev,
        queryType: event.query_type,
        tickers: event.tickers,
        notes: [...prev.notes, ...event.notes],
      };
    case "followup_classified":
      return { ...prev, notes: [...prev.notes, ...event.notes] };
    case "agent_started":
      return { ...prev, agents: setAgent(prev.agents, event.ticker, event.agent, { status: "running" }) };
    case "agent_completed":
      return {
        ...prev,
        agents: setAgent(prev.agents, event.ticker, event.agent, {
          status: event.status,
          summary: event.summary,
          findings: event.findings,
          error: event.error,
        }),
      };
    case "report_ready":
      return {
        ...prev,
        finalReport: event.final_report,
        transcript: attachToLastEntry(prev.transcript, { isReportUpdate: true }),
      };
    case "followup_answer_ready":
      return { ...prev, transcript: attachToLastEntry(prev.transcript, { answer: event.answer }) };
    case "run_completed":
      return { ...prev, status: "done" };
    case "run_failed":
      return { ...prev, status: "error", error: event.error };
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

  const closeStream = useCallback(() => {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
  }, []);

  // Persist a snapshot after every state change so the sidebar's chat list stays
  // in sync and a chat can be reopened later without re-hitting the backend.
  useEffect(() => {
    if (!sessionId || state.status === "idle") return;
    saveChatState(sessionId, titleRef.current || "New chat", state);
    setHistory(listChats());
  }, [sessionId, state]);

  const openStream = useCallback(
    (sid: string) => {
      closeStream();
      const es = new EventSource(streamUrl(sid));
      eventSourceRef.current = es;

      const handle = (raw: MessageEvent<string>) => {
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
      titleRef.current = titleFromQuestion(question);
      const entry: TranscriptEntry = {
        id: nextTranscriptId.current++,
        question,
        answer: null,
        isReportUpdate: false,
      };
      setState({ ...initialState, status: "running", transcript: [entry] });
      try {
        const { session_id } = await startResearch(question);
        setSessionId(session_id);
        openStream(session_id);
      } catch (err) {
        setState((prev) => ({ ...prev, status: "error", error: describeStartError(err) }));
      }
    },
    [openStream],
  );

  const ask = useCallback(
    async (question: string) => {
      if (!sessionId) return;
      const entry: TranscriptEntry = {
        id: nextTranscriptId.current++,
        question,
        answer: null,
        isReportUpdate: false,
      };
      setState((prev) => ({
        ...prev,
        status: "running",
        error: null,
        notes: [],
        transcript: [...prev.transcript, entry],
      }));
      try {
        await askFollowUp(sessionId, question);
        openStream(sessionId);
      } catch (err) {
        setState((prev) => ({ ...prev, status: "error", error: describeAskError(err) }));
      }
    },
    [sessionId, openStream],
  );

  const newChat = useCallback(() => {
    closeStream();
    titleRef.current = "";
    nextTranscriptId.current = 0;
    setSessionId(null);
    setState(initialState);
  }, [closeStream]);

  const loadChat = useCallback(
    (sid: string) => {
      const saved = loadChatState(sid);
      if (!saved) return;
      closeStream();
      titleRef.current = saved.transcript[0]?.question ? titleFromQuestion(saved.transcript[0].question) : "";
      nextTranscriptId.current = saved.transcript.length;
      setSessionId(sid);
      setState({ ...saved, status: saved.status === "running" ? "done" : saved.status });
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
