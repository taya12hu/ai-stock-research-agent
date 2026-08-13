import { useCallback, useRef, useState } from "react";

import { askFollowUp, startResearch, streamUrl } from "../api/client";
import { EVENT_TYPES, type AgentName, type ResearchEvent, type ResearchStreamState, type TickerAgents, type TranscriptEntry } from "../types";

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
  const eventSourceRef = useRef<EventSource | null>(null);
  const nextTranscriptId = useRef(0);

  const closeStream = useCallback(() => {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
  }, []);

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
      const entry: TranscriptEntry = {
        id: nextTranscriptId.current++,
        question,
        answer: null,
        isReportUpdate: false,
      };
      setState({ ...initialState, status: "running", transcript: [entry] });
      const { session_id } = await startResearch(question);
      setSessionId(session_id);
      openStream(session_id);
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
      await askFollowUp(sessionId, question);
      openStream(sessionId);
    },
    [sessionId, openStream],
  );

  return { sessionId, state, start, ask };
}
