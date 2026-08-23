// Mirrors backend/app/streaming/events.py and backend/app/graph/state.py.
// Kept as plain hand-written types (not codegen) — small, stable surface for this scope.

export type AgentName = "fundamentals" | "technical" | "news";
export type QueryType = "single" | "portfolio" | "comparison";
// No "queued": the backend publishes `agent_started` for every dispatched cell up front,
// so an agent is either running or settled. There is no observable pending state, and
// inventing one produced placeholder cards for analyses the turn never requested.
export type AgentStatus = "running" | "ok" | "failed";

export interface Source {
  type: "market_data" | "web";
  label: string;
  url: string | null;
  as_of: string;
}

export interface Finding {
  id: string;
  claim: string;
  evidence: string;
  source: Source;
}

export interface AgentState {
  status: AgentStatus;
  summary?: string | null;
  findings?: Finding[];
  error?: string | null;
}

export type TickerAgents = Partial<Record<AgentName, AgentState>>;

interface BaseEvent<T extends string> {
  type: T;
}

export type ResearchEvent =
  | BaseEvent<"run_started">
  | (BaseEvent<"router_completed"> & { query_type: QueryType; tickers: string[]; notes: string[] })
  | (BaseEvent<"agent_started"> & { ticker: string; agent: AgentName })
  | (BaseEvent<"agent_completed"> & {
      ticker: string;
      agent: AgentName;
      status: "ok" | "failed";
      summary: string | null;
      findings: Finding[];
      error: string | null;
    })
  | (BaseEvent<"report_ready"> & { final_report: string })
  | (BaseEvent<"followup_answer_ready"> & { answer: string })
  | BaseEvent<"run_completed">
  | (BaseEvent<"run_failed"> & { error: string });

export const EVENT_TYPES: ResearchEvent["type"][] = [
  "run_started",
  "router_completed",
  "agent_started",
  "agent_completed",
  "report_ready",
  "followup_answer_ready",
  "run_completed",
  "run_failed",
];

export type RunStatus = "idle" | "running" | "done" | "error";

// Every field below except {id, question} is populated by events scoped to THIS turn
// (see useResearchStream's applyEvent, which always targets the last transcript entry)
// — so a turn's own progress/cards/answer render immediately after its own question,
// never in a shared slot that visually detaches from whichever message triggered it.
export interface TranscriptEntry {
  id: number;
  question: string;
  answer: string | null;
  report: string | null;
  queryType: QueryType | null;
  tickers: string[];
  // Keyed by ticker, then by agent. Only the cells this turn dispatched ever appear here,
  // because entries are created by `agent_started` — which makes this the authoritative
  // answer to "what is this turn running", and the set both the progress steps and the
  // ticker cards render from.
  agents: Record<string, TickerAgents>;
  notes: string[];
  error: string | null;
  // Set once the router/followup_router has classified this turn (regardless of
  // outcome) — the signal the activity indicator uses to know classification is done,
  // since queryType/tickers/notes can all legitimately stay empty for some outcomes
  // (off-topic, no tickers resolved) and so can't be used for that alone.
  classified: boolean;
}

export interface ResearchStreamState {
  status: RunStatus;
  transcript: TranscriptEntry[];
}
