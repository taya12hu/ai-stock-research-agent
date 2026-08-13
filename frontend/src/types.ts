// Mirrors backend/app/streaming/events.py and backend/app/graph/state.py.
// Kept as plain hand-written types (not codegen) — small, stable surface for this scope.

export type AgentName = "fundamentals" | "technical" | "news";
export type QueryType = "single" | "portfolio" | "comparison";
export type AgentStatus = "queued" | "running" | "ok" | "failed";

export interface Source {
  type: "yahoo_finance" | "web";
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
  | (BaseEvent<"followup_classified"> & { path: string; tickers: string[]; notes: string[] })
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
  "followup_classified",
  "agent_started",
  "agent_completed",
  "report_ready",
  "followup_answer_ready",
  "run_completed",
  "run_failed",
];

export type RunStatus = "idle" | "running" | "done" | "error";

export interface TranscriptEntry {
  id: number;
  question: string;
  answer: string | null;
  isReportUpdate: boolean;
}

export interface ResearchStreamState {
  status: RunStatus;
  queryType: QueryType | null;
  tickers: string[];
  notes: string[];
  agents: Record<string, TickerAgents>;
  finalReport: string | null;
  error: string | null;
  transcript: TranscriptEntry[];
}
