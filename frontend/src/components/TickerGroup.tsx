import { AgentCard } from "./AgentCard";
import type { AgentName, TickerAgents } from "../types";

const AGENTS: AgentName[] = ["fundamentals", "technical", "news"];

export function TickerGroup({ ticker, agents }: { ticker: string; agents: TickerAgents }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <h3 className="mb-3 text-base font-bold text-slate-800">{ticker}</h3>
      <div className="grid gap-3 sm:grid-cols-3">
        {AGENTS.map((agent) => (
          <AgentCard key={agent} agent={agent} state={agents[agent]} />
        ))}
      </div>
    </div>
  );
}
