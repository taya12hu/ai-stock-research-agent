import { AgentCard } from "./AgentCard";
import type { AgentName, TickerAgents } from "../types";

const AGENTS: AgentName[] = ["fundamentals", "technical", "news"];

export function TickerGroup({ ticker, agents }: { ticker: string; agents: TickerAgents }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
      <h3 className="mb-3 text-base font-bold text-slate-100">{ticker}</h3>
      <div className="grid gap-3 sm:grid-cols-3">
        {AGENTS.map((agent) => (
          <AgentCard key={agent} agent={agent} state={agents[agent]} />
        ))}
      </div>
    </div>
  );
}
