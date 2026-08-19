// Chat history is persisted client-side only. The backend has no multi-session
// listing endpoint (no auth/multi-user support by design, see ARCHITECTURE.md §12),
// so "previous chats" in the sidebar is a localStorage index over full state
// snapshots keyed by session id.
import type { ResearchStreamState } from "../types";

const INDEX_KEY = "stock-research:chats";
const CHAT_KEY_PREFIX = "stock-research:chat:";

export interface ChatSummary {
  sessionId: string;
  title: string;
  updatedAt: number;
}

interface StoredChat {
  state: ResearchStreamState;
  // How many of this session's SSE events have already been applied to `state` — needed
  // so resuming a saved chat and asking another follow-up doesn't replay the whole
  // session's event history again (see streamUrl's `afterId`).
  lastEventId: number;
}

function readIndex(): ChatSummary[] {
  try {
    const raw = localStorage.getItem(INDEX_KEY);
    return raw ? (JSON.parse(raw) as ChatSummary[]) : [];
  } catch {
    return [];
  }
}

function writeIndex(index: ChatSummary[]): void {
  localStorage.setItem(INDEX_KEY, JSON.stringify(index));
}

export function listChats(): ChatSummary[] {
  return readIndex().sort((a, b) => b.updatedAt - a.updatedAt);
}

export function saveChatState(
  sessionId: string,
  title: string,
  state: ResearchStreamState,
  lastEventId: number,
): void {
  try {
    const stored: StoredChat = { state, lastEventId };
    localStorage.setItem(`${CHAT_KEY_PREFIX}${sessionId}`, JSON.stringify(stored));
    const index = readIndex().filter((c) => c.sessionId !== sessionId);
    index.push({ sessionId, title, updatedAt: Date.now() });
    writeIndex(index);
  } catch {
    // Storage full/unavailable — history persistence is a nice-to-have, not critical.
  }
}

export function loadChatState(sessionId: string): StoredChat | null {
  try {
    const raw = localStorage.getItem(`${CHAT_KEY_PREFIX}${sessionId}`);
    return raw ? (JSON.parse(raw) as StoredChat) : null;
  } catch {
    return null;
  }
}

export function deleteChat(sessionId: string): void {
  localStorage.removeItem(`${CHAT_KEY_PREFIX}${sessionId}`);
  writeIndex(readIndex().filter((c) => c.sessionId !== sessionId));
}

export function titleFromQuestion(question: string): string {
  const trimmed = question.trim().replace(/\s+/g, " ");
  return trimmed.length > 48 ? `${trimmed.slice(0, 48)}…` : trimmed;
}
