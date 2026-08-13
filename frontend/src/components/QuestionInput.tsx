import { useState, type FormEvent } from "react";

interface Props {
  onSubmit: (question: string) => void;
  disabled: boolean;
  placeholder?: string;
  submitLabel?: string;
}

export function QuestionInput({ onSubmit, disabled, placeholder, submitLabel }: Props) {
  const [value, setValue] = useState("");

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSubmit(trimmed);
    setValue("");
  };

  return (
    <form onSubmit={submit} className="flex gap-2">
      <input
        className="flex-1 rounded-lg border border-slate-300 px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:bg-slate-100"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder={placeholder ?? 'Ask about a stock, e.g. "Compare NVIDIA and AMD"'}
        disabled={disabled}
      />
      <button
        type="submit"
        disabled={disabled || !value.trim()}
        className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-indigo-500 disabled:opacity-40"
      >
        {disabled ? "Researching…" : (submitLabel ?? "Ask")}
      </button>
    </form>
  );
}
