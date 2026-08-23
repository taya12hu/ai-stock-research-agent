import { useEffect, useRef, useState, type KeyboardEvent } from "react";

interface Props {
  onSubmit: (question: string) => void;
  disabled: boolean;
  placeholder?: string;
  variant?: "hero" | "bar";
}

export function QuestionInput({ onSubmit, disabled, placeholder, variant = "bar" }: Props) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Focuses on mount (covers page load / this component becoming available after a
  // conditional-render swap) and again whenever the field re-enables after a run
  // finishes — a disabled textarea can't take focus, so mount alone would miss the
  // follow-up bar, which is always disabled at the instant it first mounts.
  useEffect(() => {
    if (!disabled) textareaRef.current?.focus();
  }, [disabled]);

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSubmit(trimmed);
    setValue("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const autoResize = (el: HTMLTextAreaElement) => {
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  };

  const isHero = variant === "hero";

  return (
    <div
      className={`w-full rounded-2xl border transition focus-within:border-blue-500/60 ${
        isHero
          ? "border-ink-700 bg-ink-900/60 shadow-[0_0_70px_-18px_rgba(37,99,235,0.55)] backdrop-blur"
          // The bar sits directly on the artwork now that its wrapper has no background of
          // its own, so it carries the full panel treatment: without the blur, candlestick
          // line art reads straight through the placeholder text.
          : "border-ink-700/80 bg-ink-900/80 backdrop-blur-md"
      }`}
    >
      <textarea
        ref={textareaRef}
        rows={1}
        value={value}
        onChange={(e) => {
          setValue(e.target.value);
          autoResize(e.target);
        }}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        placeholder={
          placeholder ?? 'Ask about a stock, e.g. "Compare NVIDIA and AMD" or "Analyze my portfolio of NVDA, AAPL, MSFT"'
        }
        className={`block w-full resize-none bg-transparent px-4 pt-3.5 text-sm text-ink-100 placeholder:text-ink-500 focus:outline-none disabled:opacity-50 ${
          isHero ? "min-h-[64px]" : "min-h-[24px]"
        }`}
      />
      <div className="flex items-center justify-between px-3 pb-2.5 pt-1">
        <span className="text-[11px] text-ink-500">Enter to send · Shift+Enter for a new line</span>
        <button
          onClick={submit}
          disabled={disabled || !value.trim()}
          aria-label="Send"
          className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-blue-600 to-indigo-700 text-white transition hover:from-blue-500 hover:to-indigo-600 disabled:cursor-not-allowed disabled:bg-ink-800 disabled:from-ink-800 disabled:to-ink-800 disabled:text-ink-600"
        >
          {disabled ? (
            <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4 animate-spin">
              <circle cx="10" cy="10" r="7.5" stroke="currentColor" strokeWidth="2" strokeOpacity="0.25" />
              <path d="M17.5 10a7.5 7.5 0 0 0-7.5-7.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
          ) : (
            <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4">
              <path
                d="M3 10l14-7-4.5 14-3-5.5L3 10z"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            </svg>
          )}
        </button>
      </div>
    </div>
  );
}
