// The product name, set once. "Lens" carries the blue gradient in every place the name
// appears — hero, sidebar, mobile header — so the brand reads the same everywhere rather
// than being a styled treatment on the home page and plain text in the chrome.
//
// Size and colour of the "Equity" half come from the caller via `className`; only the
// gradient half is fixed, because that is the part that must not drift.
export function Wordmark({ className = "" }: { className?: string }) {
  return (
    <span className={`font-display font-semibold tracking-tight ${className}`}>
      Equity
      <span className="bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent">
        Lens
      </span>
    </span>
  );
}
