// The product mark: a rising trend line with a marker on the high point.
//
// Deliberately identical to `public/favicon.svg` — same geometry, same gradient stops. A
// browser tab, the sidebar and the hero are three places people meet the same product, and
// a mark that differs between them reads as three different apps. If this changes, change
// the favicon in the same commit.
//
// Sized entirely by `className` so callers pick the scale; the viewBox does the rest.
import { useId } from "react";

export function Logo({ className = "h-7 w-7" }: { className?: string }) {
  // Gradient ids must be unique per instance — two of these on one page (sidebar + mobile
  // header) sharing an id means the second silently reuses the first's gradient, and if
  // the first unmounts the second loses its fill entirely.
  //
  // `useId` rather than a random value because this runs on every render: a fresh random
  // id each time would rewrite the <defs> and the fill reference on every parent update.
  // The colons useId produces are legal in a URL fragment but not in a CSS selector, so
  // they are stripped in case this ever needs selecting.
  const id = `logo-${useId().replace(/:/g, "")}`;

  return (
    <svg viewBox="0 0 64 64" className={className} role="img" aria-label="EquityLens">
      <rect width="64" height="64" rx="14" fill="#0b1122" />
      <path
        d="M10 42 L24 28 L34 36 L54 14"
        fill="none"
        stroke={`url(#${id})`}
        strokeWidth="6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="54" cy="14" r="5.5" fill="#60a5fa" />
      <defs>
        <linearGradient id={id} x1="10" y1="42" x2="54" y2="14" gradientUnits="userSpaceOnUse">
          <stop stopColor="#2563eb" />
          <stop offset="1" stopColor="#60a5fa" />
        </linearGradient>
      </defs>
    </svg>
  );
}
