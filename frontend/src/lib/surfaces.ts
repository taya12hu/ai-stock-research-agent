// Shared surface treatments for anything that floats over <BackgroundDecor />.
//
// The backdrop is an illustration, not a flat colour: line art, candlesticks and ticker
// numbers run right through the middle of the content column. A panel that is merely
// tinted (the earlier `bg-ink-900/30`, no blur) let all of that read straight through the
// text, which is what made the research card unreadable while the report — which happened
// to have a blur — stayed legible.
//
// Both now use the same token, so they cannot drift apart again. The blur is the
// load-bearing part: opacity alone would need to be so high the panel stops feeling
// layered, whereas blurring destroys the high-frequency line art specifically, which is
// what competes with letterforms.
export const PANEL = "border border-ink-800/80 bg-ink-900/80 backdrop-blur-md";

// For chrome that should still show the artwork through it (sidebar, input bar) — same
// family, much lighter touch.
export const CHROME = "bg-ink-950/70 backdrop-blur-md";
