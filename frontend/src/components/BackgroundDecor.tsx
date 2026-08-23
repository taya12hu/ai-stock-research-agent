// The app's ambient backdrop.
//
// Served from `public/` rather than imported, so it stays a plain file request the
// browser can cache independently of the JS bundle, instead of being fingerprinted into
// it and re-downloaded on every deploy.
//
// WebP, not the original PNG: the artwork is mostly flat gradient, which WebP encodes at
// 33KB against the PNG's 1.8MB for no visible difference at this scale. The source PNG is
// kept out of the build at `frontend/design-assets/bg-source.png`.
//
// `background-size: cover` with a right-anchored position is deliberate: the artwork
// keeps its subject matter (bull, candles, newspaper, bear) massed on the right and its
// left half nearly empty, which is exactly where the sidebar and the conversation column
// sit. Anchoring right means narrow viewports crop the empty side and keep the art,
// instead of cropping the art and keeping the emptiness.
//
// `bg-ink-950` underneath matches the image's darkest corner, so there is no flash of a
// different colour while it loads and no seam if the aspect ratio leaves any edge.
export function BackgroundDecor() {
  return (
    <div
      className="pointer-events-none fixed inset-0 -z-10 bg-ink-950 bg-cover bg-right bg-no-repeat"
      style={{ backgroundImage: "url('/bg.webp')" }}
      aria-hidden="true"
    />
  );
}
