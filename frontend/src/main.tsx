import { Analytics } from "@vercel/analytics/react";
import { SpeedInsights } from "@vercel/speed-insights/react";
import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
import "./index.css";

// Two separate Vercel products, deliberately both here rather than inside App: each
// renders nothing, and App's tree stays about the research UI.
//
// `Analytics` counts page views and visitors. `SpeedInsights` reports Core Web Vitals
// measured in real browsers, which is the half that matters for this app — the hero and
// the conversation feed both load a backdrop image and three webfonts, so LCP and CLS are
// the numbers worth watching.
//
// Both no-op outside a Vercel deployment: in `npm run dev` they log to the console instead
// of sending, so neither needs an env guard. Each still has to be switched on separately
// in the Vercel dashboard (Analytics and Speed Insights are distinct tabs); until then the
// injected script 404s and nothing is collected.
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
    <Analytics />
    <SpeedInsights />
  </React.StrictMode>,
);
