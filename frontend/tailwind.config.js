import typography from "@tailwindcss/typography";

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        // Full system fallbacks on every stack: the webfonts come off the network, and a
        // `display=swap` gap that falls back to Times looks broken rather than plain.
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        display: ["Space Grotesk", "Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
      colors: {
        // The app's surface + text ramp. Replaces Tailwind's `slate`, which is a
        // blue-grey and read as a cold, slightly muddy layer over a tinted backdrop:
        // greys sitting on a coloured field look dirty rather than neutral.
        //
        // Same 100-950 steps as `slate` so the substitution is one-for-one at every call
        // site, but every step is a true navy, matching <BackgroundDecor />'s field. The
        // low end is deliberately near-black (950 is #06080f, not a mid-grey) because the
        // reference artwork is very dark and mid-tone surfaces float off it.
        //
        // `ink-950` is exactly the darkest stop of <BackgroundDecor />'s gradient and the
        // body background colour, so translucent chrome layered over it stays in-family.
        ink: {
          100: "#e7eaf4",
          200: "#c6cde1",
          300: "#a0accb",
          400: "#7887b4",
          500: "#56679a",
          600: "#3a4b78",
          700: "#24345e",
          800: "#182448",
          850: "#101830",
          900: "#0b1122",
          950: "#06080f",
        },
      },
    },
  },
  plugins: [typography],
};
