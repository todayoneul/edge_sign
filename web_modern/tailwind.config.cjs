/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: ["selector", '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        sign: "var(--c-sign)", light: "var(--c-light)", board: "var(--c-board)",
        ink: "var(--ink)", "ink-2": "var(--ink-2)", "ink-3": "var(--ink-3)",
        surface: "var(--surface)", line: "var(--line)",
      },
      fontFamily: { sans: ["Fira Sans","Pretendard","system-ui","sans-serif"], mono: ["Fira Code","ui-monospace","monospace"] },
      borderRadius: { sm: "8px", DEFAULT: "12px", lg: "16px" },
    },
  },
}
