import { useCallback, useEffect, useState } from "react";

export type Theme = "light" | "dark";
const KEY = "edge-sign-theme";

function initial(): Theme {
  const saved = localStorage.getItem(KEY);
  return saved === "dark" || saved === "light" ? saved : "light";
}
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(initial);
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(KEY, theme);
  }, [theme]);
  const toggle = useCallback(() => setTheme((t) => (t === "light" ? "dark" : "light")), []);
  return { theme, toggle };
}
