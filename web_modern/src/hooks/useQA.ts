import { useState, useCallback } from "react";
import { askQA } from "../lib/api";
import { useStore } from "../store";
import type { Track } from "../lib/types";

export function useQA() {
  const [answer, setAnswer] = useState("");
  const [streaming, setStreaming] = useState(false);
  const byok = useStore((s) => s.byokKey);

  const ask = useCallback(
    async (tracks: Track[], q: string) => {
      setAnswer("");
      setStreaming(true);
      await askQA(tracks, q, byok || null, (e) => {
        if (e.type === "token") setAnswer((a) => a + e.text);
      });
      setStreaming(false);
    },
    [byok],
  );

  return { answer, streaming, ask };
}
