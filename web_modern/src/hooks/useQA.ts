import { useState, useCallback } from "react";
import { askQA } from "../lib/api";
import { useStore } from "../store";
import type { Track } from "../lib/types";

export function useQA() {
  const [answer, setAnswer] = useState("");
  const [streaming, setStreaming] = useState(false);
  const byok = useStore((s) => s.byokKey);

  /**
   * ask — Q&A 요청.
   * @param tracks  현재 트랙 배열
   * @param q       질문 문자열
   * @param onToken (선택) 각 토큰을 받을 때마다 호출되는 콜백.
   *                제공하면 내부 answer 상태도 누적, 콜백도 호출됨.
   */
  const ask = useCallback(
    async (tracks: Track[], q: string, onToken?: (chunk: string) => void) => {
      setAnswer("");
      setStreaming(true);
      await askQA(tracks, q, byok || null, (e) => {
        if (e.type === "token") {
          setAnswer((a) => a + e.text);
          onToken?.(e.text);
        }
      });
      setStreaming(false);
    },
    [byok],
  );

  return { answer, streaming, ask };
}
