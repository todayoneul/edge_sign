/**
 * useStream — 클라 캡처 모드① (/ws/stream)
 *
 * 백엔드 계약 (app.js / src/pipeline/app.py ws_stream):
 *  - 송신: JSON { type: "frame", data: "<base64 data-URL JPEG>", variant?: string }
 *  - 수신: JSON { type: "result", data: FrameResult }
 *         JSON { type: "error",  message: string }
 *  - reset: JSON { type: "reset" } — 트랙 ID 시퀀스 초기화
 *
 * DOM 캡처(canvas.toDataURL)는 Viewport(T6)가 담당.
 * 이 hook은 소켓 생명주기 + store 업데이트만 처리.
 */

import { useCallback, useEffect, useRef } from "react";
import { wsBase } from "../lib/api";
import { useStore } from "../store";

interface SendFrameOpts {
  /** base64 data-URL (image/jpeg) — Viewport가 canvas.toDataURL("image/jpeg", 0.8)로 생성 */
  data: string;
  variant?: string | null;
}

/**
 * @param fps  서버 전송 프레임레이트 (기본 10). Viewport가 setInterval로 getFrame()을 호출.
 */
export function useStream(fps = 10) {
  const wsRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const setConnected = useStore((s) => s.setConnected);
  const setFrame = useStore((s) => s.setFrame);
  const reset = useStore((s) => s.reset);

  /** WebSocket 열기 + 이벤트 연결 */
  const connect = useCallback(() => {
    if (wsRef.current) return; // 이미 연결 중
    const ws = new WebSocket(`${wsBase()}/ws/stream`);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      useStore.setState({ sourceKind: "stream" });
    };
    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
      // 3초 후 재연결 (app.js 패턴)
      setTimeout(connect, 3000);
    };
    ws.onerror = () => {
      // onerror 뒤에 onclose가 호출되므로 여기서는 별도 처리 불필요
    };
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data as string);
        if (msg.type === "result") setFrame(msg.data);
        else if (msg.type === "error") console.warn("[useStream]", msg.message);
      } catch {
        // 파싱 실패 무시
      }
    };
  }, [setConnected, setFrame]);

  /** 프레임 전송 (Viewport 콜백에서 base64 JPEG를 받아 WS 전송) */
  const sendFrame = useCallback((opts: SendFrameOpts) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const payload: Record<string, unknown> = { type: "frame", data: opts.data };
    if (opts.variant) payload.variant = opts.variant;
    ws.send(JSON.stringify(payload));
  }, []);

  /** reset 명령 전송 → store reset (app.js resetPipeline) */
  const sendReset = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "reset" }));
    reset();
  }, [reset]);

  /**
   * start — Viewport가 getFrame()을 주입하면 interval로 sendFrame 호출.
   * getFrame은 canvas.toDataURL("image/jpeg", 0.8)을 반환하는 함수.
   * @param getFrame () => { data: string; variant?: string | null } | null
   */
  const start = useCallback(
    (getFrame: () => SendFrameOpts | null) => {
      if (timerRef.current) clearInterval(timerRef.current);
      useStore.setState({ playing: true });
      timerRef.current = setInterval(() => {
        const frame = getFrame();
        if (frame) sendFrame(frame);
      }, Math.round(1000 / fps));
    },
    [fps, sendFrame],
  );

  /** stop — interval 정지 */
  const stop = useCallback(() => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    useStore.setState({ playing: false });
  }, []);

  /** reset — stop + store reset */
  const resetStream = useCallback(() => {
    stop();
    sendReset();
  }, [stop, sendReset]);

  // 컴포넌트 마운트 시 자동 연결
  useEffect(() => {
    connect();
    return () => {
      // 언마운트 시 타이머만 정리; WS는 재연결 루프를 위해 열어 둠
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [connect]);

  return { sendFrame, start, stop, reset: resetStream, sendReset };
}
