/**
 * useSession — 서버 인제스트 모드② (/api/ingest → /ws/session)
 *
 * 백엔드 계약 (app.js / src/pipeline/app.py ws_session):
 *  - ingest: POST /api/ingest (FormData: kind, file | url) → { session_id?, error? }
 *  - WS /ws/session
 *      수신 JSON { type: "frame", frame_id, tracks, inference_ms, variant?,
 *                  w, h, pos, total, fps, seekable }
 *      수신 JSON { type: "ended" }
 *      수신 JSON { type: "error", message }
 *      수신 binary (ArrayBuffer) → JPEG 프레임 이미지
 *      송신 JSON { type: "control", action, value? }
 *        actions: "play" | "pause" | "stop" | "seek" (value=frameIdx) | "speed" (value=rate) | "variant" (value=name)
 *
 * 서버가 JPEG에 박스를 직접 그리지 않음 — 클라가 drawOverlay로 렌더(app.js 패턴).
 * 이 hook은 소켓 + store 업데이트 + blob URL 관리를 담당.
 * DOM(img 엘리먼트)은 Viewport(T7)가 담당; frameBlobUrl을 ref로 노출.
 */

import { useCallback, useRef, useState } from "react";
import { ingest as apiIngest, wsBase } from "../lib/api";
import { useStore } from "../store";

interface SessionFrameMsg {
  type: "frame";
  frame_id?: number;
  tracks: import("../lib/types").Track[];
  inference_ms?: number;
  variant?: string;
  w?: number;
  h?: number;
  pos?: number;
  total?: number;
  fps?: number;
  seekable?: boolean;
  stage_ms?: import("../lib/types").StageMs;
}

export interface SeekInfo {
  pos: number;        // 현재 프레임 위치
  total: number;      // 총 프레임 수
  fps: number;        // 소스 FPS
  seekable: boolean;
}

export function useSession() {
  const wsRef = useRef<WebSocket | null>(null);
  const blobUrlRef = useRef<string | null>(null);

  const [frameBlobUrl, setFrameBlobUrl] = useState<string | null>(null);
  const [seekInfo, setSeekInfo] = useState<SeekInfo>({ pos: 0, total: 0, fps: 30, seekable: false });
  const [serverPlaying, setServerPlaying] = useState(false);
  const [ended, setEnded] = useState(false);

  const setFrame = useStore((s) => s.setFrame);
  const reset = useStore((s) => s.reset);

  /** 제어 명령 송신 (app.js sessionControl) */
  const control = useCallback(
    (action: "play" | "pause" | "stop" | "seek" | "speed" | "variant", value?: number | string) => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const msg: Record<string, unknown> = { type: "control", action };
      if (value !== undefined) msg.value = value;
      ws.send(JSON.stringify(msg));
    },
    [],
  );

  /** WS 열기 (ingest 성공 후 호출) */
  const openSessionWS = useCallback(
    (label: string) => {
      if (wsRef.current) {
        try { wsRef.current.close(); } catch { /* ignore */ }
        wsRef.current = null;
      }
      if (blobUrlRef.current) { URL.revokeObjectURL(blobUrlRef.current); blobUrlRef.current = null; }

      useStore.setState({ sourceKind: "session", playing: true });
      setServerPlaying(true);
      setEnded(false);

      const ws = new WebSocket(`${wsBase()}/ws/session`);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onmessage = (e) => {
        if (typeof e.data === "string") {
          try {
            const msg = JSON.parse(e.data as string);
            if (msg.type === "frame") {
              const fm = msg as SessionFrameMsg;
              // store 업데이트 (FPS·추론ms·트랙)
              setFrame({
                frame_id: fm.frame_id ?? 0,
                inference_ms: fm.inference_ms ?? 0,
                tracks: fm.tracks ?? [],
                variant: fm.variant,
                stage_ms: fm.stage_ms,
              });
              // seekbar 정보 갱신
              if (fm.total != null) {
                setSeekInfo({
                  pos: fm.pos ?? fm.frame_id ?? 0,
                  total: fm.total,
                  fps: fm.fps ?? 30,
                  seekable: !!fm.seekable,
                });
              }
            } else if (msg.type === "ended") {
              setEnded(true);
              setServerPlaying(false);
              useStore.setState({ playing: false });
            } else if (msg.type === "error") {
              console.warn("[useSession]", msg.message);
            }
          } catch {
            // 파싱 실패 무시
          }
        } else {
          // binary JPEG → blob URL (app.js startServerStream onmessage binary 분기)
          const blob = new Blob([e.data as ArrayBuffer], { type: "image/jpeg" });
          if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
          blobUrlRef.current = URL.createObjectURL(blob);
          setFrameBlobUrl(blobUrlRef.current);
        }
      };

      ws.onclose = () => {
        if (useStore.getState().sourceKind === "session") {
          useStore.setState({ playing: false });
        }
      };

      console.info(`[useSession] opened for: ${label}`);
    },
    [setFrame],
  );

  /**
   * ingest — POST /api/ingest → sessionId → WS 연결
   * @param form FormData containing: kind ("image"|"video"|"url"), file OR url
   */
  const ingest = useCallback(
    async (form: FormData) => {
      const label = (form.get("url") as string) || (form.get("file") as File)?.name || "입력";
      reset();
      const data = await apiIngest(form);
      if (data.error) {
        console.error("[useSession] ingest error:", data.error);
        return;
      }
      openSessionWS(label);
    },
    [reset, openSessionWS],
  );

  /** stop — WS 종료 + blob URL 해제 */
  const stop = useCallback(() => {
    if (wsRef.current) {
      control("stop");
      try { wsRef.current.close(); } catch { /* ignore */ }
      wsRef.current = null;
    }
    if (blobUrlRef.current) { URL.revokeObjectURL(blobUrlRef.current); blobUrlRef.current = null; }
    setFrameBlobUrl(null);
    setServerPlaying(false);
    useStore.setState({ sourceKind: "none", playing: false });
  }, [control]);

  /** play/pause 토글 (app.js togglePlay 서버 분기) */
  const togglePlay = useCallback(() => {
    const next = !serverPlaying;
    setServerPlaying(next);
    control(next ? "play" : "pause");
    useStore.setState({ playing: next });
  }, [serverPlaying, control]);

  /** seek to frame index */
  const seek = useCallback((frameIdx: number) => control("seek", frameIdx), [control]);

  /** playback speed */
  const setSpeed = useCallback((rate: number) => control("speed", rate), [control]);

  /** variant 전환 */
  const setVariant = useCallback((name: string) => control("variant", name), [control]);

  return {
    ingest,
    stop,
    togglePlay,
    seek,
    setSpeed,
    setVariant,
    control,
    frameBlobUrl,
    seekInfo,
    serverPlaying,
    ended,
  };
}
