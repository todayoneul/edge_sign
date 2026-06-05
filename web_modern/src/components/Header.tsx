/**
 * Header.tsx — 브랜드 + KPI 클러스터 + 상태 필 + 테마 토글
 * web/detection/index.html <header> 포팅
 *
 * KPI: FPS 스파크라인, 추론 지연(ms), 활성 트랙, 누적 검출
 * 스파크라인은 최근 48개 FPS 값을 인라인 SVG polyline으로 렌더.
 */

import { useEffect, useRef, useState } from "react";
import { useStore } from "../store";

interface Props {
  onToggleTheme: () => void;
  onOpenShortcuts?: () => void;
}

// FPS 스파크라인 — Canvas 버전 (app.js drawSpark 포팅)
function FpsSpark({ history }: { history: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 40;
    const h = canvas.clientHeight || 16;
    if (canvas.width !== w * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (history.length < 2) return;

    const max = Math.max(15, ...history);
    const n = history.length;
    const style = getComputedStyle(document.documentElement);
    const stroke = style.getPropertyValue("--spark").trim() || "#71717a";
    const ink = style.getPropertyValue("--ink").trim() || "#fafafa";

    ctx.beginPath();
    history.forEach((v, i) => {
      const x = (i / (n - 1)) * w;
      const y = h - (v / max) * (h - 2) - 1;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.25;
    ctx.lineJoin = "round";
    ctx.stroke();

    // 끝점 강조
    const lx = w;
    const ly = h - (history[n - 1] / max) * (h - 2) - 1;
    ctx.beginPath();
    ctx.arc(lx - 1, ly, 1.6, 0, Math.PI * 2);
    ctx.fillStyle = ink;
    ctx.fill();
  }, [history]);

  return (
    <canvas
      ref={canvasRef}
      id="fps-spark"
      width={80}
      height={32}
      aria-hidden="true"
      style={{ width: 40, height: 16, display: "block", opacity: 0.85 }}
    />
  );
}

export default function Header({ onToggleTheme, onOpenShortcuts }: Props) {
  const telemetry = useStore((s) => s.telemetry);
  const tracks = useStore((s) => s.tracks);
  const totalDetections = useStore((s) => s.totalDetections);
  const connected = useStore((s) => s.connected);
  const pipelineMode = useStore((s) => s.pipelineMode);
  const variant = useStore((s) => s.telemetry.variant);
  const [fpsHistory, setFpsHistory] = useState<number[]>([]);

  const onDevice = pipelineMode === "ondevice";
  const engineLabel = onDevice
    ? variant?.includes("wasm")
      ? "온디바이스 · WASM"
      : "온디바이스 · WebGPU"
    : "서버 추론";

  // Accumulate FPS history for sparkline (app.js fpsHistory max 48)
  const prevFps = useRef(0);
  useEffect(() => {
    const fps = telemetry.fps;
    if (fps === prevFps.current) return;
    prevFps.current = fps;
    if (fps > 0) {
      setFpsHistory((h) => {
        const next = [...h, fps];
        return next.length > 48 ? next.slice(next.length - 48) : next;
      });
    }
  }, [telemetry.fps]);

  return (
    <header>
      {/* 브랜드 */}
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none">
            <path
              d="M12 2.5 21.5 12 12 21.5 2.5 12 12 2.5Z"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinejoin="round"
            />
            <circle cx="12" cy="12" r="3" fill="var(--c-sign)" />
          </svg>
        </span>
        <span className="brand-text">
          <h1>Edge&#8209;Sign Console</h1>
          <span className="sub">주행 인지 관제</span>
        </span>
      </div>

      {/* KPI 클러스터 */}
      <div className="kpis" role="group" aria-label="실시간 지표">
        <div className="kpi">
          <span className="kpi-label">처리 FPS</span>
          <div className="kpi-row">
            <span className="kpi-val" id="kpi-fps">
              {telemetry.fps > 0 ? telemetry.fps.toFixed(1) : "0.0"}
            </span>
            <FpsSpark history={fpsHistory} />
          </div>
        </div>
        <div className="kpi">
          <span className="kpi-label">추론 지연</span>
          <div className="kpi-row">
            <span className="kpi-val" id="kpi-lat">
              {telemetry.inferenceMs > 0 ? telemetry.inferenceMs.toFixed(0) : "—"}
            </span>
            <span className="kpi-unit">ms</span>
          </div>
        </div>
        <div className="kpi">
          <span className="kpi-label">활성 트랙</span>
          <div className="kpi-row">
            <span className="kpi-val" id="kpi-tracks">
              {tracks.length}
            </span>
          </div>
        </div>
        <div className="kpi">
          <span className="kpi-label">누적 검출</span>
          <div className="kpi-row">
            <span className="kpi-val" id="kpi-total">
              {totalDetections}
            </span>
          </div>
        </div>
      </div>

      {/* 오른쪽: 추론 엔진 + 상태 필 + 버튼들 */}
      <div className="header-right">
        {/* 추론 위치 인디케이터 — 서버 vs 브라우저 온디바이스(WebGPU) */}
        <span
          className={`engine-pill${onDevice ? " ondevice" : ""}`}
          title="현재 추론이 실행되는 위치"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
            <rect x="6" y="6" width="12" height="12" rx="2" />
            <path d="M9 1.5v3M15 1.5v3M9 19.5v3M15 19.5v3M1.5 9h3M1.5 15h3M19.5 9h3M19.5 15h3" />
          </svg>
          {engineLabel}
        </span>

        <span className="status-pill">
          <span
            id="status-dot"
            title="서버 연결 상태"
            className={connected ? "connected" : ""}
          />
          <span id="status-text">{connected ? "연결됨" : "대기"}</span>
        </span>

        {/* Phase 1 한글 손글씨 OCR 캔버스 데모 (별도 페이지 — /detection/ocr/) */}
        <a
          className="icon-btn"
          id="ocr-demo-link"
          href="ocr/"
          target="_blank"
          rel="noopener"
          title="한글 손글씨 OCR 데모 (Phase 1, 온디바이스 W8A8)"
          aria-label="한글 OCR 데모 새 탭으로 열기"
          style={{
            width: "auto",
            padding: "0 10px",
            gap: 6,
            display: "inline-flex",
            alignItems: "center",
            fontSize: "0.72rem",
            fontWeight: 600,
            textDecoration: "none",
          }}
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{ width: 16, height: 16 }}
          >
            <path d="M12 20h9" />
            <path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4 12.5-12.5Z" />
          </svg>
          OCR 데모
        </a>

        {/* 단축키 도움말 버튼 (T8에서 모달 연결 완성) */}
        <button
          className="icon-btn"
          id="help-btn"
          title="단축키 (?)"
          aria-label="단축키 도움말"
          onClick={onOpenShortcuts}
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="12" cy="12" r="9.5" />
            <path d="M9.2 9.3a2.8 2.8 0 0 1 5.4 1c0 1.9-2.6 2.2-2.6 4" />
            <circle cx="12" cy="17" r="0.6" fill="currentColor" />
          </svg>
        </button>

        {/* 테마 토글 */}
        <button
          className="icon-btn"
          id="theme-toggle"
          title="라이트/다크 전환"
          aria-label="테마 전환"
          onClick={onToggleTheme}
        >
          <svg
            className="sun"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
          >
            <circle cx="12" cy="12" r="4" />
            <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
          </svg>
          <svg
            className="moon"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8Z" />
          </svg>
        </button>
      </div>
    </header>
  );
}
