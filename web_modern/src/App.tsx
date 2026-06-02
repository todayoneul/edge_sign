/**
 * App.tsx — 앱 루트 레이아웃
 * header / main(stage-left + rail-placeholder-right) / footer
 * T6: 셸+뷰포트+클라 캡처 모드①
 */

import { useEffect, useRef, useState } from "react";
import { useTheme } from "./hooks/useTheme";
import { useHotkeys } from "./hooks/useHotkeys";
import { useStore } from "./store";
import { getStatus } from "./lib/api";
import Header from "./components/Header";
import Viewport from "./components/Viewport";
import Splash from "./components/Splash";

// Rail will be fully built in T8; minimal placeholder here.
function RailPlaceholder() {
  return (
    <aside
      id="rail"
      aria-label="결과 및 어시스턴트"
      style={{
        width: 380,
        display: "flex",
        flexDirection: "column",
        borderLeft: "1px solid var(--line)",
        background: "var(--surface)",
        minHeight: 0,
        overflowY: "auto",
      }}
    >
      <div
        style={{
          margin: "auto",
          padding: "var(--sp-8) var(--sp-5)",
          textAlign: "center",
          color: "var(--ink-3)",
          fontFamily: "var(--f-mono)",
          fontSize: "0.72rem",
          letterSpacing: "0.02em",
          lineHeight: 1.8,
        }}
      >
        Rail (T8)
      </div>
    </aside>
  );
}

function Footer() {
  const telemetry = useStore((s) => s.telemetry);
  const tracks = useStore((s) => s.tracks);
  const connected = useStore((s) => s.connected);

  return (
    <footer id="footer">
      <span id="frame-info">frame —</span>
      <span id="time-info">
        추론 {telemetry.inferenceMs > 0 ? `${telemetry.inferenceMs.toFixed(0)} ms` : "— ms"}
      </span>
      <span id="track-count">tracks {tracks.length}</span>
      <span id="ws-info">{connected ? "WS 연결됨" : "WS 미연결"}</span>
      <span className="ftr-keys">
        <span className="kbd">Space</span> 재생/정지 ·{" "}
        <span className="kbd">/</span> 질문 ·{" "}
        <span className="kbd">?</span> 단축키
      </span>
      <span className="ftr-model">
        YOLOv8n-INT8 · OCR-INT8 · ByteTrack
      </span>
    </footer>
  );
}

export default function App() {
  const { toggle: toggleTheme } = useTheme();
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [splashVisible, setSplashVisible] = useState(true);
  const chatInputRef = useRef<HTMLInputElement | null>(null);

  // getStatus on mount — update store / local state
  useEffect(() => {
    getStatus()
      .then(() => {
        // Status available; splash will be dismissed once WS connects.
      })
      .catch(() => {
        // Server not reachable — still dismiss splash after timeout
      });
    // Fallback: hide splash max 2.5s (app.js pattern)
    const t = setTimeout(() => setSplashVisible(false), 2500);
    return () => clearTimeout(t);
  }, []);

  // Dismiss splash when WS connects
  const connected = useStore((s) => s.connected);
  useEffect(() => {
    if (connected) setSplashVisible(false);
  }, [connected]);

  useHotkeys({
    onTogglePlay: () => {
      // Viewport owns play/pause; bubbled via document — Viewport listens via its own toggle
      // Dispatch a synthetic space to let Viewport's internal handler pick it up.
      // Viewport registers its own handler via useHotkeys; we pass noop here to avoid double handling.
    },
    onFocusChat: () => {
      // Will be wired to chat input in T8; noop for now.
      chatInputRef.current?.focus();
    },
    onToggleTheme: toggleTheme,
    onToggleShortcuts: (open) => {
      setShortcutsOpen((prev) => (open !== undefined ? open : !prev));
    },
    onStepBack: () => {
      // Will be fully wired in T7/T8; noop for now.
    },
    onStepForward: () => {
      // Will be fully wired in T7/T8; noop for now.
    },
  });

  return (
    <>
      {splashVisible && <Splash />}

      <Header onToggleTheme={toggleTheme} />

      <main style={{ flex: 1, display: "flex", overflow: "hidden", minHeight: 0 }}>
        {/* 좌: 스테이지 */}
        <section
          id="stage"
          aria-label="영상 뷰포트"
          style={{
            flex: 1.5,
            display: "flex",
            flexDirection: "column",
            gap: "var(--sp-4)",
            padding: "var(--sp-5)",
            minWidth: 0,
          }}
        >
          <Viewport />
        </section>

        {/* 우: 레일 (T8에서 완성) */}
        <RailPlaceholder />
      </main>

      <Footer />

      {/* 단축키 모달 — T8에서 완성, 기본 구조만 */}
      {shortcutsOpen && (
        <div
          id="shortcuts"
          className="open"
          role="dialog"
          aria-modal="true"
          aria-labelledby="sc-title"
          onClick={(e) => {
            if (e.target === e.currentTarget) setShortcutsOpen(false);
          }}
        >
          <div className="modal-card">
            <div className="modal-head">
              <h3 id="sc-title">키보드 단축키</h3>
              <button
                className="icon-btn"
                aria-label="닫기"
                onClick={() => setShortcutsOpen(false)}
              >
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={2}
                  strokeLinecap="round"
                >
                  <path d="M6 6l12 12M18 6 6 18" />
                </svg>
              </button>
            </div>
            <div className="modal-body">
              <div className="kb-row">
                <span>재생 / 일시정지</span>
                <span className="kbd">Space</span>
              </div>
              <div className="kb-row">
                <span>질문 입력으로 이동</span>
                <span className="kbd">/</span>
              </div>
              <div className="kb-row">
                <span>5초 뒤로 / 앞으로</span>
                <span>
                  <span className="kbd">←</span> <span className="kbd">→</span>
                </span>
              </div>
              <div className="kb-row">
                <span>테마 전환</span>
                <span className="kbd">T</span>
              </div>
              <div className="kb-row">
                <span>이 도움말 열기 / 닫기</span>
                <span>
                  <span className="kbd">?</span> <span className="kbd">Esc</span>
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 토스트 컨테이너 (T8에서 활용) */}
      <div id="toast-wrap" aria-live="polite" aria-atomic="false" />
    </>
  );
}
