/**
 * App.tsx — 앱 루트 레이아웃 (T8 완성)
 * header / main(stage-left + Rail-right) / footer
 * T8: PerfStrip + Rail + Q&A + Toast + ShortcutsModal + 단축키 완전 연결
 */

import { useEffect, useRef, useState } from "react";
import { useTheme } from "./hooks/useTheme";
import { useHotkeys } from "./hooks/useHotkeys";
import { useStore } from "./store";
import { getStatus } from "./lib/api";
import Header from "./components/Header";
import Viewport from "./components/Viewport";
import Splash from "./components/Splash";
import Rail from "./components/Rail";
import Toast from "./components/Toast";
import ShortcutsModal from "./components/ShortcutsModal";

function Footer() {
  const telemetry = useStore((s) => s.telemetry);
  const tracks = useStore((s) => s.tracks);
  const connected = useStore((s) => s.connected);
  const selectedVariant = useStore((s) => s.selectedVariant);

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
        {selectedVariant
          ? `YOLOv8s-${selectedVariant.toUpperCase()} · OCR-INT8 · ByteTrack`
          : "YOLOv8n-INT8 · OCR-INT8 · ByteTrack"}
      </span>
    </footer>
  );
}

export default function App() {
  const { toggle: toggleTheme } = useTheme();
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [splashVisible, setSplashVisible] = useState(true);
  // chatInputRef: forwarded to QAPanel's <textarea> via Rail
  const chatInputRef = useRef<HTMLTextAreaElement>(null);

  const setVariants = useStore((s) => s.setVariants);
  const setSelectedVariant = useStore((s) => s.setSelectedVariant);
  const pushToast = useStore((s) => s.pushToast);

  // getStatus on mount — populate variants + active_variant
  useEffect(() => {
    getStatus()
      .then((status) => {
        setVariants(status.variants ?? [], status.active_variant ?? null);
      })
      .catch(() => {
        // Server not reachable — still dismiss splash after timeout
      });
    // Fallback: hide splash max 2.5s (app.js pattern)
    const t = setTimeout(() => setSplashVisible(false), 2500);
    return () => clearTimeout(t);
  }, [setVariants]);

  // Dismiss splash when WS connects
  const connected = useStore((s) => s.connected);
  useEffect(() => {
    if (connected) {
      setSplashVisible(false);
      pushToast("파이프라인 서버에 연결되었습니다", "ok");
    }
  }, [connected, pushToast]);

  // Sync server-reported variant to selectedVariant
  const serverVariant = useStore((s) => s.telemetry.variant);
  const variants = useStore((s) => s.variants);
  const selectedVariant = useStore((s) => s.selectedVariant);
  useEffect(() => {
    if (serverVariant && variants.find((v) => v.name === serverVariant) && !selectedVariant) {
      setSelectedVariant(serverVariant);
    }
  }, [serverVariant, variants, selectedVariant, setSelectedVariant]);

  // Hotkeys — fully wired (T8)
  useHotkeys({
    onTogglePlay: () => {
      // Viewport registers its own Space handler; no double-handling needed here.
      // The Viewport listens to keydown directly via its own useHotkeys instance.
    },
    onFocusChat: () => {
      // Switch to QA tab and focus chat input
      useStore.getState().setTab("qa");
      // Small delay to let tab switch render
      setTimeout(() => chatInputRef.current?.focus(), 50);
    },
    onToggleTheme: toggleTheme,
    onToggleShortcuts: (open) => {
      setShortcutsOpen((prev) => (open !== undefined ? open : !prev));
    },
    onStepBack: () => {
      // Handled by Viewport's internal stepBack; this noop avoids double wiring
    },
    onStepForward: () => {
      // Handled by Viewport
    },
  });

  return (
    <>
      {splashVisible && <Splash />}

      <Header
        onToggleTheme={toggleTheme}
        onOpenShortcuts={() => setShortcutsOpen(true)}
      />

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

        {/* 우: 레일 (T8 완성) */}
        <Rail ref={chatInputRef} />
      </main>

      <Footer />

      {/* 단축키 모달 (T8) */}
      <ShortcutsModal
        open={shortcutsOpen}
        onOpenChange={setShortcutsOpen}
      />

      {/* 토스트 큐 포탈 컨테이너 */}
      <div id="toast-wrap" aria-live="polite" aria-atomic="false" />
      <Toast />
    </>
  );
}
