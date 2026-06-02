/**
 * TracksPanel.tsx — 트랙 목록 (app.js updateTrackList 포팅)
 *
 * - store.tracks → 행 렌더 (id, label, conf 바, class 색상, rowIn 애니메이션)
 * - 행 hover → store.setHoverId → Viewport overlay 하이라이트
 * - canvas hover → store.hoverId → 행 .hl 클래스 토글
 */

import { useStore } from "../store";
import { classColor } from "../lib/draw";

const KIND_KR: Record<number, string> = { 0: "교통표지판", 1: "신호등", 2: "간판" };

export default function TracksPanel() {
  const tracks = useStore((s) => s.tracks);
  const hoverId = useStore((s) => s.hoverId);
  const setHoverId = useStore((s) => s.setHoverId);

  if (!tracks.length) {
    return (
      <div id="track-list" style={{ flex: 1, overflowY: "auto", padding: "var(--sp-4)", display: "flex", flexDirection: "column", gap: "var(--sp-2)" }}>
        <div className="empty" id="no-tracks" style={{ margin: "auto", textAlign: "center", color: "var(--ink-3)", fontFamily: "var(--f-mono)", fontSize: "0.78rem", lineHeight: 1.8 }}>
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={1.4}
            style={{ width: 28, height: 28, marginBottom: "var(--sp-2)", display: "block", margin: "0 auto var(--sp-2)" }}
            aria-hidden="true"
          >
            <path d="M3 12h4l2 6 4-12 2 6h6" />
          </svg>
          인식된 객체가 여기 표시됩니다
          <br />
          영상을 시작하세요
        </div>
      </div>
    );
  }

  return (
    <div
      id="track-list"
      style={{ flex: 1, overflowY: "auto", padding: "var(--sp-4)", display: "flex", flexDirection: "column", gap: "var(--sp-2)" }}
    >
      {tracks.map((t) => {
        const label = t.label ?? t.class_name;
        const clsKr = KIND_KR[t.class] ?? "객체";
        const pct = Math.round(t.conf * 100);
        const color = classColor(t.class);
        const isHl = hoverId === t.id;

        return (
          <div
            key={t.id}
            className={`track-item ${t.class_name}${isHl ? " hl" : ""}`}
            data-id={t.id}
            style={{
              borderLeftColor: color,
              background: isHl ? "var(--surface-3)" : undefined,
              borderColor: isHl ? color : undefined,
            }}
            onMouseEnter={() => setHoverId(t.id)}
            onMouseLeave={() => setHoverId(null)}
          >
            <span className="track-dot" style={{ background: color }} />
            <div className="track-info">
              <span className="track-id">#{t.id} · {clsKr}</span>
              <span className="track-label">{label}</span>
            </div>
            <div className="track-conf-wrap">
              <span className="track-conf">{pct}%</span>
              <span className="conf-bar">
                <i style={{ width: `${pct}%` }} />
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
