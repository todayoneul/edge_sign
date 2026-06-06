/**
 * Rail.tsx — 우측 레일 (Direction A "랩 노트북")
 *
 * 탭 구조 제거 → 세로 스택: 상단 라이브 검출 객체(TracksPanel) +
 * 하단 장면 질의(Scene Q&A, QAPanel) 항상 표면화.
 */

import { forwardRef } from "react";
import { useStore } from "../store";
import TracksPanel from "./TracksPanel";
import QAPanel from "./QAPanel";

// Expose chatInputRef so App.tsx can forward it for / hotkey focus.
const Rail = forwardRef<HTMLTextAreaElement>((_, chatRef) => {
  const tracks = useStore((s) => s.tracks);

  return (
    <aside id="rail" aria-label="검출 결과 및 주행 어시스턴트">
      {/* 상단: 라이브 검출 객체 */}
      <div className="rail-section">
        검출 객체 <span className="tag">{tracks.length}</span>
      </div>
      <div className="rail-tracks">
        <TracksPanel />
      </div>

      {/* 하단: 장면 질의 (Scene Q&A) — 항상 표면화 */}
      <div className="rail-section">
        장면 질의 <span className="tag">Scene Q&amp;A</span>
      </div>
      <QAPanel ref={chatRef} />
    </aside>
  );
});

Rail.displayName = "Rail";
export default Rail;
