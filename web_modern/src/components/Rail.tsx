/**
 * Rail.tsx — 우측 레일: 인식 결과 탭 + 주행 어시스턴트 탭
 * (app.js 세그먼트 탭 + index.html #rail 포팅)
 *
 * Radix Tabs 사용.
 * TracksPanel: store.tracks → 트랙 행 렌더 + hover↔overlay 연동
 * QAPanel: useQA SSE 채팅 + BYOK + quick chips
 */

import { forwardRef } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import { useStore } from "../store";
import TracksPanel from "./TracksPanel";
import QAPanel from "./QAPanel";

// Expose chatInputRef so App.tsx can forward it for / hotkey focus.
const Rail = forwardRef<HTMLTextAreaElement>((_, chatRef) => {
  const tracks = useStore((s) => s.tracks);
  const activeTab = useStore((s) => s.activeTab);
  const setTab = useStore((s) => s.setTab);

  return (
    <aside id="rail" aria-label="결과 및 어시스턴트">
      <Tabs.Root
        value={activeTab}
        onValueChange={(v) => setTab(v as "tracks" | "qa")}
        style={{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0 }}
      >
        {/* 탭 목록 */}
        <Tabs.List className="seg" aria-label="결과 탭">
          <Tabs.Trigger
            value="tracks"
            className="seg-btn"
            aria-selected={activeTab === "tracks"}
            id="tab-tracks"
          >
            인식 결과
            <span className="badge">{tracks.length}</span>
          </Tabs.Trigger>
          <Tabs.Trigger
            value="qa"
            className="seg-btn"
            aria-selected={activeTab === "qa"}
            id="tab-qa"
          >
            주행 어시스턴트
          </Tabs.Trigger>
        </Tabs.List>

        {/* 트랙 패널 */}
        <Tabs.Content
          value="tracks"
          id="tracks-panel"
          className={`panel${activeTab === "tracks" ? " active" : ""}`}
          style={{ display: activeTab === "tracks" ? "flex" : "none", flexDirection: "column", flex: 1, minHeight: 0 }}
        >
          <div className="panel-head">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.4} style={{ width: 14, height: 14 }} aria-hidden="true">
              <path d="M3 12h4l2 6 4-12 2 6h6" />
            </svg>
            인식 객체
            <span className="tally" id="track-tally">{tracks.length}</span>
          </div>
          <TracksPanel />
        </Tabs.Content>

        {/* Q&A 패널 */}
        <Tabs.Content
          value="qa"
          id="qa-panel"
          className={`panel${activeTab === "qa" ? " active" : ""}`}
          style={{ display: activeTab === "qa" ? "flex" : "none", flexDirection: "column", flex: 1, minHeight: 0 }}
        >
          <QAPanel ref={chatRef} />
        </Tabs.Content>
      </Tabs.Root>
    </aside>
  );
});

Rail.displayName = "Rail";
export default Rail;
