/**
 * Controls.tsx — 컨트롤 바 (소스 선택 버튼 + URL 입력 + 드롭존 + 속도)
 * web/detection/index.html .controls + .playback 포팅
 *
 * 모드①(webcam/H.264 file) → useStream 경로 (Viewport가 직접 처리)
 * 모드②(URL/image/incompatible) → T7에서 useSession 연결 예정, stub 처리
 */

import { useRef, useState } from "react";

interface Props {
  playing: boolean;
  onSample: () => void;
  onWebcam: () => void;
  onFile: (f: File) => void;
  onStop: () => void;
  /** T7 stub — URL/이미지 서버 인제스트 */
  onUrl?: (url: string) => void;
  /** T7 stub — image file */
  onImageFile?: (f: File) => void;
  stageStatus?: string;
  stageStatusLive?: boolean;
  playbackRate: number;
  onPlaybackRate: (r: number) => void;
  onStepBack: () => void;
  onStepFwd: () => void;
}

export default function Controls({
  playing,
  onSample,
  onWebcam,
  onFile,
  onStop,
  onUrl,
  stageStatus = "서버 연결 대기 중…",
  stageStatusLive = false,
  playbackRate,
  onPlaybackRate,
  onStepBack,
  onStepFwd,
}: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [urlValue, setUrlValue] = useState("");
  const [dragover, setDragover] = useState(false);

  function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    const f = files[0];
    if (f.type.startsWith("image/")) {
      // T7: image → ingest stub
      onUrl?.(`data:image;name=${encodeURIComponent(f.name)}`);
    } else {
      onFile(f);
    }
  }

  function submitUrl() {
    const u = urlValue.trim();
    if (u) onUrl?.(u);
  }

  return (
    <>
      {/* ── 컨트롤 바 ── */}
      <div className="controls">
        <button className="btn btn-ghost" id="sample-btn" onClick={onSample}>
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M8 5v14l11-7z" />
          </svg>
          샘플
        </button>

        <button className="btn btn-ghost" id="webcam-btn" onClick={onWebcam}>
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <rect x="2" y="6" width="14" height="12" rx="2" />
            <path d="M16 10l5-3v10l-5-3" />
          </svg>
          웹캠
        </button>

        <button
          className="btn btn-ghost"
          id="file-btn"
          onClick={() => fileInputRef.current?.click()}
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M5 4h5l2 3h7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2Z" />
          </svg>
          동영상
        </button>

        <button
          className="btn btn-quiet"
          id="stop-btn"
          disabled={!playing}
          onClick={onStop}
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
          >
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
          정지
        </button>

        {/* hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          id="file-input"
          accept="video/*,image/*"
          style={{ display: "none" }}
          onChange={(e) => {
            handleFiles(e.target.files);
            e.target.value = "";
          }}
        />

        {/* URL 입력 (T7 stub) */}
        <input
          type="text"
          id="url-input"
          placeholder="영상 URL / RTSP / YouTube"
          value={urlValue}
          onChange={(e) => setUrlValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") submitUrl();
          }}
          style={{
            flex: "0 1 220px",
            minWidth: 140,
            background: "var(--surface-2)",
            border: "1px solid var(--line-2)",
            borderRadius: "var(--r-sm)",
            padding: "7px 10px",
            color: "var(--ink)",
            fontFamily: "var(--f-mono)",
            fontSize: "0.72rem",
            outline: "none",
          }}
        />
        <button className="btn btn-ghost" id="url-btn" onClick={submitUrl}>
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M10 13a5 5 0 0 0 7 0l3-3a5 5 0 0 0-7-7l-1 1" />
            <path d="M14 11a5 5 0 0 0-7 0l-3 3a5 5 0 0 0 7 7l1-1" />
          </svg>
          URL
        </button>

        <div className="spacer" />
        <span
          className={`stage-status${stageStatusLive ? " live" : ""}`}
          id="stage-status"
        >
          {stageStatus}
        </span>
      </div>

      {/* ── 재생 속도 + 5초 점프 + 드롭존 ── */}
      <div className="playback">
        <label>
          속도
          <input
            type="range"
            id="speed-range"
            min="0.25"
            max="3"
            step="0.25"
            value={playbackRate}
            onChange={(e) => onPlaybackRate(parseFloat(e.target.value))}
            aria-label="재생 속도"
          />
          <span className="speed-val mono" id="speed-val">
            {playbackRate.toFixed(2)}×
          </span>
        </label>

        <button className="pc-btn" id="step-back-btn" title="5초 뒤로" onClick={onStepBack}>
          ⏪ 5s
        </button>
        <button className="pc-btn" id="step-fwd-btn" title="5초 앞으로" onClick={onStepFwd}>
          5s ⏩
        </button>

        <div className="spacer" />

        {/* 드롭존 */}
        <div
          id="dropzone"
          className={dragover ? "dragover" : ""}
          role="button"
          tabIndex={0}
          onClick={() => fileInputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click();
          }}
          onDragOver={(e) => {
            e.preventDefault();
            setDragover(true);
          }}
          onDragLeave={() => setDragover(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragover(false);
            handleFiles(e.dataTransfer.files);
          }}
        >
          영상 파일을 드래그하거나 클릭
        </div>
      </div>
    </>
  );
}
