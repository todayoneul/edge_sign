/**
 * Controls.tsx — 소스 선택 바 (웹캠/동영상/URL/정지 + 상태)
 * web/detection/index.html .controls 포팅
 *
 * 재생 컨트롤(재생/정지·탐색·속도·5초 점프)은 뷰포트 내부 오버레이
 * 트랜스포트 바(SeekBar)로 옮겼다. 여기는 소스 선택만 담당.
 *
 * 모드①(webcam/H.264 file) → 온디바이스 추론 (Viewport가 직접 처리)
 * 모드②(URL/image/incompatible) → useSession 서버 인제스트
 * 추론 위치 토글은 없앴다: 공개 데모는 온디바이스 고정, 모델 선택은 PerfStrip.
 */

import { useRef } from "react";
import { SAMPLES } from "../lib/samples";

interface Props {
  playing: boolean;
  /** file 미지정 시 첫 샘플. 칩별로 명시 파일명 전달. */
  onSample: (file?: string) => void;
  onFile: (f: File) => void;
  onStop: () => void;
  /** URL/이미지 서버 인제스트 */
  onUrl?: (url: string) => void;
  stageStatus?: string;
  stageStatusLive?: boolean;
}

export default function Controls({
  playing,
  onSample,
  onFile,
  onStop,
  onUrl,
  stageStatus = "서버 연결 대기 중…",
  stageStatusLive = false,
}: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  return (
    <>
      {/* ── 컨트롤 바 ── */}
      <div className="controls">
        {/* 내장 샘플 칩(H.264, 온디바이스로 매끄럽게 도는 기준 입력) */}
        {SAMPLES.map((s, i) => (
          <button
            key={s.file}
            className="btn btn-ghost"
            id={i === 0 ? "sample-btn" : undefined}
            onClick={() => onSample(s.file)}
            title={`샘플: ${s.label}`}
          >
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
            {s.label}
          </button>
        ))}

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

        <div className="spacer" />
        <span
          className={`stage-status${stageStatusLive ? " live" : ""}`}
          id="stage-status"
        >
          {stageStatus}
        </span>
      </div>
    </>
  );
}
