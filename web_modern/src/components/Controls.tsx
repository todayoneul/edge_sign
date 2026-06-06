/**
 * Controls.tsx — 소스 선택 바 (웹캠/동영상/URL/정지 + 상태)
 * web/detection/index.html .controls 포팅
 *
 * 재생 컨트롤(재생/정지·탐색·속도·5초 점프)은 뷰포트 내부 오버레이
 * 트랜스포트 바(SeekBar)로 옮겼다. 여기는 소스 선택만 담당.
 *
 * 모드①(webcam/H.264 file) → useStream 경로 (Viewport가 직접 처리)
 * 모드②(URL/image/incompatible) → useSession 서버 인제스트
 */

import { useRef } from "react";
import { useStore } from "../store";

interface Props {
  playing: boolean;
  onWebcam: () => void;
  onFile: (f: File) => void;
  onStop: () => void;
  /** URL/이미지 서버 인제스트 */
  onUrl?: (url: string) => void;
  stageStatus?: string;
  stageStatusLive?: boolean;
}

export default function Controls({
  playing,
  onWebcam,
  onFile,
  onStop,
  onUrl,
  stageStatus = "서버 연결 대기 중…",
  stageStatusLive = false,
}: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pipelineMode = useStore((s) => s.pipelineMode);
  const setPipelineMode = useStore((s) => s.setPipelineMode);
  const ondeviceModel = useStore((s) => s.ondeviceModel);
  const setOndeviceModel = useStore((s) => s.setOndeviceModel);
  const pushToast = useStore((s) => s.pushToast);

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

        {/* 추론 위치 토글: 서버 WS ⇄ 브라우저 온디바이스(WebGPU). 다음 소스 시작 시 적용. */}
        <div
          className="mode-toggle"
          role="group"
          aria-label="추론 위치"
          title="추론을 서버에서 할지, 브라우저(WebGPU)에서 직접 할지"
        >
          {(["server", "ondevice"] as const).map((m) => (
            <button
              key={m}
              type="button"
              className={`mode-btn${m === "ondevice" ? " mode-btn--edge" : ""}`}
              aria-pressed={pipelineMode === m}
              onClick={() => {
                if (pipelineMode === m) return;
                setPipelineMode(m);
                pushToast(
                  m === "ondevice"
                    ? "온디바이스(WebGPU) — 다음 웹캠/영상 시작부터 브라우저에서 추론"
                    : "서버 추론 모드로 전환",
                  "ok",
                );
              }}
            >
              {m === "server" ? "서버" : "온디바이스"}
            </button>
          ))}
        </div>

        {/* 온디바이스 정밀도: fp32(빠름) ⇄ fp16(작음) — 다음 시작 시 적용 */}
        {pipelineMode === "ondevice" && (
          <div
            className="mode-toggle"
            role="group"
            aria-label="온디바이스 모델 정밀도"
            title="fp32=빠름(43MB) · fp16=작음(22MB). 다음 시작부터 적용"
          >
            {(["fp32", "fp16"] as const).map((p) => (
              <button
                key={p}
                type="button"
                className="mode-btn"
                aria-pressed={ondeviceModel === p}
                onClick={() => {
                  if (ondeviceModel === p) return;
                  setOndeviceModel(p);
                  pushToast(
                    p === "fp16" ? "온디바이스 FP16 (작음·22MB)" : "온디바이스 FP32 (빠름·43MB)",
                    "ok",
                  );
                }}
              >
                {p.toUpperCase()}
              </button>
            ))}
          </div>
        )}

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
