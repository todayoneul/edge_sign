/**
 * Hero.tsx — 뷰포트 시작 전 히어로 오버레이
 * web/detection/index.html #hero 포팅
 */

import LogoMark from "./LogoMark";

interface Props {
  onFile: () => void;
  onWebcam: () => void;
}

export default function Hero({ onFile, onWebcam }: Props) {
  return (
    <div id="hero">
      <span className="hero-mark" aria-hidden="true">
        <LogoMark size={50} color="rgba(255,255,255,0.85)" />
      </span>

      <h2>
        <span className="eyebrow">On-device perception</span>
        주행 장면을
        <br />
        엣지에서 직접 분석합니다
      </h2>

      <p className="lead">
        표지판 · 신호등 · 간판을 <b>브라우저(WebGPU)</b>에서 동시에 검출 · 추적 · 인식합니다.{" "}
        <span className="acc">서버 전송 없이</span>, 영상은 기기 밖으로 나가지 않습니다.
      </p>

      <div className="hero-cta">
        <button className="btn btn-primary" id="hero-file" onClick={onFile}>
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
          동영상 열기
        </button>

        <button className="btn btn-ghost" id="hero-webcam" onClick={onWebcam}>
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
          웹캠 시작
        </button>
      </div>

      <p className="hero-hint">영상을 드래그·업로드하거나 웹캠으로 바로 체험하세요</p>
    </div>
  );
}
