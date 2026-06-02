/**
 * Hero.tsx — 뷰포트 시작 전 히어로 오버레이
 * web/detection/index.html #hero 포팅
 */

interface Props {
  onFile: () => void;
  onSample: () => void;
  onWebcam: () => void;
}

export default function Hero({ onFile, onSample, onWebcam }: Props) {
  return (
    <div id="hero">
      <span className="hero-mark" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none">
          <path
            d="M12 2.5 21.5 12 12 21.5 2.5 12 12 2.5Z"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinejoin="round"
          />
          <circle cx="12" cy="12" r="3" fill="var(--c-sign)" />
        </svg>
      </span>

      <h2>
        주행 인지를
        <br />
        엣지에서, 실시간으로.
      </h2>

      <p className="lead">
        초경량 양자화 검출 · 추적 · 인식 파이프라인. 표지판 · 신호등 · 간판을 동시에 감지하고,
        장면을 그대로 질문하세요.
      </p>

      <div className="hero-stats">
        <span className="hero-stat">
          <b>
            &lt;15<small>MB</small>
          </b>
          <span>모델 크기</span>
        </span>
        <span className="hero-stat">
          <b>
            30+<small> FPS</small>
          </b>
          <span>실시간 추론</span>
        </span>
        <span className="hero-stat">
          <b>INT8</b>
          <span>양자화</span>
        </span>
      </div>

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

        <button className="btn btn-ghost" id="hero-sample" onClick={onSample}>
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
          샘플 영상
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

      <p className="hero-hint">샘플로 바로 체험하거나, 본인 영상을 드래그·업로드하세요</p>
    </div>
  );
}
