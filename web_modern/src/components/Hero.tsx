/**
 * Hero.tsx — 뷰포트 시작 전 히어로 오버레이
 * web/detection/index.html #hero 포팅
 */

import LogoMark from "./LogoMark";
import { SAMPLES } from "../lib/samples";

interface Props {
  onFile: () => void;
  /** 샘플 클립 로드 — file로 어떤 샘플인지 지정 */
  onSample: (file: string) => void;
}

export default function Hero({ onFile, onSample }: Props) {
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
        {SAMPLES.map((s, i) => (
          <button
            key={s.file}
            className={`btn ${i === 0 ? "btn-primary" : "btn-ghost"}`}
            id={i === 0 ? "hero-sample" : undefined}
            onClick={() => onSample(s.file)}
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

        <button className="btn btn-ghost" id="hero-file" onClick={onFile}>
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
          내 영상 열기
        </button>
      </div>

      <p className="hero-hint">
        두 샘플 모두 서울 도심 주행 클립 · 내 영상(H.264 mp4)을 끌어다 놓아도 됩니다
      </p>
    </div>
  );
}
