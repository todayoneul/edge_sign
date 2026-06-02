/**
 * Splash.tsx — 시작 스플래시 화면
 * web/detection/index.html #splash 포팅
 */

export default function Splash() {
  return (
    <div id="splash" role="status" aria-live="polite">
      <span className="splash-mark" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none">
          <path
            d="M12 2.5 21.5 12 12 21.5 2.5 12 12 2.5Z"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinejoin="round"
          />
          <circle cx="12" cy="12" r="3" fill="currentColor" />
        </svg>
      </span>
      <span className="splash-word">EDGE&middot;SIGN</span>
      <div className="splash-bar" aria-hidden="true">
        <i />
      </div>
      <span className="splash-status" id="splash-status">
        시스템 초기화 중…
      </span>
    </div>
  );
}
