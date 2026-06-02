/**
 * draw.ts — 캔버스 한글 박스·라벨 렌더 (web/detection/app.js drawOverlay 포팅)
 *
 * 순수 함수만 노출. React / DOM 참조 없음.
 * 레터박스 오프셋(object-fit: contain)은 Viewport(T6)가 처리,
 * renderTracks는 src→dst 스케일만 담당.
 */

import type { Track } from "./types";

// ── 색상 매핑 ────────────────────────────────────────────────────────────────
// app.js COLORS: traffic_sign(0)='#22c55e', traffic_light(1)='#ef4444', signboard(2)='#f59e0b'
const COLORS: Record<number, string> = {
  0: "#22c55e", // traffic_sign  — 녹색
  1: "#ef4444", // traffic_light — 적색
  2: "#f59e0b", // signboard     — 황색
};

export const classColor = (cls: number): string => COLORS[cls] ?? "#a1a1aa";

// ── mapBox ───────────────────────────────────────────────────────────────────
/**
 * 소스 해상도 bbox를 표시(dst) 좌표로 선형 스케일.
 * [x1, y1, x2, y2] → [x1*sx, y1*sy, x2*sx, y2*sy]
 */
export function mapBox(
  b: [number, number, number, number],
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number,
): [number, number, number, number] {
  const sx = dstW / srcW,
    sy = dstH / srcH;
  return [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy];
}

// ── roundRect 폴리필 ─────────────────────────────────────────────────────────
// app.js roundRect(c, x, y, w, h, r): 네이티브 roundRect 우선, 없으면 arcTo 폴백
function roundRect(
  c: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  r = Math.min(r, w / 2, h / 2);
  if (typeof c.roundRect === "function") {
    c.beginPath();
    c.roundRect(x, y, w, h, r);
    return;
  }
  // arcTo 폴백 (구형 브라우저)
  c.beginPath();
  c.moveTo(x + r, y);
  c.arcTo(x + w, y, x + w, y + h, r);
  c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r);
  c.arcTo(x, y, x + w, y, r);
  c.closePath();
}

// ── renderTracks ─────────────────────────────────────────────────────────────
/**
 * 트랙 배열을 캔버스에 렌더. app.js drawOverlay의 레터박스 독립 버전.
 *
 * @param ctx     - 오버레이 캔버스 2D 컨텍스트
 * @param tracks  - 서버에서 수신한 Track 배열
 * @param srcW    - 서버가 처리한 프레임 너비 (bbox 좌표 기준)
 * @param srcH    - 서버가 처리한 프레임 높이
 * @param dstW    - 캔버스 CSS 표시 너비 (레터박스 내 실제 영상 영역)
 * @param dstH    - 캔버스 CSS 표시 높이
 */
export function renderTracks(
  ctx: CanvasRenderingContext2D,
  tracks: Track[],
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number,
): void {
  ctx.clearRect(0, 0, dstW, dstH);

  const sx = dstW / srcW,
    sy = dstH / srcH;

  for (const t of tracks) {
    const [x1, y1, x2, y2] = t.bbox;
    const bx = x1 * sx,
      by = y1 * sy;
    const bw = (x2 - x1) * sx,
      bh = (y2 - y1) * sy;

    const color = classColor(t.class);

    // ── 박스 (app.js lines 267-271) ──────────────────────────────────────────
    // lineWidth 2, roundRect radius 7, stroke with class color
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.shadowBlur = 0;
    roundRect(ctx, bx, by, bw, bh, 7);
    ctx.stroke();

    // ── 라벨 pill (app.js lines 275-288) ─────────────────────────────────────
    const label = t.label ?? t.class_name;
    const txt = `#${t.id}  ${label}  ${(t.conf * 100).toFixed(0)}%`;

    ctx.font = "600 12px 'Fira Code','Pretendard',monospace";
    const tw = ctx.measureText(txt).width;

    const padX = 7;  // 수평 패딩
    const ph = 19;   // pill 높이
    // pill y: 박스 위쪽에, 화면 위에 잘리지 않도록 max(..., 2)
    const ly = Math.max(by - ph - 3, 2);
    const pillW = tw + padX * 2 + 10; // 점(dot) 여백 10px 포함

    // pill 배경 (class color 채움, radius 6)
    roundRect(ctx, bx, ly, pillW, ph, 6);
    ctx.fillStyle = color;
    ctx.fill();

    // 점 (반경 2.5, 어두운 색)
    ctx.fillStyle = "rgba(0,0,0,0.85)";
    ctx.beginPath();
    ctx.arc(bx + padX + 2, ly + ph / 2, 2.5, 0, Math.PI * 2);
    ctx.fill();

    // 라벨 텍스트 (어두운 색, 세로 중앙)
    ctx.fillStyle = "#0a0a0a";
    ctx.textBaseline = "middle";
    ctx.fillText(txt, bx + padX + 10, ly + ph / 2 + 0.5);

    // ── 신뢰도 바 — 박스 하단 내부 (app.js lines 291-293) ───────────────────
    // height 2.5, width = bw * conf (conf ≤ 1), globalAlpha 0.9
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = color;
    ctx.fillRect(bx, by + bh - 2.5, bw * Math.min(1, t.conf), 2.5);
    ctx.globalAlpha = 1;
  }
}
