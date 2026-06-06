# Handoff: Edge-Sign 디자인 개편 — 방향 A "랩 노트북"

## Overview

기존 `web_modern` 콘솔의 **AI/관제실 분위기를 걷어내고** 연구·분석 도구(사이언티픽/랩)로 재프레이밍하는 디자인 개편입니다.

**핵심 변경 동기:**
- 보라 그라데이션 블롭 로고 → 표지판 기하학 추상 마크
- 빈 `0.0 fps / — ms / 0 / 0` KPI 격자 → idle 상태 제거, "분석 준비됨" 뱃지로 대체
- URL 입력 필드 제거
- 웹캠/영상 선택 버튼을 히어로 안으로 이동 (발견성↑)
- 주행 어시스턴스(Scene Q&A)를 우측 레일에 항상 표면화
- 전체 글자 크기 확대 (기존 대비 +20~30%)
- "관제 시스템" → "주행 인지 분석" 재프레이밍

## About the Design Files

`design_handoff/` 안의 HTML/CSS는 **디자인 레퍼런스** 입니다. 프로덕션 코드로 직접 사용하는 것이 아니라, **기존 `web_modern` React/TypeScript/Vite 코드베이스에서 동일한 시각 결과를 구현하는 가이드**로 사용하세요. 각 컴포넌트별로 어떤 값을 어떻게 바꾸는지 이 문서에 정리했습니다.

## Fidelity

**High-fidelity** — 색상, 타이포그래피, 간격, 상호작용 상태가 픽셀 수준으로 지정되어 있습니다. 기존 코드베이스의 컴포넌트 구조는 유지하면서 스타일 값을 이 문서 기준으로 교체해 주세요.

---

## 1. 디자인 토큰 변경 (`src/styles/tokens.css`)

기존 보라 계열 `--brand-*` 토큰을 아래로 교체합니다. 시그널 색은 의미 보존을 위해 유지하되, 채도를 통일합니다.

```css
/* ── Neutral / Paper ─────────────────────────── */
--paper:      #fbfbf9;   /* 앱 배경 */
--surface:    #ffffff;   /* 카드, 헤더, 레일 */
--surface-2:  #f4f5f3;   /* 입력 배경, 칩 배경 */
--surface-3:  #eaecee;   /* hover 상태 */
--ink:        #15171b;   /* 본문 */
--ink-2:      #4b525d;   /* 보조 텍스트 */
--ink-3:      #828a95;   /* 레이블, 캡션 */
--ink-4:      #aab0ba;   /* 플레이스홀더 */
--line:       #e7e9ec;   /* 주 구분선 */
--line-2:     #d6dade;   /* 보조 구분선 */
--line-3:     #c4c9cf;   /* 강조 구분선 */

/* ── Video stage (어두운 배경, 바운딩박스 대비용) ─ */
--video:      #0b0d11;
--video-2:    #12151b;

/* ── Signal colours (유지, 채도 통일) ──────────── */
--sign:       #18a268;   /* 교통표지판 */
--light:      #e23b3b;   /* 신호등 */
--board:      #cf8a09;   /* 간판 */

/* ── On-device engine accent ────────────────── */
--edge:       #4b57d6;
--edge-ink:   #ffffff;

/* ── 제거할 기존 토큰 ─────────────────────────── */
/* --brand-purple, --brand-gradient, --glow-*, --glass-* 등 AI/관제 계열 전부 삭제 */
```

---

## 2. 로고 마크 교체

### 새 로고: "Detection Cell" (방향 A)

기존의 보라 그라데이션 블롭 + 다이아몬드 마크를 아래 SVG로 교체합니다.

**`public/favicon.svg` 전체 교체:**
```svg
<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 40 40" fill="none">
  <rect x="9" y="9" width="22" height="22" rx="5.5"
        transform="rotate(45 20 20)" stroke="#15171b" stroke-width="2"/>
  <rect x="16" y="16" width="8" height="8" rx="1.6" fill="#18a268"/>
</svg>
```

**컴포넌트 내 로고 마크 (헤더·히어로에서 모두 동일 SVG 사용):**
```tsx
// LogoMark.tsx (신규 컴포넌트 또는 인라인)
export const LogoMark = ({ size = 38, color = 'currentColor' }) => (
  <svg width={size} height={size} viewBox="0 0 40 40" fill="none">
    <rect
      x="9" y="9" width="22" height="22" rx="5.5"
      transform="rotate(45 20 20)"
      stroke={color} strokeWidth="2"
    />
    <rect x="16" y="16" width="8" height="8" rx="1.6" fill="var(--sign)" />
  </svg>
);
```

**워드마크 구조:**
```tsx
<div className="brand">
  <LogoMark size={38} />
  <div className="wordmark">
    <b>Edge<span className="accent">·</span>Sign</b>
    {/* accent color: var(--sign) = #18a268 */}
    <span className="sub">온디바이스 주행 인지</span>
    {/* font: Fira Code, 11.5px, weight 500, letter-spacing 0.22em, uppercase */}
  </div>
</div>
```

---

## 3. Header.tsx 변경

### 제거
- KPI 격자 (`FPS 스파크라인 / 추론 지연 / 활성 트랙 / 누적 검출`) — idle 상태에서 `0.0`, `—`, `0` 등을 표시하는 부분 전체 제거
- URL 입력 필드 (`<input type="url" ...>`) 제거

### 추가/변경

**헤더 오른쪽 클러스터 (왼→오 순서):**

```tsx
{/* 1. 온디바이스 엔진 뱃지 */}
<span className="engine-badge">
  <span className="dot" /> {/* 7px 원, background: var(--edge) */}
  온디바이스 · WebGPU
</span>
{/* 스타일:
  font: Fira Code 13px weight 600 letter-spacing 0.02em
  color: var(--edge) = #4b57d6
  padding: 8px 13px  border-radius: 9px
  border: 1px solid color-mix(in srgb, var(--edge) 40%, transparent)
  background: color-mix(in srgb, var(--edge) 8%, transparent)
*/}

{/* 2. Ready 상태 뱃지 (idle 시에만 표시, 분석 중에는 숨김) */}
{!isRunning && (
  <span className="ready-badge">
    <span className="dot" /> {/* 8px 원, bg: var(--sign), glow: 0 0 0 3px sign/22% */}
    분석 준비됨
  </span>
)}
{/* 스타일:
  font: Fira Code 13px weight 500 letter-spacing 0.04em
  color: var(--ink-2)
  padding: 8px 14px  border: 1px solid var(--line-2)  border-radius: 9px
*/}

{/* 3. 테마 토글 아이콘버튼 (기존 유지, 크기만 조정) */}
{/* 4. 단축키/도움말 아이콘버튼 */}
{/* 아이콘버튼 공통: 44×44px, border: 1px solid var(--line-2), border-radius: 10px */}
```

**헤더 레이아웃:**
```css
.header {
  display: flex;
  align-items: center;
  gap: 22px;
  padding: 18px 28px;        /* 기존보다 여유 확보 */
  background: var(--surface);
  border-bottom: 1px solid var(--line);
  height: auto;              /* 고정 높이 제거 */
}
.header .spacer { flex: 1; }
```

---

## 4. Hero.tsx 변경 (가장 중요)

히어로는 idle 상태에서 뷰포트를 가득 채우는 오버레이입니다. 기존의 작은 텍스트 + 분리된 버튼 레이아웃을 아래로 교체합니다.

### 레이아웃 구조
```tsx
<div className="hero-overlay"> {/* position: absolute; inset: 0; z-index: 2 */}
  <LogoMark size={54} color="rgba(255,255,255,0.85)" />

  <h2>
    <span className="eyebrow">On-device perception</span>
    {/* font: Fira Code 15px weight 500 letter-spacing 0.26em uppercase
        color: rgba(255,255,255,0.42)  margin-bottom: 16px */}
    주행 장면을<br />엣지에서 직접 분석합니다
    {/* font: Fira Sans 46px weight 600 letter-spacing -0.025em line-height 1.08
        color: #fbfbfb */}
  </h2>

  <p className="lead">
    표지판 · 신호등 · 간판을{' '}
    <b>브라우저(WebGPU)</b>에서 동시에 검출·추적·인식합니다.{' '}
    <span className="acc">서버 전송 없이</span>, 영상은 기기 밖으로 나가지 않습니다.
    {/* font: 19px line-height 1.62 color: rgba(255,255,255,0.66) max-width: 46ch
        b → color:#fff  .acc → color:#9aa6ff */}
  </p>

  {/* CTA 버튼 클러스터 — 히어로 안에 위치, 외부 컨트롤 바 불필요 */}
  <div className="actions">
    <button className="btn-primary" onClick={onFile}>
      <ScanIcon /> 영상 분석 시작
    </button>
    <button className="btn-ghost" onClick={onCam}>
      <CamIcon /> 웹캠으로 보기
    </button>
  </div>

  <p className="hint">영상 파일을 이 영역에 끌어다 놓아도 됩니다</p>
  {/* font: Fira Code 13px letter-spacing 0.04em color: rgba(255,255,255,0.34) */}
</div>
```

### 버튼 스타일
```css
.btn { /* 공통 */
  font: 500 16px 'Fira Sans';
  min-height: 54px;
  padding: 0 24px;
  display: inline-flex; align-items: center; gap: 11px;
  border-radius: 11px; border: 1px solid transparent;
}
.btn-primary {
  background: #fbfbfb; color: #111317;
}
.btn-ghost {
  background: rgba(255,255,255,0.07); color: #f4f5f7;
  border-color: rgba(255,255,255,0.22);
}
```

### 코너 등록 틱 (기존 AI 브라켓 대체)
```css
/* 뷰포트 모서리에 카메라 조준경 느낌의 L자 틱 */
.viewport-ticks i {
  position: absolute; width: 13px; height: 13px; z-index: 3;
  border: 1.5px solid rgba(255,255,255,0.28);
}
.viewport-ticks i:nth-child(1) { top:14px; left:14px;  border-right:0; border-bottom:0; }
.viewport-ticks i:nth-child(2) { top:14px; right:14px; border-left:0;  border-bottom:0; }
.viewport-ticks i:nth-child(3) { bottom:14px; left:14px;  border-right:0; border-top:0; }
.viewport-ticks i:nth-child(4) { bottom:14px; right:14px; border-left:0;  border-top:0; }
```

---

## 5. 우측 Rail 변경

Rail 너비: `366px`  
Rail 배경: `var(--surface)`, `border-left: 1px solid var(--line)`

### 섹션 레이블 (`.rail-section`)
```css
.rail-section {
  display: flex; align-items: center; gap: 9px;
  padding: 16px 18px 14px;
  font: 600 12px 'Fira Code';
  letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--ink-3);
  border-bottom: 1px solid var(--line);
}
.rail-section .tag { margin-left: auto; color: var(--ink-4); font-weight: 500; }
```

### 검출 범례 (기존 유지, 스타일 정제)
```css
.legend-row {
  display: flex; align-items: center; gap: 12px;
  padding: 13px 18px; border-bottom: 1px solid var(--line);
  font-size: 15.5px; font-weight: 500;
}
.legend-swatch {
  width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0;
}
.legend-en {
  font: 500 11.5px 'Fira Code';
  letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--ink-3); margin-left: auto;
}
/* 색: sign=#18a268, light=#e23b3b, board=#cf8a09 */
```

### Scene Q&A (기존 숨겨진/접힌 상태 → 항상 표면화)
```tsx
{/* rail 하단 섹션으로 항상 펼쳐서 표시 */}
<div className="rail-section">
  장면 질의 <span className="tag">Scene Q&A</span>
</div>
<div className="qa-panel">
  <div className="qa-title">
    <b>장면 질의</b>
    <span className="qa-en">Scene Q&A</span>
    {/* .qa-en: Fira Code 11.5px letter-spacing 0.12em uppercase color:var(--ink-3) */}
  </div>
  <p className="qa-desc">
    분석 중인 장면을 자연어로 질문하세요. 검출된 표지판·신호등·간판을 근거로 답합니다.
    {/* 14.5px color: var(--ink-2) line-height: 1.55 */}
  </p>
  <div className="chips">
    <button className="chip">제한속도는?</button>
    <button className="chip">앞에 신호등 있나요?</button>
    <button className="chip">간판 텍스트 읽어줘</button>
    {/* chip: 13.5px padding 8px 13px bg:var(--surface-2)
        border: 1px solid var(--line)  border-radius: 999px */}
  </div>
  <div className="ask-input">
    <span className="placeholder">장면에 대해 물어보기…</span>
    {/* placeholder: 15px color: var(--ink-4) */}
    <button className="send-btn">
      <SendIcon /> {/* 38×38px bg:var(--ink) color:var(--paper) border-radius:9px */}
    </button>
  </div>
  {/* ask-input: padding 13px 15px bg:var(--surface-2)
      border: 1px solid var(--line-2)  border-radius: 12px */}
</div>
```

---

## 6. Footer / StatusBar 변경

### 제거
- `처리 FPS: 0.0` — idle 시 숫자 표시 제거
- `추론 지연 — ms` — idle 시 표시 제거
- `활성 트랙: 0`, `누적 검출: 0` — idle 시 표시 제거

### idle 상태 Footer
```tsx
{/* 분석 미실행(idle) 시 */}
<footer className="status-bar">
  <span className="status-item">
    <b>대기</b> · 입력 소스를 선택하세요
  </span>
  <span className="status-item">
    검출기 <b>YOLOv8s-signs</b>
  </span>
  <span className="status-item">
    추론 <b>INT8</b> · 13 MB
  </span>
  <span className="status-right">v3 · WebGPU EP</span>
</footer>

{/* 분석 실행 시 — 기존 KPI 표시 로직 유지 */}
```

```css
.status-bar {
  display: flex; align-items: center; gap: 26px;
  padding: 13px 28px;
  background: var(--surface); border-top: 1px solid var(--line);
  font: 400 13px 'Fira Code';
  letter-spacing: 0.03em; color: var(--ink-3);
}
.status-item b { color: var(--ink-2); font-weight: 600; }
.status-right { margin-left: auto; color: var(--ink-4); }
```

---

## 7. 제거 목록 요약

| 위치 | 제거 항목 | 이유 |
|------|-----------|------|
| `Header.tsx` | KPI 격자 (FPS·지연·트랙·검출 idle 표시) | idle에 0.0 표시 → 미완성처럼 보임 |
| `Header.tsx` | URL 입력 필드 | 사용 위치 불명확, 사용자가 불필요 판단 |
| 별도 컨트롤 바 | 영상/웹캠 선택 버튼 (헤더 또는 스트립) | 히어로 안으로 이동 |
| `PerfStrip.tsx` 또는 헤더 | "주행 어시스턴스" 접힌 섹션 토글 | Q&A가 레일에 항상 노출되므로 중복 제거 |

---

## 8. 타이포그래피 (변경 없음, 크기만 확대)

| 요소 | 기존 | 변경 후 |
|------|------|---------|
| 히어로 제목 | ~32px | **46px** weight 600 |
| 히어로 본문 | ~15px | **19px** |
| 버튼 | 14px | **16px** |
| 레일 본문 | 13px | **14.5~15.5px** |
| 푸터/레이블 | 11~12px | **13px** (모노) |

폰트 패밀리 유지: `Fira Sans` (산세리프), `Fira Code` (모노/레이블/뱃지), `Pretendard` (한글)

---

## 9. 파일 목록

| 파일 | 역할 |
|------|------|
| `Edge-Sign Redesign.html` | 4방향 비교 캔버스 (디자인 레퍼런스) |
| `mockup.css` | 공유 토큰·레이아웃 스타일 (Direction A 기준) |
| `directions.css` | 방향별 오버라이드 |
| `directions.jsx` | 4방향 HTML 마크업 + SVG 마크 정의 |

---

## 10. 구현 순서 제안

1. `tokens.css` 토큰 교체 (15분)
2. `public/favicon.svg` + `LogoMark` 컴포넌트 교체 (10분)
3. `Header.tsx` — KPI 격자 / URL 입력 제거, 뱃지 추가 (30분)
4. `Hero.tsx` — 전체 재작성 (45분)
5. Rail Q&A 섹션 항상 표시로 변경 (20분)
6. Footer idle 상태 교체 (15분)
7. 전체 폰트 크기 변수/클래스 확대 적용 (20분)
