# 프론트엔드 현대화 (SP-C) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** vanilla `web/detection/`(959+874줄)를 동작 패리티로 보존하며 `web_modern/`의 React+Vite+TS 컴포넌트 앱 + 글래스모피즘 리디자인(양 테마·라이트 기본)으로 재구축한다.

**Architecture:** `web_modern/`에 병행 구축(기존 데모·HF 무중단). 핵심 로직(캔버스 한글 렌더·WS/SSE·상태)은 프레임워크 비의존 `lib/`·`hooks/`·zustand `store/`로 분리하고 TDD. UI는 기존 `web/detection/app.js`·`index.html`을 **동작·마크업 명세로 읽어** 포팅. 백엔드 API 무변경. 패리티 도달 시 `app.py` 마운트를 dist-우선+레거시 폴백으로 컷오버.

**Tech Stack:** React 18 · Vite · TypeScript · Tailwind CSS 3.4 + 커스텀 토큰 · Radix UI · zustand · Vitest + React Testing Library + jsdom · Node 24 (설치 확인됨)

---

## 사전 점검 (Pre-flight)

- 브랜치 `feat/frontend-modernization` 확인: `git branch --show-current`
- Node/npm 확인(설치됨): `node --version`(v24) · `npm --version`(11).
- **작업 디렉터리(cwd) 규칙**: `npm`/`vite`/`vitest` 명령은 **`web_modern/`에서** 실행(`cd web_modern && <cmd>` 한 줄로, 또는 `npm --prefix web_modern run <script>`). `git` 명령은 **저장소 루트에서** 실행(`git add web_modern`). PowerShell에서 `cd`가 권한 프롬프트를 띄우면 `npm --prefix web_modern ...` 형태 사용.
- **백엔드 무변경 원칙**: Task 9(컷오버) 외에는 `src/`·`tests/`를 건드리지 않는다. 백엔드 회귀 게이트는 convnext_env에서:
  `& "C:\Users\leegy\miniconda3\envs\convnext_env\python.exe" -m pytest tests/ -q` → 20 passed (시작·종료 시 동일해야 함).
- **포팅 원천**: `web/detection/index.html`(UI 마크업·디자인 토큰·단축키), `web/detection/app.js`(WS 2모드·캔버스 렌더·seek·A/B·SSE Q&A·BYOK). 실행자는 해당 파일을 읽어 동작을 1:1 보존한다.
- **커밋**: 한국어 conventional-commit, **Co-Authored-By 줄 없음**. `web_modern/`은 `.md`가 아니므로 일반 `git add` 사용. `dist/`·`node_modules/`는 커밋 금지(아래 .gitignore).

## 파일 구조 (생성 — 전부 `web_modern/` 하위, 백엔드 무관)

| 파일 | 책임 | 태스크 |
| :--- | :--- | :--- |
| `package.json`·`vite.config.ts`·`tsconfig*.json`·`index.html`·`.gitignore` | 스캐폴드·프록시·빌드 | T1 |
| `tailwind.config.ts`·`postcss.config.js`·`src/styles/tokens.css`·`globals.css` | 글래스·시그널·양 테마 토큰 | T2 |
| `src/hooks/useTheme.ts` (+test) | 테마(라이트 기본·persist·FOUC) | T2 |
| `src/lib/types.ts`·`src/lib/api.ts` (+test) | Track/FrameResult 타입 · REST/WS/SSE 클라 | T3 |
| `src/lib/draw.ts` (+test) | 캔버스 박스·한글 라벨 렌더(순수) | T4 |
| `src/store/index.ts` (+test) | zustand: session·tracks·telemetry·ui·qa | T5 |
| `src/hooks/{useStream,useSession,useQA,useHotkeys}.ts` (+test) | WS 모드①②·SSE·단축키 | T5 |
| `src/components/*` · `src/App.tsx`·`main.tsx` | 셸·뷰포트·컨트롤·레일·모달 | T6–T8 |
| `src/pipeline/app.py` 마운트 | dist-우선+레거시 폴백 컷오버 | T9 |

---

## Task 1: 스캐폴드 (Vite + React + TS + Tailwind + Vitest)

**Files:** `web_modern/` 신규 (package.json, vite.config.ts, tsconfig.json, tsconfig.node.json, index.html, .gitignore, src/main.tsx, src/App.tsx, src/vite-env.d.ts)

- [ ] **Step 1: Vite 프로젝트 생성**

Run (저장소 루트에서):
```bash
npm create vite@latest web_modern -- --template react-ts
cd web_modern && npm install
npm install -D tailwindcss@^3.4 postcss autoprefixer vitest @testing-library/react @testing-library/jest-dom @testing-library/user-event jsdom
npm install zustand @radix-ui/react-dialog @radix-ui/react-tabs @radix-ui/react-slider
npx tailwindcss init -p
```
Expected: `web_modern/` 생성, 의존성 설치 완료.

- [ ] **Step 2: `web_modern/.gitignore` 작성** (dist·node_modules 제외)

```
node_modules
dist
*.local
.vite
coverage
```

- [ ] **Step 3: `web_modern/vite.config.ts` — 프록시 + Vitest 설정**

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 개발: /api·/ws 를 FastAPI(:8000)로 프록시 (ws: true 로 WebSocket 프록시)
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/ws": { target: "ws://localhost:8000", ws: true },
      "/detection": { target: "http://localhost:8000", changeOrigin: true }, // 샘플 클립 등
    },
  },
  build: { outDir: "dist" },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
  },
});
```

- [ ] **Step 4: `web_modern/src/test-setup.ts`**

```ts
import "@testing-library/jest-dom";
```

- [ ] **Step 5: `package.json` 스크립트 확인/추가** (`scripts`에 test 추가)

```json
"scripts": {
  "dev": "vite",
  "build": "tsc -b && vite build",
  "preview": "vite preview",
  "test": "vitest run",
  "test:watch": "vitest"
}
```

- [ ] **Step 6: 빌드·테스트 도구 동작 확인**

Run: `npm run build`
Expected: `dist/` 생성, 에러 없음.
Run: `npx vitest run`
Expected: "No test files found" 또는 0 통과(아직 테스트 없음) — 도구가 실행되면 성공.

- [ ] **Step 7: 커밋**

```bash
cd ..   # 저장소 루트
git add web_modern
git commit -m "feat(web): web_modern 스캐폴드 (Vite+React+TS+Tailwind+Vitest)"
```

---

## Task 2: 디자인 토큰 + 테마 (글래스·라이트 기본, TDD)

**Files:** Create `web_modern/tailwind.config.ts`, `src/styles/tokens.css`, `src/styles/globals.css`, `src/hooks/useTheme.ts`, `src/hooks/useTheme.test.ts`. Modify `src/main.tsx`.

> 토큰 값은 `web/detection/index.html`의 `:root`/`html[data-theme=...]` CSS 변수를 **원천**으로 이식하되, 글래스 레이어(반투명·blur·깊이 그림자)를 추가한다. 시그널색(`--c-sign #22c55e`/`--c-light #ef4444`/`--c-board #f59e0b`)은 그대로.

- [ ] **Step 1: `useTheme` 실패 테스트** (`src/hooks/useTheme.test.ts`)

```ts
import { renderHook, act } from "@testing-library/react";
import { useTheme } from "./useTheme";

beforeEach(() => localStorage.clear());

test("기본 테마는 light (저장값 없을 때)", () => {
  const { result } = renderHook(() => useTheme());
  expect(result.current.theme).toBe("light");
  expect(document.documentElement.getAttribute("data-theme")).toBe("light");
});

test("toggle 시 dark로 전환되고 localStorage에 저장", () => {
  const { result } = renderHook(() => useTheme());
  act(() => result.current.toggle());
  expect(result.current.theme).toBe("dark");
  expect(localStorage.getItem("edge-sign-theme")).toBe("dark");
  expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
});

test("저장된 dark를 복원", () => {
  localStorage.setItem("edge-sign-theme", "dark");
  const { result } = renderHook(() => useTheme());
  expect(result.current.theme).toBe("dark");
});
```

- [ ] **Step 2: 실패 확인** — Run: `npx vitest run src/hooks/useTheme.test.ts` → FAIL (모듈 없음)

- [ ] **Step 3: `src/hooks/useTheme.ts` 구현**

```ts
import { useCallback, useEffect, useState } from "react";

export type Theme = "light" | "dark";
const KEY = "edge-sign-theme";

function initial(): Theme {
  const saved = localStorage.getItem(KEY);
  return saved === "dark" || saved === "light" ? saved : "light"; // 라이트 기본
}

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(initial);
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(KEY, theme);
  }, [theme]);
  const toggle = useCallback(
    () => setTheme((t) => (t === "light" ? "dark" : "light")),
    [],
  );
  return { theme, toggle };
}
```

- [ ] **Step 4: 통과 확인** — Run: `npx vitest run src/hooks/useTheme.test.ts` → 3 passed

- [ ] **Step 5: `tokens.css`·`globals.css` 작성**

`src/styles/tokens.css`: `web/detection/index.html`의 `:root`·`html[data-theme="light"]`·`html[data-theme="dark"]` 변수 블록을 그대로 이식하고, 글래스 토큰을 추가한다(라이트/다크 각각):
```css
/* 추가 글래스 토큰 (양 테마) — 정의된 보더 + 은은한 blur 로 라이트에서도 안 흐려지게 */
html[data-theme="light"] {
  --glass-bg: rgba(255,255,255,0.62);
  --glass-brd: rgba(16,18,24,0.10);
  --glass-blur: 14px;
  --depth: 0 8px 30px -12px rgba(16,18,24,0.28);
}
html[data-theme="dark"] {
  --glass-bg: rgba(20,20,24,0.55);
  --glass-brd: rgba(255,255,255,0.08);
  --glass-blur: 16px;
  --depth: 0 18px 44px -22px rgba(0,0,0,0.92);
}
.glass { background: var(--glass-bg); backdrop-filter: blur(var(--glass-blur)); -webkit-backdrop-filter: blur(var(--glass-blur)); border: 1px solid var(--glass-brd); box-shadow: var(--depth); }
```
`src/styles/globals.css`: `@tailwind base; @tailwind components; @tailwind utilities;` + `web/detection/index.html`의 전역 리셋·폰트·`html{font-size:18px}`·스크롤바·`prefers-reduced-motion` 블록 이식. `@import "./tokens.css";` 선행.

- [ ] **Step 6: `tailwind.config.ts` — 토큰을 theme로 노출**

```ts
import type { Config } from "tailwindcss";
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: ["selector", '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        sign: "var(--c-sign)", light: "var(--c-light)", board: "var(--c-board)",
        ink: "var(--ink)", "ink-2": "var(--ink-2)", "ink-3": "var(--ink-3)",
        surface: "var(--surface)", line: "var(--line)",
      },
      fontFamily: { sans: ["Fira Sans","Pretendard","system-ui","sans-serif"], mono: ["Fira Code","ui-monospace","monospace"] },
      borderRadius: { sm: "8px", DEFAULT: "12px", lg: "16px" },
    },
  },
} satisfies Config;
```

- [ ] **Step 7: `index.html`·`main.tsx` 연결** — `web_modern/index.html`에 `web/detection/index.html`의 폰트 `<link>`(Pretendard·Fira) + 테마 FOUC 부트스트랩 인라인 스크립트(라이트 기본) 이식. `main.tsx`에서 `import "./styles/globals.css"`.

- [ ] **Step 8: 전체 테스트 + 빌드** — Run: `npx vitest run` → 통과 · `npm run build` → OK

- [ ] **Step 9: 커밋**
```bash
git add web_modern && git commit -m "feat(web): 디자인 토큰·글래스·테마(라이트 기본, TDD)"
```

---

## Task 3: 타입 + API 클라이언트 (`lib/types.ts`·`lib/api.ts`, TDD)

**Files:** Create `web_modern/src/lib/types.ts`, `src/lib/api.ts`, `src/lib/api.test.ts`.

> 백엔드 계약 원천: `src/pipeline/app.py`(엔드포인트·메시지 형식)·`src/pipeline/e2e_pipeline.py`(`process_frame` 반환: `frame_id·inference_ms·tracks·variant·model_mb·stage_ms`)·`qa_bridge.py`(SSE `{type:context|token|done, text}`). 실행자는 이를 읽어 타입을 정확히 맞춘다.

- [ ] **Step 1: `src/lib/types.ts`** (백엔드 반환과 일치)
```ts
export interface Track { id: number; class: number; class_name: string; conf: number; label?: string; bbox: [number, number, number, number]; }
export interface StageMs { detect: number; track: number; recognize: number; }
export interface FrameResult { frame_id: number; inference_ms: number; tracks: Track[]; variant?: string; model_mb?: number; stage_ms?: StageMs; }
export interface VariantInfo { name: string; mb: number; }
export interface Status { yolo: boolean; ocr: boolean; tsign: boolean; taxonomy: string; variants: VariantInfo[]; active_variant: string | null; }
export type QAEvent = { type: "context" | "token"; text: string } | { type: "done" };
```

- [ ] **Step 2: SSE 파서 실패 테스트** (`src/lib/api.test.ts`) — 결정적으로 테스트 가능한 순수 파서만 검증
```ts
import { parseSSELine } from "./api";
test("SSE data 라인을 QAEvent로 파싱", () => {
  expect(parseSSELine('data: {"type":"token","text":"안"}')).toEqual({ type: "token", text: "안" });
  expect(parseSSELine("event: ping")).toBeNull();
  expect(parseSSELine("")).toBeNull();
});
```

- [ ] **Step 3: 실패 확인** — Run: `npx vitest run src/lib/api.test.ts` → FAIL

- [ ] **Step 4: `src/lib/api.ts` 구현** (순수 파서 + 클라이언트 함수)
```ts
import type { FrameResult, QAEvent, Status, Track } from "./types";

export function parseSSELine(line: string): QAEvent | null {
  if (!line.startsWith("data:")) return null;
  try { return JSON.parse(line.slice(5).trim()) as QAEvent; } catch { return null; }
}
export const wsBase = () => (location.protocol === "https:" ? "wss:" : "ws:") + "//" + location.host;

export async function getStatus(): Promise<Status> {
  const r = await fetch("/api/status"); return r.json();
}
export async function ingest(form: FormData): Promise<{ session_id?: string; error?: string }> {
  const r = await fetch("/api/ingest", { method: "POST", body: form }); return r.json();
}
// SSE Q&A — tracks+question(+apiKey) → onToken/onDone 콜백. (EventSource는 POST 불가 → fetch 스트림)
export async function askQA(
  tracks: Track[], question: string, apiKey: string | null,
  onEvent: (e: QAEvent) => void,
): Promise<void> {
  const r = await fetch("/api/qa", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tracks, question, api_key: apiKey }),
  });
  const reader = r.body!.getReader(); const dec = new TextDecoder(); let buf = "";
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n"); buf = parts.pop() ?? "";
    for (const p of parts) for (const ln of p.split("\n")) { const e = parseSSELine(ln); if (e) onEvent(e); }
  }
}
export type { FrameResult };
```

- [ ] **Step 5: 통과 확인** — Run: `npx vitest run src/lib/api.test.ts` → passed

- [ ] **Step 6: 커밋** — `git add web_modern && git commit -m "feat(web): 백엔드 타입 + API/SSE 클라이언트 (TDD)"`

---

## Task 4: 캔버스 한글 렌더 `lib/draw.ts` (최난도 패리티, TDD)

**Files:** Create `web_modern/src/lib/draw.ts`, `src/lib/draw.test.ts`.

> 원천: `web/detection/app.js`의 캔버스 렌더 함수(박스·라벨 pill·시그널색·좌표 스케일). 실행자는 그 로직을 읽어 **시각 결과를 1:1**로 옮긴다. 아래는 인터페이스·좌표수학·테스트를 고정하고, 실제 스타일(둥근 박스·pill·폰트)은 원천을 따른다.

- [ ] **Step 1: 좌표 매핑 실패 테스트** (순수 함수만 — 캔버스 컨텍스트 불필요)
```ts
import { mapBox, classColor } from "./draw";
test("원본 bbox를 표시 크기로 스케일", () => {
  // 원본 640x480 프레임의 [100,50,200,150] → 표시 1280x960(2x)
  expect(mapBox([100, 50, 200, 150], 640, 480, 1280, 960)).toEqual([200, 100, 400, 300]);
});
test("클래스별 시그널색", () => {
  expect(classColor(0)).toBe("#22c55e"); // traffic_sign
  expect(classColor(1)).toBe("#ef4444"); // traffic_light
  expect(classColor(2)).toBe("#f59e0b"); // signboard
});
```

- [ ] **Step 2: 실패 확인** — Run: `npx vitest run src/lib/draw.test.ts` → FAIL

- [ ] **Step 3: `src/lib/draw.ts` 구현** (순수 헬퍼 + 렌더)
```ts
import type { Track } from "./types";

const COLORS: Record<number, string> = { 0: "#22c55e", 1: "#ef4444", 2: "#f59e0b" };
export const classColor = (cls: number): string => COLORS[cls] ?? "#a1a1aa";

// 원본 프레임 좌표 → 표시 캔버스 좌표 (object-fit: contain 가정은 호출측에서 letterbox 보정)
export function mapBox(
  b: [number, number, number, number], srcW: number, srcH: number, dstW: number, dstH: number,
): [number, number, number, number] {
  const sx = dstW / srcW, sy = dstH / srcH;
  return [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy];
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath(); ctx.roundRect(x, y, w, h, r); }

// tracks 를 캔버스에 렌더. labelOf: track → 표시 문자열. 폰트는 호출 전 document.fonts.ready 보장.
export function renderTracks(
  ctx: CanvasRenderingContext2D, tracks: Track[],
  srcW: number, srcH: number, dstW: number, dstH: number,
): void {
  ctx.clearRect(0, 0, dstW, dstH);
  ctx.lineWidth = 2;
  ctx.font = '600 14px "Fira Sans","Pretendard",sans-serif';
  for (const t of tracks) {
    const [x1, y1, x2, y2] = mapBox(t.bbox, srcW, srcH, dstW, dstH);
    const c = classColor(t.class);
    ctx.strokeStyle = c; roundRect(ctx, x1, y1, x2 - x1, y2 - y1, 8); ctx.stroke();
    const label = `#${t.id} ${t.label ?? t.class_name}`;
    const pad = 6, tw = ctx.measureText(label).width;
    ctx.fillStyle = c; roundRect(ctx, x1, y1 - 22, tw + pad * 2, 20, 6); ctx.fill();
    ctx.fillStyle = "#0a0a0a"; ctx.fillText(label, x1 + pad, y1 - 8);
  }
}
```
> 정확한 색/둥글기/pill 위치·신뢰도 표시는 `web/detection/app.js` 원천과 시각 비교해 일치시킨다(스펙 §6).

- [ ] **Step 4: 통과 확인** — Run: `npx vitest run src/lib/draw.test.ts` → passed

- [ ] **Step 5: 커밋** — `git add web_modern && git commit -m "feat(web): 캔버스 한글 박스·라벨 렌더 lib/draw (TDD)"`

---

## Task 5: zustand store + WS/SSE/단축키 hooks (TDD)

**Files:** Create `src/store/index.ts` (+`store.test.ts`), `src/hooks/{useStream,useSession,useQA,useHotkeys}.ts` (+ 최소 1개 hook 테스트 `useQA.test.ts`).

> 원천: `web/detection/app.js`의 WS 메시지 처리(`/ws/stream` 송수신, `/ws/session` 제어/프레임), 누적카운트·텔레메트리 갱신, BYOK 저장. 실행자는 그 흐름을 hooks로 옮긴다.

- [ ] **Step 1: store 실패 테스트** (`src/store/store.test.ts`)
```ts
import { useStore } from "./index";
test("setFrame 시 트랙·텔레메트리·누적 갱신", () => {
  const s = useStore.getState();
  s.setFrame({ frame_id: 1, inference_ms: 12, tracks: [{ id: 1, class: 0, class_name: "traffic_sign", conf: 0.9, bbox: [0,0,1,1] }] });
  const st = useStore.getState();
  expect(st.tracks.length).toBe(1);
  expect(st.telemetry.inferenceMs).toBe(12);
  expect(st.totalDetections).toBe(1);
});
```

- [ ] **Step 2: 실패 확인** — Run: `npx vitest run src/store/store.test.ts` → FAIL

- [ ] **Step 3: `src/store/index.ts` 구현**
```ts
import { create } from "zustand";
import type { FrameResult, Track } from "../lib/types";

interface State {
  connected: boolean; sourceKind: "none" | "stream" | "session"; playing: boolean;
  tracks: Track[]; totalDetections: number;
  telemetry: { fps: number; inferenceMs: number; stageMs?: FrameResult["stage_ms"]; variant?: string; modelMb?: number };
  activeTab: "tracks" | "qa"; byokKey: string;
  setFrame: (r: FrameResult) => void;
  setConnected: (b: boolean) => void;
  setTab: (t: "tracks" | "qa") => void;
  setByok: (k: string) => void;
  reset: () => void;
}
export const useStore = create<State>((set) => ({
  connected: false, sourceKind: "none", playing: false,
  tracks: [], totalDetections: 0, telemetry: { fps: 0, inferenceMs: 0 },
  activeTab: "tracks", byokKey: localStorage.getItem("edge-sign-byok") ?? "",
  setFrame: (r) => set((s) => ({
    tracks: r.tracks, totalDetections: s.totalDetections + r.tracks.length,
    telemetry: { ...s.telemetry, inferenceMs: r.inference_ms, stageMs: r.stage_ms, variant: r.variant, modelMb: r.model_mb },
  })),
  setConnected: (b) => set({ connected: b }),
  setTab: (t) => set({ activeTab: t }),
  setByok: (k) => { localStorage.setItem("edge-sign-byok", k); set({ byokKey: k }); },
  reset: () => set({ tracks: [], totalDetections: 0, sourceKind: "none", playing: false }),
}));
```

- [ ] **Step 4: 통과 확인** — Run: `npx vitest run src/store/store.test.ts` → passed

- [ ] **Step 5: `useQA` 실패 테스트** (`src/hooks/useQA.test.ts`) — `askQA`를 모킹해 토큰 누적 검증
```ts
import { renderHook, act, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import * as api from "../lib/api";
import { useQA } from "./useQA";

test("ask 시 토큰을 순서대로 누적", async () => {
  vi.spyOn(api, "askQA").mockImplementation(async (_t, _q, _k, onEvent) => {
    onEvent({ type: "token", text: "안" }); onEvent({ type: "token", text: "녕" }); onEvent({ type: "done" });
  });
  const { result } = renderHook(() => useQA());
  await act(async () => { await result.current.ask([], "?"); });
  await waitFor(() => expect(result.current.answer).toBe("안녕"));
});
```

- [ ] **Step 6: 실패 확인** — Run: `npx vitest run src/hooks/useQA.test.ts` → FAIL

- [ ] **Step 7: hooks 구현** — `useQA.ts`(아래), `useStream.ts`·`useSession.ts`·`useHotkeys.ts`는 `web/detection/app.js` 원천 흐름을 옮긴다(WebSocket 송수신→`store.setFrame`, 캔버스는 `draw.renderTracks`, seek·속도·variant 제어 메시지, `NotSupportedError` 폴백, 단축키 Space/`/`/←→/T/?).
```ts
// src/hooks/useQA.ts
import { useState, useCallback } from "react";
import { askQA } from "../lib/api";
import { useStore } from "../store";
import type { Track } from "../lib/types";

export function useQA() {
  const [answer, setAnswer] = useState(""); const [streaming, setStreaming] = useState(false);
  const byok = useStore((s) => s.byokKey);
  const ask = useCallback(async (tracks: Track[], q: string) => {
    setAnswer(""); setStreaming(true);
    await askQA(tracks, q, byok || null, (e) => {
      if (e.type === "token") setAnswer((a) => a + e.text);
    });
    setStreaming(false);
  }, [byok]);
  return { answer, streaming, ask };
}
```

- [ ] **Step 8: 통과 확인** — Run: `npx vitest run` → store·useQA·draw·api·useTheme 전부 passed

- [ ] **Step 9: 커밋** — `git add web_modern && git commit -m "feat(web): zustand store + WS/SSE/단축키 hooks (TDD)"`

---

## Task 6: 앱 셸 + 뷰포트 + 클라 캡처 모드① (패리티)

**Files:** Create `src/components/{Header,Viewport,Hero,Splash,Controls}.tsx`, modify `src/App.tsx`.

> 원천 마크업: `web/detection/index.html`(header·KPI·hero·viewport·controls). 원천 동작: `app.js`(웹캠/파일→`/ws/stream`, 프레임 캡처·전송, 좌표 수신→`draw.renderTracks`). 실행자는 이를 컴포넌트+`useStream`으로 옮긴다. 글래스 토큰(`.glass`)·라이트 기본 적용.

- [ ] **Step 1: `App.tsx` 레이아웃** — header / main(stage + rail) / footer 골격, `useTheme`·`useHotkeys` 연결, `getStatus()`로 부팅 상태.
- [ ] **Step 2: `Header.tsx`** — 브랜드·KPI(FPS 스파크라인·지연·트랙·누적, store 구독)·상태 pill·테마 토글. 마크업은 원천 이식, 패널은 `.glass`.
- [ ] **Step 3: `Viewport.tsx`** — `<video>`+`<img>`(서버 모드)+`<canvas>` 오버레이. `useStream`이 프레임 수신 시 `renderTracks(ctx, tracks, srcW, srcH, canvas.width, canvas.height)` 호출. letterbox(object-fit contain) 보정 포함.
- [ ] **Step 4: `Hero.tsx`·`Splash.tsx`·`Controls.tsx`** — 원천 이식(샘플/웹캠/동영상/URL/파일·드래그). 웹캠·H.264 → `useStream` 시작.
- [ ] **Step 5: 컴포넌트 렌더 테스트** (`src/components/Header.test.tsx`) — store에 트랙 주입 시 KPI 트랙 수가 렌더되는지 RTL로 검증(1건).
- [ ] **Step 6: 빌드 + 수동 패리티** — 백엔드 실행(`uvicorn ... :8000`) 후 `npm run dev` → 웹캠/샘플로 **클라 캡처 모드 박스 렌더** 확인. `npx vitest run` 통과. `& "...convnext_env\python.exe" -m pytest tests/ -q` → 20 passed(백엔드 무변경 확인).
- [ ] **Step 7: 커밋** — `git add web_modern && git commit -m "feat(web): 앱 셸+뷰포트+클라 캡처 모드 (모드① 패리티)"`

---

## Task 7: 서버 인제스트 모드② + 통합 seek (패리티)

**Files:** Create `src/components/SeekBar.tsx`, modify `Viewport.tsx`/`Controls.tsx`/hooks.

> 원천: `app.js`의 `/api/ingest`→`/ws/session` 흐름, `NotSupportedError` 자동 폴백, 통합 seekbar(클라 native currentTime · 서버 pos+seek 드래그), 속도·5초 스텝. `useSession` 으로 옮긴다.

- [ ] **Step 1: `useSession`** — `ingest(form)`→session_id→`WS /ws/session`. 프레임 메시지(JSON+JPEG bytes) 수신: `<img>`에 Blob URL, `store.setFrame`, `renderTracks`. 제어(play/pause/seek/speed/variant) 송신.
- [ ] **Step 2: 자동 폴백** — `<video>` 로드 `error`(`NotSupportedError`) 또는 URL/이미지 → 모드② 전환(원천 로직).
- [ ] **Step 3: `SeekBar.tsx`** — 드래그→(모드① video.currentTime / 모드② seek 제어). pos·total·fps로 진행, 라이브는 비활성. 마크업 원천 이식.
- [ ] **Step 4: 빌드 + 수동 패리티** — MPEG-4/이미지/URL 업로드 → 서버 주석 프레임 표시·seek 드래그 확인. `npx vitest run` 통과.
- [ ] **Step 5: 커밋** — `git add web_modern && git commit -m "feat(web): 서버 인제스트 모드②+자동 폴백+통합 seek (모드② 패리티)"`

---

## Task 8: PerfStrip(A/B·단계 플로우) + Rail(Tracks·QA) + 모달 (패리티)

**Files:** Create `src/components/{PerfStrip,Rail,TracksPanel,QAPanel,Toast,ShortcutsModal}.tsx`.

> 원천: `app.js`의 variant A/B 토글(크기·FPS Δ)·단계 플로우(병목 강조)·트랙 리스트·SSE Q&A(`useQA`)·퀵칩·BYOK·토스트·단축키 모달. Radix `Tabs`(레일)·`Dialog`(단축키)·`Slider`(seek/속도) 사용.

- [ ] **Step 1: `PerfStrip.tsx`** — `status.variants`로 A/B 세그 토글(전환 시 store/variant + WS 제어), `telemetry.stageMs`로 검출/추적/인식 바·병목 강조, 크기·FPS Δ.
- [ ] **Step 2: `Rail.tsx`+`TracksPanel.tsx`** — Radix Tabs(인식 결과/주행 어시스턴트). 트랙 리스트(신뢰도 바·클래스색, store 구독, 애니메이션).
- [ ] **Step 3: `QAPanel.tsx`** — `useQA`로 스트리밍 채팅, 퀵칩, BYOK 입력(`store.setByok`, password 토글). 전송 시 현재 `store.tracks` 전달.
- [ ] **Step 4: `Toast.tsx`·`ShortcutsModal.tsx`** — 토스트 큐, Radix Dialog 단축키(원천 항목).
- [ ] **Step 5: 컴포넌트 테스트** (`QAPanel.test.tsx`) — `useQA.ask` 모킹, 입력·전송 시 답변 영역에 토큰 렌더 검증.
- [ ] **Step 6: 빌드 + 수동 패리티(§8 전체)** — A/B 전환·단계 바·트랙·Q&A 스트리밍·BYOK·토스트·단축키·테마·반응형 확인. `npx vitest run` 통과.
- [ ] **Step 7: 커밋** — `git add web_modern && git commit -m "feat(web): PerfStrip+Rail+QA+모달 (패리티 완성)"`

---

## Task 9: 패리티 검증 + 컷오버 (app.py dist-우선+레거시 폴백)

**Files:** Modify `src/pipeline/app.py` (StaticFiles 마운트), `web_modern/` 빌드.

- [ ] **Step 1: 패리티 체크리스트 통과** — 스펙 §8 전 항목을 vanilla(`/detection-legacy/` 역할의 기존 `/detection/`)와 나란히 수동 검증. 미달 항목은 해당 Task로 돌아가 보완.

- [ ] **Step 2: 프로덕션 빌드** — Run: `cd web_modern && npm run build` → `web_modern/dist/` 생성.

- [ ] **Step 3: `app.py` 마운트 컷오버** — 현재 정적 마운트 블록(`web/detection`)을 아래로 교체. dist 있으면 우선, 없으면 레거시 폴백 + vanilla 항상 `/detection-legacy/`.
```python
WEB_LEGACY = ROOT / "web" / "detection"
WEB_DIST = ROOT / "web_modern" / "dist"
# 신규 React 빌드가 있으면 /detection/ 에 서빙, 없으면 레거시 폴백(HF는 SP-D 빌드 전까지 레거시).
_web_new = WEB_DIST if WEB_DIST.exists() else WEB_LEGACY
if _web_new.exists():
    app.mount("/detection", StaticFiles(directory=str(_web_new), html=True), name="detection")
if WEB_LEGACY.exists():
    app.mount("/detection-legacy", StaticFiles(directory=str(WEB_LEGACY), html=True), name="detection-legacy")
```
> 기존 `WEB_DIR = ROOT/"web"/"detection"` 및 그 마운트 라인을 위 블록으로 대체. 변수명이 다르면 실행자가 맞춘다. **API 핸들러는 무변경**.

- [ ] **Step 4: 백엔드 회귀 + 정적/타입 게이트** (convnext_env)
Run: `& "C:\Users\leegy\miniconda3\envs\convnext_env\python.exe" -m pytest tests/ -q` → 20 passed
Run: `& "...python.exe" -m ruff check src/pipeline/app.py` → All checks passed
Run: `& "...python.exe" -m mypy src/pipeline` → Success (app.py 마운트 변경 후에도 0)

- [ ] **Step 5: 통합 스모크** — `uvicorn` 실행 → 브라우저 `http://localhost:8000/detection/`(신규 React) · `/detection-legacy/`(vanilla) 둘 다 동작 확인.

- [ ] **Step 6: 커밋** — `git add web_modern src/pipeline/app.py && git commit -m "feat(web): React 빌드 컷오버 — /detection 신규, /detection-legacy 폴백 (dist 우선)"`

---

## 완료 기준 (스펙 §1·§8 대조)

- [ ] `web_modern/` React 앱이 §8 패리티 체크리스트 전 항목 충족(수동 검증)
- [ ] `npx vitest run` 전 테스트 통과(useTheme·api·draw·store·useQA·컴포넌트)
- [ ] `npm run build` 성공 → `dist/` 서빙
- [ ] 글래스모피즘 양 테마(라이트 기본)·AA·캔버스 한글 라벨 정상
- [ ] **백엔드 무변경 증명**: `pytest tests/` 20 passed · `mypy src/pipeline` 0 · `ruff check` 0 (시작 시와 동일)
- [ ] `/detection/`(신규)·`/detection-legacy/`(vanilla) 동시 동작

## 참고 / 다음 단계

- **HF Dockerfile 멀티스테이지(Node 빌드→dist COPY)는 SP-D 소관.** SP-C는 로컬 빌드+컷오버까지. dist 폴백 로직 덕에 HF는 SP-D 전까지 레거시로 무중단.
- UI 컴포넌트 태스크(T6–T8)는 `web/detection/{index.html,app.js}`를 동작·마크업 명세로 읽어 포팅 — 색/간격/애니메이션/단축키를 시각 비교로 일치시킨다.
- 커밋: 한국어 conventional-commit, Co-Authored-By 줄 없음.
