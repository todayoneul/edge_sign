# SP-C 프론트엔드 현대화 — 진행 핸드오프 (2026-06-02)

> 기반 레이어(T1–T5) 완료 시점 체크포인트. 다음 세션에서 **T6부터** 이어간다.
> 실행: **superpowers:subagent-driven-development** (태스크별 새 서브에이전트 + 리스크별 리뷰). 계획 전문: `docs/superpowers/plans/2026-06-02-frontend-modernization.md`, 설계: `docs/superpowers/specs/2026-06-02-frontend-modernization-design.md`.

## ⚠️ 환경 / 규칙
- 브랜치 **`feat/frontend-modernization`** (main 미병합). 프론트는 `web_modern/`.
- **cwd:** `npm`/`vitest`는 `web_modern/`에서 (`npm --prefix web_modern run test`, 또는 `cd web_modern && npx vitest run <file>`). `git`은 저장소 루트. PowerShell `cd` 프롬프트 시 `--prefix` 사용.
- 백엔드 pytest(T9 게이트)는 convnext_env: `& "C:\Users\leegy\miniconda3\envs\convnext_env\python.exe" -m pytest tests/ -q` → 20 passed.
- **`web/`·`src/`(백엔드)·`tests/` 무변경** (T9 컷오버에서 app.py 마운트만). 커밋: 한국어 conventional, **Co-Authored-By 없음**.
- 스택 실측: Vite 8 · React 19 · Tailwind **3.4.19 (`.cjs` 설정, CommonJS)** · Vitest 4 · Node 24.

## 커밋 이력 (이 브랜치)
```
e6e033d  T5 store + WS/SSE/단축키 hooks
b2e5689  T4 캔버스 한글 렌더 lib/draw
6a37499  T3 백엔드 타입 + API/SSE 클라
a16436d  T2 디자인 토큰·글래스·테마(라이트 기본)
c2888e6  T1 web_modern 스캐폴드
5146927  (plan)  ·  73e1711 (spec)
```

## 완료 (T1–T5) — 기반 레이어, 5 test files 통과·build OK
- **T1 스캐폴드**: `web_modern/`(Vite+React+TS+Tailwind3.4.cjs+Vitest), `vite.config.ts` 프록시(`/api`·`/ws` ws:true·`/detection`→:8000), `.gitignore`(node_modules/dist).
- **T2 디자인시스템**: `src/styles/tokens.css`(vanilla 토큰 이식 + 글래스 `--glass-*`/`.glass`), `globals.css`, `tailwind.config.cjs`(토큰 노출, darkMode selector), `src/hooks/useTheme.ts`(**라이트 기본**·persist, TDD), `index.html` 폰트+FOUC 부트스트랩.
- **T3 api**: `src/lib/types.ts`(Track·FrameResult·StageMs·Status·VariantInfo·QAEvent — 백엔드 대조 검증됨), `src/lib/api.ts`(`getStatus`·`ingest`·`askQA`(fetch SSE)·`parseSSELine`·`wsBase`, TDD).
- **T4 draw**: `src/lib/draw.ts`(`classColor`·`mapBox`·`renderTracks` — app.js 255-296 시각 1:1 이식, 순수함수, TDD). **연기→T6**: letterbox 오프셋(object-fit contain), hover 하이라이트(hoverId).
- **T5 store/hooks**: `src/store/index.ts`(zustand: connected·sourceKind·playing·tracks·totalDetections·telemetry·activeTab·byokKey; `setFrame`·`setConnected`·`setTab`·`setByok`·`reset`, TDD). hooks: `useStream`(/ws/stream 모드①)·`useSession`(/api/ingest→/ws/session 모드②)·`useQA`(SSE, TDD)·`useHotkeys`(Space·/·←→·T·?·Esc). WS 프로토콜 app.js 대조됨.

### ⚠️ T6/T7가 wiring할 통합 계약(T5가 의도적으로 연기한 DOM 부분)
- `useStream.start(getFrame)`: **Viewport가 `getFrame()` 콜백 제공**(캔버스 캡처 `canvas.toDataURL('image/jpeg')`). 훅은 소켓+store만.
- `useSession`: Viewport가 `frameBlobUrl`을 `<img>`에 표시, `seekInfo`(pos·total·fps·seekable) 사용.
- `useHotkeys`: Viewport가 콜백 주입(togglePlay·focusChat(ref.focus)·toggleTheme·toggleShortcuts·step±5s).
- `draw.renderTracks`: Viewport가 object-fit contain **letterbox 오프셋 계산** 후 src/dst dims 전달. hover 하이라이트는 Viewport 책임(draw 확장 또는 트랙 필터).

## 남은 작업 (T6–T9) — 계획 문서의 해당 Task 전문 사용
- **T6 셸+뷰포트+모드①**: `App`·`Header`(KPI·테마·상태)·`Viewport`(video·img·overlay canvas + `renderTracks` + letterbox/hover)·`Hero`·`Splash`·`Controls`. `useStream` 클라 캡처 배선. 마크업·동작은 `web/detection/index.html`+`app.js` 이식. `.glass` 적용. **수동 패리티**: 웹캠/샘플 → 박스 렌더(백엔드 `uvicorn :8000` + `npm --prefix web_modern run dev`).
- **T7 모드②+seek**: `useSession`/`ingest`/자동폴백(`NotSupportedError`)/`SeekBar`(통합 드래그). 패리티: MPEG-4/이미지/URL.
- **T8 PerfStrip+Rail+Q&A+모달**: A/B 토글·단계 플로우(병목)·`TracksPanel`·`QAPanel`(useQA·퀵칩·BYOK)·`Toast`·`ShortcutsModal`(Radix Dialog/Tabs/Slider). 패리티 §8 전체.
- **T9 컷오버**: `npm run build` → `app.py` `/detection/` 마운트를 **dist-우선+레거시 폴백**으로(계획 T9 Step3 전체 코드), vanilla→`/detection-legacy/`. 백엔드 게이트: pytest 20·ruff·mypy. 스모크: `/detection/`(신규)·`/detection-legacy/`(vanilla).

## 완료 후
- 패리티 게이트(§8) 통과 → `superpowers:finishing-a-development-branch`(머지/PR).
- **HF Dockerfile 멀티스테이지(Node 빌드→dist COPY)는 SP-D 소관.** dist 폴백 덕에 HF는 SP-D 전까지 레거시로 무중단.
- 리뷰 배분: 보일러/기계적(T6 일부)은 컨트롤러 검증, 로직/통합·컷오버(T7·T9)는 풀 리뷰 권장.
