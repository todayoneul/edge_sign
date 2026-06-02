# 프론트엔드 현대화 (SP-C) — 설계 문서

- **작성일**: 2026-06-02
- **상태**: 설계 합의 완료 (구현 계획 대기)
- **범위**: SP-C (React+Vite+TS 프론트 재구축 + 글래스모피즘 리디자인). 상용화 고도화 3단계.
- **선행/후속**: SP-A(코드품질) 완료. SP-B(CI) 보류. **SP-D(docker-compose/HF 멀티스테이지 빌드)와 연계** — 본 작업이 Node 빌드 스테이지를 요구.

---

## 1. 목표 / 동기

> "이미 완성도 높은 vanilla 콘솔을, 동작 회귀 없이 React+TS 컴포넌트 아키텍처 + 글래스모피즘 리디자인으로 끌어올린다."

현 `web/detection/`(index.html 959줄 + app.js 874줄)은 **시각적으로 이미 프리미엄**(OLED 모노크롬 콘솔·다크/라이트·마이크로 애니메이션·접근성)이다. 따라서 SP-C의 가치는 **두 가지**다:
1. **아키텍처**: 모놀리식 HTML+JS → React+TypeScript 컴포넌트(유지보수·타입안전·재사용·테스트 가능).
2. **비주얼**: 새 디렉션 = **글래스모피즘 + 깊이감**, 양 테마(**라이트 기본** + 다크), WCAG AA.

핵심 제약: 어렵게 완성된 **동작들(2모드 WS·캔버스 한글 렌더·양자화 A/B·통합 seek·SSE Q&A·BYOK·텔레메트리)을 패리티로 보존**하고, 기존 데모·HF 배포를 **무중단**으로 둔다.

### 성공 기준
- `web_modern/` React 앱이 vanilla `web/detection/`의 **전 기능을 패리티**로 재현(아래 §8 체크리스트).
- 글래스모피즘 양 테마(라이트 기본) · AA 대비 · 캔버스 한글 라벨 정상 렌더.
- **백엔드 API 무변경** — 기존 pytest 그린 유지.
- 개발 시 `vite dev`(프록시), 프로덕션 시 `vite build`→FastAPI가 `dist/` 서빙(HF 단일 컨테이너 호환).
- 컷오버 전까지 vanilla 데모가 `/detection-legacy/`로 계속 동작(폴백).

---

## 2. 결정 사항 (합의됨)

| 항목 | 결정 |
| :--- | :--- |
| **마이그레이션** | **A안 — 병행 구축 + 컷오버**: `web_modern/` 신규 구축, 패리티 도달 시 `/detection/` 전환, vanilla는 `/detection-legacy/` 폴백 |
| **스택** | React 18 + Vite + TypeScript |
| **스타일링** | Tailwind CSS + 커스텀 디자인 토큰, Radix UI 프리미티브(Dialog/Tabs/Slider 등) |
| **비주얼** | 글래스모피즘 + 깊이감, **라이트 기본** + 다크, 시그널색(표지/신호등/간판) 보존, AA |
| **상태관리** | zustand (연결·세션·트랙·텔레메트리·테마·variant·BYOK) |
| **백엔드** | **무변경** (정적 서빙 마운트 경로만 컷오버 시 변경) |
| **테스트** | Vitest + React Testing Library (store·hooks·draw 로직) + 수동 패리티 체크리스트 |
| **빌드/배포** | vite dev 프록시 / vite build → FastAPI 서빙 / **HF Dockerfile 멀티스테이지(Node 빌드)** |

---

## 3. 아키텍처 / 디렉토리

```
web_modern/
├── index.html                  # Vite 엔트리 (폰트 preconnect)
├── vite.config.ts              # 프록시(/ws·/api→:8000, ws:true), build outDir
├── tailwind.config.ts          # 디자인 토큰(glass·signal·spacing) theme 확장
├── tsconfig.json · package.json
├── src/
│   ├── main.tsx · App.tsx
│   ├── components/             # 프레젠테이션 컴포넌트(단일 책임)
│   │   ├── Header.tsx(KPI·테마·상태) · Viewport.tsx(video·img·overlay canvas)
│   │   ├── SeekBar.tsx · PerfStrip.tsx(A/B 토글 + 단계 플로우) · Controls.tsx
│   │   ├── Rail.tsx → TracksPanel.tsx · QAPanel.tsx(채팅·퀵칩·BYOK)
│   │   ├── Toast.tsx · ShortcutsModal.tsx · Splash.tsx · Hero.tsx
│   ├── hooks/                  # useStream(클라캡처) · useSession(서버스트림) · useQA(SSE) · useTheme · useHotkeys
│   ├── store/                  # zustand 슬라이스(session·tracks·telemetry·ui·qa)
│   ├── lib/
│   │   ├── api.ts              # REST(/api/ingest·status) · WS(/ws/stream·session) · SSE(/api/qa) 클라이언트
│   │   └── draw.ts             # 캔버스 박스·한글 라벨 렌더 (둥근 박스·pill·시그널색) — 최난도 패리티
│   └── styles/                 # tokens.css(글래스·테마 변수) · globals.css
└── (build) dist/               # vite build 산출 → FastAPI 서빙
```

**격리 원칙**: 각 컴포넌트는 단일 책임 + 명확한 props 인터페이스. WS/SSE/캔버스 같은 부수효과는 hooks로 분리(컴포넌트는 순수 렌더에 집중). `lib/`는 프레임워크 비의존(테스트 용이).

---

## 4. 백엔드 연동 (API 무변경)

기존 계약을 그대로 사용한다(서버 코드 변경 없음):

| 용도 | 엔드포인트 | 클라이언트 |
| :--- | :--- | :--- |
| 상태 | `GET /api/status` | 부팅 시 variant·taxonomy 조회 |
| 클라 캡처(모드①) | `WS /ws/stream` | 웹캠·H.264 프레임(+variant) 전송 → 좌표 JSON 수신 → 캔버스 렌더 |
| 서버 인제스트 | `POST /api/ingest` → `WS /ws/session` | 비호환 코덱·URL·이미지 → 서버 디코딩 → 주석 JPEG + 좌표 수신 |
| Q&A | `POST /api/qa` (SSE) | tracks + 질문(+BYOK 키) → 스트리밍 답변 |

- **2모드 자동 폴백**: `<video>` 로드 실패(`NotSupportedError`) 감지 → 서버 인제스트로 전환(현 동작 이식).
- **통합 seek**: `/ws/session`의 `pos·total·fps·seekable`로 드래그 탐색(현 동작 이식).
- **컷오버 시 유일한 백엔드 변경**: `app.py`의 `/detection/` StaticFiles 마운트를 **`web_modern/dist`가 존재하면 그것을, 없으면 `web/detection`(레거시)을 서빙**하도록 변경(빌드 산출물 부재 시 안전 폴백 → `dist`는 gitignore 산출물이라 HF 이미지엔 SP-D 빌드 스테이지 전까지 없음 → 그래도 안 깨짐). vanilla는 항상 `/detection-legacy/`에도 유지. **API 로직 무변경**.

---

## 5. 빌드 / 서빙 / 배포

- **개발**: `npm run dev`(Vite, 별도 포트). `vite.config.ts` 프록시로 `/api`·`/ws`(`ws:true`)를 FastAPI `:8000`에 연결. 백엔드는 평소처럼 `uvicorn`.
- **프로덕션(로컬·HF)**: `npm run build` → `web_modern/dist/`. FastAPI가 `dist/`를 정적 서빙(단일 origin → CORS·WS 동일 호스트). HF는 단일 컨테이너 유지.
- **HF Dockerfile(SP-D 연계)**: 멀티스테이지 — `node:20` 스테이지에서 `web_modern` 빌드 → `dist/`만 python 런타임 스테이지로 COPY. 이미지에 Node 런타임 미포함(빌드 산출물만). 본 spec은 이 변경의 **요구사항만** 정의하고, 실제 Dockerfile 수정은 **SP-D 소관**.
- **WS 프로토콜 자동선택**: 현 코드처럼 `https→wss` 자동(이식).

---

## 6. 캔버스 한글 렌더 (최난도 패리티 — `lib/draw.ts`)

서버 `cv2.putText`는 한글 미지원이라 **박스·라벨을 클라이언트가 캔버스에 직접 렌더**한다(두 모드 라벨링 일치의 핵심). vanilla `app.js`의 렌더 로직을 `draw.ts`로 정밀 이식한다:
- 클래스별 시그널색(표지=녹색/신호등=빨강/간판=주황), 둥근 박스, track ID·라벨 pill, 신뢰도.
- 좌표 스케일링: 모드①(원본 프레임→표시 캔버스), 모드②(서버 `w·h` 기준).
- **폰트 보장**: `document.fonts.ready` 후 렌더(한글 웹폰트 로드 전 그리면 깨짐).
- 렌더는 `requestAnimationFrame`/프레임 수신 시 호출(핫패스 — 객체 할당 최소화).

> 이 모듈은 프레임워크 비의존 순수 함수(canvas ctx + tracks → 그림)로 작성해 단위 테스트(Vitest)로 좌표·라벨 매핑을 검증한다.

---

## 7. 상태 / 데이터 흐름

- **zustand 슬라이스**: `session`(소스종류·재생·속도·pos), `tracks`(현 프레임 트랙·누적 카운트), `telemetry`(fps·지연·stage_ms·model_mb·variant), `ui`(테마·활성탭·토스트·모달), `qa`(메시지·스트리밍중·BYOK 키).
- **hooks**가 WS/SSE 수신 → store 갱신 → 컴포넌트 구독 렌더. 캔버스는 store 트랙을 구독해 `draw.ts` 호출.
- **테마**: `useTheme`(localStorage `edge-sign-theme`, 라이트 기본) — FOUC 방지 인라인 부트스트랩(현 패턴 이식).
- **BYOK 키**: localStorage 저장, `/api/qa` 요청 시에만 전송(현 보안 모델 이식).

---

## 8. 패리티 체크리스트 (vanilla 대비 — 컷오버 게이트)

- [ ] 스플래시 · 히어로(샘플/웹캠/동영상 CTA) · KPI(FPS 스파크라인·지연·트랙·누적)
- [ ] 뷰포트 + 오버레이 캔버스(한글 라벨) · 코너 틱
- [ ] 2모드 WS(클라 캡처 / 서버 인제스트) + `NotSupportedError` 자동 폴백
- [ ] 통합 seek 바(드래그·재생/정지·시간) · 속도 · 5초 스텝
- [ ] perf-strip: 양자화 A/B 토글(크기·FPS Δ) + 단계 플로우(검출/추적/인식·병목 강조)
- [ ] 입력: 파일·드래그·웹캠·URL/RTSP·이미지
- [ ] 트랙 패널(신뢰도 바·클래스색) · Q&A(스트리밍·퀵칩·BYOK) · 토스트 · 단축키 모달
- [ ] 테마 토글(라이트 기본) · 반응형 · `prefers-reduced-motion` · ARIA/포커스

---

## 9. 테스트

- **단위(Vitest + RTL)**: `lib/draw.ts`(좌표·라벨 매핑), store 슬라이스(상태 전이), hooks(WS/SSE 메시지→상태, 모킹 소켓).
- **백엔드 회귀**: 기존 `pytest tests/`(20) 무변경 유지 — 백엔드 손대지 않음.
- **수동 패리티**: §8 체크리스트를 vanilla와 나란히 검증(컷오버 전 게이트).

---

## 10. 범위 밖 (YAGNI)

- 백엔드 API/파이프라인 변경, 패리티 외 신기능.
- 전체 클라이언트 사이드 ONNX 추론(WASM/WebGPU) — 장기 목표.
- SSR/Next.js(단일 SPA로 충분), i18n 프레임워크(현재 한국어 단일).
- E2E 브라우저 자동화(Playwright) — 후속 고려, SP-C는 단위+수동 패리티.
- Phase-1 `web/`(OCR 캔버스 데모)는 손대지 않음 — `web/detection/`만 대상.

---

## 11. 단계적 구현 (계획 개요)

1. 스캐폴드(Vite+TS+Tailwind+Radix, 프록시) + 디자인 토큰/테마(글래스·라이트 기본)
2. 앱 셸(Header·KPI·테마·Splash·Hero) + 라우팅/레이아웃
3. 뷰포트 + `lib/draw.ts`(한글 캔버스) + `useStream`(모드①) — 코어 인식 루프
4. 서버 인제스트(`useSession`·`/api/ingest`) + 자동 폴백 + 통합 seek
5. PerfStrip(A/B·단계 플로우) + Controls(입력 전종) + 텔레메트리
6. Rail: TracksPanel + QAPanel(SSE·퀵칩·BYOK) + Toast·ShortcutsModal
7. 패리티 검증(§8) → 컷오버(app.py 마운트 전환, vanilla→legacy) + HF Dockerfile 멀티스테이지(SP-D 연계)

---

## 12. 의존 관계 / 비고

- 백엔드는 `convnext_env`에서 실행(무변경). 프론트 빌드는 Node 20(개발 머신 또는 Docker 빌드 스테이지).
- **컷오버(app.py 마운트 dist-우선+레거시 폴백 로직)는 SP-C 말미**에 수행 — `dist` 부재 시 레거시 폴백이라 HF는 영향 없음. **HF Dockerfile 멀티스테이지(Node 빌드→dist COPY)는 SP-D 소관**(컨테이너 빌드 주체). 이 폴백 로직이 두 시점을 안전하게 연결.
- 커밋 스타일: 한국어 conventional-commit, **Co-Authored-By 줄 없음**. 브랜치 `feat/frontend-modernization`.
