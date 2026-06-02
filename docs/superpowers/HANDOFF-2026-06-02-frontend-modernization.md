# SP-C 프론트엔드 현대화 — 상태: 코드 완성 (2026-06-02)

> **T1–T9 전부 구현·커밋 완료.** 브랜치 `feat/frontend-modernization` (main 미병합).
> 남은 것은 **(1) 사용자 브라우저 시각 패리티 검증(§8), (2) 머지** 두 단계뿐.

## ✅ 완료 (T1–T9, 커밋됨)
- T1 스캐폴드 · T2 디자인토큰/글래스/테마(라이트 기본) · T3 api/타입 · T4 캔버스 한글 렌더(`draw.ts`) · T5 store+hooks · T6 셸+뷰포트+모드① · T7 서버 인제스트 모드②+seek · T8 PerfStrip+Rail+Q&A+모달 · T9 컷오버.
- **검증된 것:** `npm run build` OK · `vitest` 19 passed(7 files) · `tsc` clean · 백엔드 `pytest` 20 · `ruff` 0 · `mypy src/pipeline` 0 · 컷오버 스모크(`/detection/`→REACT 200·에셋 200, `/detection-legacy/`→VANILLA 200).
- **컷오버:** `app.py` `/detection/` = `web_modern/dist` 우선 + 없으면 레거시 폴백, vanilla는 `/detection-legacy/` 유지. `vite base "./"`(상대)로 하위경로 서빙 에셋 정확. dist는 gitignore(로컬 `npm --prefix web_modern run build` 필요).

## ⏳ 남은 작업 (사용자/브라우저 필요 — 코드 아님)
1. **시각 패리티 검증 (§8)** — 서브에이전트·컨트롤러는 픽셀을 못 봄. 사용자가 직접:
   ```
   # 백엔드
   & "C:\Users\leegy\miniconda3\envs\convnext_env\python.exe" -m uvicorn src.pipeline.app:app --port 8000
   # (이미 dist 빌드됨) 브라우저:
   #   http://localhost:8000/detection/         ← 신규 React (글래스·라이트 기본)
   #   http://localhost:8000/detection-legacy/  ← 기존 vanilla (비교 기준)
   ```
   §8 체크리스트(스펙) 항목을 나란히 비교: 박스 렌더·2모드·seek·A/B·단계 플로우·트랙·Q&A·BYOK·토스트·단축키·테마·반응형. (개발 중엔 `npm --prefix web_modern run dev` + 위 백엔드 → http://localhost:5173 )
   - 알려진 확인 포인트: 캔버스 한글 라벨, 모드② 서버프레임 표시, hover↔박스, 글래스 라이트 대비(AA).
2. **머지** — 패리티 OK면 `superpowers:finishing-a-development-branch`로 main 병합(또는 PR). 미흡 항목 있으면 해당 컴포넌트 보완 후 재빌드.

## 참고
- **QAPanel.test.tsx 미작성**(세션 중단) — Q&A 핵심 로직은 `useQA.test.tsx`(T5)가 커버. 원하면 추가.
- **HF Dockerfile 멀티스테이지(Node 빌드→dist COPY)는 SP-D 소관.** dist 폴백 덕에 SP-D 전까지 HF는 레거시로 무중단.
- main이 origin보다 앞섬(SP-A+loguru, 미push). SP-C 브랜치도 미병합.
- 커밋 스타일: 한국어 conventional, Co-Authored-By 없음.

## 브랜치 커밋 (SP-C)
```
169fe5a T9 컷오버 · 8a1ddd6 T8 · 814cf57 T7 · 3c7ed03 T6 · e6e033d T5 · b2e5689 T4 · 6a37499 T3 · a16436d T2 · c2888e6 T1 · 5146927 plan · 73e1711 spec
```
