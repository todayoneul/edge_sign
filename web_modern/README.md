# Edge-Sign 웹 프런트엔드

Edge-Sign의 검출·추적·인식 파이프라인을 브라우저에서 실행하는 React 애플리케이션이다. 모델은 ONNX Runtime Web으로 불러와 사용자의 기기에서 추론하며, 검출기와 정밀도, 실행 환경(WebGPU, WebAssembly)을 화면에서 바꿀 수 있다.

## 구성

| 경로 | 내용 |
| :--- | :--- |
| `src/lib/clientPipeline.ts` | 브라우저 추론 파이프라인(전처리, 검출, 디코딩, 인식) |
| `src/lib/byteTrack.ts` | ByteTrack의 TypeScript 구현 |
| `src/lib/models.ts` | 검출기, 정밀도, 실행 환경의 목록 |
| `src/components/`, `src/hooks/`, `src/store/` | 화면 구성과 상태 관리 |
| `public/sample/` | 내장 샘플 영상 |

## 개발

FastAPI 서버(저장소 루트에서 `uvicorn src.pipeline.app:app --port 8000`)를 먼저 띄운 뒤 개발 서버를 실행한다. 개발 서버는 `/api`, `/ws` 요청을 8000번 포트로 넘긴다.

```bash
npm ci
npm run dev      # http://localhost:5173
npm test         # Vitest
npm run build    # dist/ 생성, FastAPI가 /detection/ 경로로 제공
```
