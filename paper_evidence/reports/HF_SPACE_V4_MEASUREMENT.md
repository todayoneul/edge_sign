# 공개 Hugging Face Space YOLO26 v4 측정: 한눈에 보기

측정일: 2026-09-25. 공개 [Edge-Sign Space](https://huggingface.co/spaces/gyann/edge-sign)에 논문용 **YOLO26-n v4 전용 경로**(`/ws/paper-v4`)를 추가해 배포하고 측정했다. 하드웨어는 `cpu-basic`, Space 소스 commit은 `f645ad510f3fcbdccab4d8cda21d6de62ccaa4e2`이다. 이 commit은 v3 commit `9fc2593`에 v4 경로와 v4 모델 두 개만 더한 것으로, 웹 화면과 v3 `/ws/stream`은 그대로이다. 배포 뒤 v3 `/api/status`(FP32·INT8 변형), `/detection/`, 샘플 영상이 모두 정상 응답했다.

v3 측정과 같은 공개 샘플 15프레임(같은 SHA-256)을 반복하며 초당 10프레임을 보냈다. 각 구성의 추적기를 초기화하고 10프레임 준비 실행 후 50프레임을 측정했다. 브라우저 화면 그리기는 포함하지 않았다.

| Space v4 | Space 검출→추적→인식 평균 / p95 | 검출기 평균 | 결과 수신 속도 | 첫 송신→마지막 수신 속도 | 선택 모델 파일 합계 |
|---|---:|---:|---:|---:|---:|
| head-excluded INT8 | **50.278 / 73.569 ms** | 46.723 ms | 12.182 FPS | 9.662 FPS | **3,533,232 B** |
| FP32 | **77.619 / 98.077 ms** | 73.706 ms | 11.884 FPS | 9.509 FPS | 9,922,835 B |

**해석:**

- v4 파이프라인(50–78 ms)이 입력 간격(100 ms)보다 짧아 **Space가 포화되지 않았다.** 완료 속도(9.5–9.7 FPS)는 입력 속도 10 FPS를 따라간 값이고, 결과 수신 속도가 10을 넘는 것은 측정 시작 시점에 준비 실행의 대기열이 비워졌기 때문이다. 따라서 이 수치는 Space의 최대 처리량이 아니며, 포화 상태였던 v3 측정(349–456 ms)과 처리량으로 비교하지 않는다.
- 서버 내부 평균 50.3 ms는 네트워크를 빼더라도 단일 스트림 약 20 FPS에 해당해 **30 FPS에 미치지 못한다.** 30 FPS 판정에는 포화 조건(예: 30 FPS 이상 입력)을 미리 정해 측정하고 브라우저 경로도 포함해야 한다.
- 이 Linux `cpu-basic` 환경에서는 head-excluded INT8 검출기가 FP32보다 **1.58배 빨랐다**(46.7 vs 73.7 ms). 로컬 Windows ORT CPU 측정(16.1 vs 21.6 ms)과는 반대 결과다. CPU·OS·스레드 수가 다르고 Space 수치는 공유 하드웨어의 단일 실행이므로, 일반적인 INT8 가속 주장이 아니라 **런타임에 따라 결과가 뒤집히는 관찰**로만 쓴다.
- v4 경로와 v3 경로는 디코더, 임계값, 추적기 수명, 인식 방식이 다르다([설정 비교](SPACE_V4_DEPLOYMENT_READINESS.md)). 두 모델 세대의 동일 조건 비교가 아니며, 이 측정은 정확도 시험도 아니다.

trace에서 파이프라인 평균·p95, 왕복 p50·p95, 결과 수신·완료 FPS, 측정 구간 송신 간격을 다시 계산해 스크립트 출력과 일치함을 확인했다. 두 trace 모두 프레임 ID 1–60, 요청한 변형 이름, 같은 입력 해시를 기록했다.

- head-excluded INT8: [설정](../runtime/hf_space_v4/20260925_head_excluded_qdq_final/config.json) · [요약](../runtime/hf_space_v4/20260925_head_excluded_qdq_final/metrics.json) · [프레임별 trace](../runtime/hf_space_v4/20260925_head_excluded_qdq_final/trace.jsonl)
- FP32: [설정](../runtime/hf_space_v4/20260925_fp32_final/config.json) · [요약](../runtime/hf_space_v4/20260925_fp32_final/metrics.json) · [프레임별 trace](../runtime/hf_space_v4/20260925_fp32_final/trace.jsonl)
- [배포 대상, 모델 해시, 측정 조건과 한계](../runtime/hf_space_v4/MEASUREMENT_NOTES.md) · [v4 경로 준비와 v3/v4 설정 차이](SPACE_V4_DEPLOYMENT_READINESS.md)
