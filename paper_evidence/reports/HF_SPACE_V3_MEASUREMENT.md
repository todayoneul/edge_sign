# 공개 Hugging Face Space 재측정: 한눈에 보기

측정일: 2026-09-24. 공개 [Edge-Sign Space](https://huggingface.co/spaces/gyann/edge-sign)를 재시작해 실행 상태와 `/api/status` HTTP 200을 확인했다. 현재 Space는 논문 재평가의 **YOLO26-n v4가 아니라 YOLOv8s v3**를 배포한다. 하드웨어는 Hugging Face `cpu-basic`, Space 소스 commit은 `9fc2593358a678a5b1597e978a63778bc909fc31`이다.

공개 샘플 15프레임을 반복하며 실제 `/ws/stream`에 초당 10프레임을 보냈다. 각 구성의 추적기를 초기화하고 10프레임 준비 실행 후 50프레임을 측정했다. 브라우저 화면 그리기는 포함하지 않았다.

| 현재 Space v3 | 결과 수신 속도 | Space 검출→추적→인식 평균 | 실제 선택 모델 파일 합계 |
|---|---:|---:|---:|
| head-excluded INT8 | **2.500 FPS** | **349.238 ms/프레임** | **20,994,769 B** |
| FP32 | **1.991 FPS** | **456.192 ms/프레임** | **47,744,540 B** |

**해석:** 현재 공개 Space의 **서버 추론 경로**는 15 MB와 30 FPS 목표를 모두 충족하지 못했다. 10 FPS로 계속 보내면 처리보다 입력이 빨라 대기열이 늘어나므로, 위 수신 속도를 브라우저 전체 FPS라고 부르지 않는다. 샘플을 반복한 속도 시험이며 새 정확도 시험도 아니다. YOLO26 v4 조합(로컬 모델 합계 3,533,232 B)의 실제 웹 배포 성능은 아직 검증되지 않았다.

독립 검토 에이전트가 양쪽 60프레임 trace의 계산값, 프레임 ID 1–60, 동일 입력, 약 100 ms 전송 간격을 재검산했다. 공개 웹 화면의 서버 영상 경로에는 첫 재생 시 프레임 전송 콜백과 FPS 표시 문제도 발견돼, 화면 렌더까지 포함한 수치는 별도로 측정해야 한다.

- [상세 TIIS evidence 보고서](TIIS_EVIDENCE_REPORT.md)
- INT8: [설정](../runtime/hf_space_v3/20260924_int8_final/config.json) · [요약](../runtime/hf_space_v3/20260924_int8_final/metrics.json) · [프레임별 trace](../runtime/hf_space_v3/20260924_int8_final/trace.jsonl)
- FP32: [설정](../runtime/hf_space_v3/20260924_fp32_final/config.json) · [요약](../runtime/hf_space_v3/20260924_fp32_final/metrics.json) · [프레임별 trace](../runtime/hf_space_v3/20260924_fp32_final/trace.jsonl)
- [환경과 한계](../runtime/hf_space_v3/ENVIRONMENT.txt) · [예비 실행 제외 이유](../runtime/hf_space_v3/PILOT_NOTES.md)
