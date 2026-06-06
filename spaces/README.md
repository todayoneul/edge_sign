---
title: Edge-Sign
emoji: 🚦
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: 온디바이스 주행 인지 — 검출·추적·인식 + 양자화 A/B 실시간 데모
---

# Edge·Sign — 온디바이스 주행 인지 분석 (HF Space)

엣지 디바이스용 **초경량 양자화** 검출 + 추적 + 인식 파이프라인의 실시간 웹 데모.
브라우저에서 영상을 넣고, **서버 ⇄ 온디바이스(WebGPU)** 추론과 **FP32 ⇄ INT8 양자화**를
토글하며 모델 크기·FPS·단계별 지연을 눈으로 비교할 수 있습니다.

## 이 Space의 구성
- **검출**: YOLOv8s (신호등 분리 v3) — FP32 43MB ⇄ INT8 static 18MB A/B 토글
- **추적**: ByteTrack (Kalman + IoU)
- **인식**: 한국 표지판/신호등 분류기(14클래스) · 한글 OCR
- **Q&A**: Groq LLM — **BYOK**(방문자 본인 키 입력) 방식

## 사용법
1. 히어로의 **영상 분석 시작**(파일 업로드) 또는 **웹캠으로 보기**로 바로 체험.
2. 컨트롤 바에서 **서버 ⇄ 온디바이스** · **FP32 ⇄ INT8** 토글 → 크기·FPS·단계 지연 변화 확인.
3. 우측 레일의 **장면 질의(Scene Q&A)** 에서 질문 — 검출 결과를 근거로 답합니다. Groq 키는
   레일 하단 "Groq API 키" 토글에서 입력(선택, [키 발급](https://console.groq.com/keys)).
   키는 브라우저에만 저장되고 질문 시에만 전달됩니다.

## 배포 메모 (이 폴더를 Space 루트로)
- 이 `spaces/README.md`의 내용을 **Space 리포 루트의 `README.md`**로 사용해야 HF가 YAML 헤더를 인식합니다.
- 본 메인 리포의 `Dockerfile`·`requirements-hf.txt`·`src/`·`web_modern/`(node_modules·dist 제외)·
  필요 `model_space/*.onnx`·`data/roi_cls/classes.json`·`data/idx_to_char.json`을 Space 리포에 포함
  (대용량 onnx는 **git LFS**). Dockerfile 멀티스테이지가 node 빌더로 `web_modern/dist`를 생성하므로
  dist를 직접 커밋할 필요는 없음(OCR 데모 자산은 `web_modern/public/ocr/`에 동봉).
- CPU 전용: `EDGE_SIGN_CPU_ONLY=1`(Dockerfile에 설정). GPU 불필요.
- 서버 키가 필요 없도록 Q&A는 BYOK 기본. (원한다면 Space Secret `GROQ_API_KEY`로 서버 폴백 가능.)

빌드 검증(로컬):
```bash
docker build -t edge-sign .
docker run --rm -p 7860:7860 edge-sign
# http://localhost:7860/detection/
```
