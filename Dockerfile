# Edge-Sign — HF Spaces (Docker, CPU 전용) 실시간 양자화 데모
# 검출(YOLOv8s INT8) + 추적(ByteTrack) + 인식 + Q&A(BYOK). GPU·학습 의존성 없음.

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: 프론트 빌드 — web_modern(React/Vite) → dist/ (OCR 데모 public/ocr 포함)
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-slim AS web-builder
WORKDIR /build
COPY web_modern/package.json web_modern/package-lock.json ./
RUN npm ci
COPY web_modern/ ./
RUN npm run build      # → /build/dist (public/ocr → dist/ocr 로 함께 번들)

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: 런타임 — Python CPU 추론 서버
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# OpenCV(headless도 libGL/glib 필요) + 영상 디코딩(ffmpeg) 런타임
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces 권장: 비root 사용자(uid 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    EDGE_SIGN_CPU_ONLY=1 \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements-hf.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-hf.txt

# 앱 소스
COPY --chown=user src/ ./src/
# 프론트 빌드 산출물 (web_modern/dist) — app.py가 /detection/ 로 서빙, OCR 데모는 /detection/ocr/
COPY --from=web-builder --chown=user /build/dist ./web_modern/dist
# 인식 자원 (분류기 클래스 매핑 + OCR 인덱스)
COPY --chown=user data/roi_cls/classes.json ./data/roi_cls/classes.json
COPY --chown=user data/idx_to_char.json ./data/idx_to_char.json
# 추론 모델 (A/B용 fp32 + int8_static, OCR w8a8, 분류기 fp32) — .dockerignore에서 선별
COPY --chown=user model_space/ ./model_space/

USER user
EXPOSE 7860
CMD ["uvicorn", "src.pipeline.app:app", "--host", "0.0.0.0", "--port", "7860"]
