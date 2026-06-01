"""파이프라인 검출기 variant 전환 + 단계별 레이턴시 계측 테스트.

A/B 양자화 토글(FP32⇄INT8)과 파이프라인 단계 시각화의 백엔드 계약을 검증한다.
실제 ONNX 모델을 로드하므로 onnxruntime + model_space 모델이 있을 때만 실행.
"""
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "model_space"

# FP32 vs INT8 static(QDQ) — 둘 다 CPU EP에서 실행 가능(동적 INT8은 ConvInteger 미지원).
# 전환 메커니즘만 검증하므로 검출 정확도/택소노미는 무관.
VAR_A = MODEL_DIR / "yolov8s_signs_fp32.onnx"
VAR_B = MODEL_DIR / "yolov8s_signs_int8_static.onnx"

pytest.importorskip("onnxruntime")
if not (VAR_A.exists() and VAR_B.exists()):
    pytest.skip("검출기 ONNX variant 파일 없음", allow_module_level=True)

# CUDA EP 시도/경고를 피하고 CPU로만 로드 (CI/테스트 환경 안정성)
os.environ["EDGE_SIGN_CPU_ONLY"] = "1"

from src.pipeline.e2e_pipeline import EdgeSignPipeline


@pytest.fixture(scope="module")
def pipe():
    return EdgeSignPipeline(
        yolo_variants={"a": str(VAR_A), "b": str(VAR_B)},
        det_taxonomy="v2",
        conf_thres=0.15,
    )


def test_loads_all_variants_with_sizes(pipe):
    info = pipe.variant_info()                 # [{"name","mb"}, ...]
    names = {v["name"] for v in info}
    assert names == {"a", "b"}
    assert all(v["mb"] > 0 for v in info)      # 파일 크기(MB) 기록


def test_default_active_is_first_variant(pipe):
    assert pipe.active_variant == "a"          # dict 첫 키가 기본


def test_set_variant_switches_active(pipe):
    pipe.set_variant("b")
    assert pipe.active_variant == "b"
    pipe.set_variant("a")                       # 원복 (모듈 fixture 공유)
    assert pipe.active_variant == "a"


def test_set_unknown_variant_raises(pipe):
    with pytest.raises(KeyError):
        pipe.set_variant("nope")


def test_process_frame_reports_variant_and_stage_ms(pipe):
    frame = np.zeros((480, 640, 3), np.uint8)
    out = pipe.process_frame(frame, variant="b")
    assert out["variant"] == "b"               # 요청한 variant 반영
    assert out["model_mb"] > 0
    stage = out["stage_ms"]
    assert set(stage) == {"detect", "track", "recognize"}
    assert all(v >= 0 for v in stage.values())
    assert "inference_ms" in out               # 기존 총합 키 유지
    # 단계 합이 총합에 근접 (오차 허용 — 후처리/직렬화 오버헤드)
    assert sum(stage.values()) <= out["inference_ms"] + 5


def test_process_frame_persists_active_variant(pipe):
    """variant 미지정 호출은 active_variant를 사용."""
    pipe.set_variant("b")
    out = pipe.process_frame(np.zeros((120, 160, 3), np.uint8))
    assert out["variant"] == "b"
    pipe.set_variant("a")
