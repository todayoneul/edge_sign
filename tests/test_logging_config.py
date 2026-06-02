"""configure_logging() — 멱등성·레벨·stdlib 인터셉트 검증."""

import io
import logging

from loguru import logger

import src.pipeline.logging_config as lc


def _reset() -> None:
    # 테스트 간 격리: 1회-구성 플래그 + loguru 싱크 모두 초기화
    lc._CONFIGURED = False
    logger.remove()


def test_configure_logging_is_idempotent():
    _reset()
    lc.configure_logging()
    lc.configure_logging()  # 두 번째 호출은 no-op, 예외 없어야 함
    assert lc._CONFIGURED is True


def test_logger_emits_at_info_level():
    _reset()
    lc.configure_logging(level="INFO")
    sink = io.StringIO()
    sink_id = logger.add(sink, level="INFO", format="{message}")
    try:
        logger.info("hello-edge-sign")
    finally:
        logger.remove(sink_id)
    assert "hello-edge-sign" in sink.getvalue()


def test_stdlib_logging_is_intercepted():
    _reset()
    lc.configure_logging(level="DEBUG")
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG", format="{message}")
    std = logging.getLogger("uvicorn.error")
    std.setLevel(logging.DEBUG)
    try:
        std.info("via-stdlib")
    finally:
        logger.remove(sink_id)
    assert "via-stdlib" in sink.getvalue()


def test_env_var_level_is_respected(monkeypatch, capsys):
    # EDGE_SIGN_LOG_LEVEL=WARNING → INFO는 stderr 싱크에서 필터링, WARNING만 출력
    _reset()
    monkeypatch.setenv("EDGE_SIGN_LOG_LEVEL", "WARNING")
    lc.configure_logging()  # 명시 인자 없음 → env 레벨 사용
    logger.info("below-threshold")
    logger.warning("above-threshold")
    err = capsys.readouterr().err
    assert "above-threshold" in err
    assert "below-threshold" not in err
