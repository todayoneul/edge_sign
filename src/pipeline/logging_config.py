"""중앙 로깅 설정 — loguru 싱크 + 표준 logging 인터셉트.

서버 startup에서 configure_logging()을 1회 호출한다.
라이브러리 모듈은 `from loguru import logger`만 사용(설정은 여기서 일원화).
"""

from __future__ import annotations

import logging
import os
import sys
from types import FrameType

from loguru import logger

_CONFIGURED = False


class InterceptHandler(logging.Handler):
    """표준 logging 레코드를 loguru로 전달 (uvicorn/fastapi 로그 통합)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame: FrameType | None = logging.currentframe()
        depth = 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging(level: str | None = None) -> None:
    """loguru 싱크를 stderr로 설정하고 표준 logging을 인터셉트한다.

    멱등(idempotent) — 여러 번 호출해도 싱크는 1회만 구성된다.

    Args:
        level: 로그 레벨. None이면 EDGE_SIGN_LOG_LEVEL 환경변수(기본 "INFO").
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    log_level = (level or os.environ.get("EDGE_SIGN_LOG_LEVEL", "INFO")).upper()

    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
        backtrace=False,
        diagnose=False,
    )
    intercept = InterceptHandler()
    logging.basicConfig(handlers=[intercept], level=0, force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        lg = logging.getLogger(name)
        lg.handlers = [intercept]
        lg.propagate = False

    _CONFIGURED = True
