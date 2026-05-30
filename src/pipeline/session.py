"""단일 서버 스트림 세션 — FrameSource + 재생 제어 상태."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from src.pipeline.sources import (
    FrameSource, ImageSource, VideoFileSource, UrlStreamSource,
)

MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500MB


class Session:
    def __init__(self, source: FrameSource, temp_path: Optional[Path] = None):
        self.source = source
        self.temp_path = temp_path
        self.playing = True
        self.speed = 1.0

    def control(self, action: str, value=None):
        if action == "play":
            self.playing = True
        elif action == "pause":
            self.playing = False
        elif action == "speed" and value:
            self.speed = float(value)
        elif action == "seek" and value is not None and self.source.is_seekable:
            self.source.seek(int(value))

    def close(self):
        try:
            self.source.release()
        finally:
            if self.temp_path and self.temp_path.exists():
                self.temp_path.unlink(missing_ok=True)


class SessionManager:
    """동시 세션 1개. 새 세션 생성 시 이전 세션 정리."""

    def __init__(self):
        self._current: Optional[Session] = None

    def _replace(self, sess: Session) -> str:
        if self._current is not None:
            self._current.close()
        self._current = sess
        return "session"

    def get(self) -> Optional[Session]:
        return self._current

    def from_image(self, path: Path) -> str:
        return self._replace(Session(ImageSource(str(path)), temp_path=path))

    def from_video(self, path: Path) -> str:
        return self._replace(Session(VideoFileSource(str(path)), temp_path=path))

    def from_url(self, url: str) -> str:
        return self._replace(Session(UrlStreamSource(url)))

    def close(self):
        if self._current is not None:
            self._current.close()
            self._current = None


def save_upload(data: bytes, suffix: str) -> Path:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"업로드 크기 초과 (> {MAX_UPLOAD_BYTES // (1024*1024)}MB)")
    fd = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    fd.write(data)
    fd.close()
    return Path(fd.name)
