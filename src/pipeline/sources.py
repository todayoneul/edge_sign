"""범용 프레임 소스 추상화 — 입력 종류와 무관하게 BGR 프레임을 산출.

ImageSource    : 정지 이미지 (동일 프레임 반복)
VideoFileSource: 모든 코덱 동영상 (cv2/ffmpeg)
UrlStreamSource: 직접 URL·RTSP (+ YouTube는 yt-dlp 옵션)
웹캠은 클라이언트 캡처이므로 여기 없음.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np


class FrameSource(ABC):
    is_seekable: bool = False
    fps: float = 0.0
    frame_count: int = 0

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """다음 BGR 프레임 또는 None(끝/실패)."""

    def seek(self, frame_idx: int) -> None:
        """seekable 소스만 의미 있음 (기본 no-op)."""

    def release(self) -> None:
        pass


class ImageSource(FrameSource):
    """정지 이미지: read()가 항상 같은 프레임을 반환."""

    def __init__(self, path: str):
        # 한글 경로 대비: np.fromfile + imdecode
        data = np.fromfile(path, dtype=np.uint8)
        self._frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if self._frame is None:
            raise ValueError(f"이미지 디코딩 실패: {path}")
        self.is_seekable = False
        self.fps = 1.0
        self.frame_count = 1

    def read(self) -> Optional[np.ndarray]:
        return self._frame.copy()


class VideoFileSource(FrameSource):
    """동영상 파일 — OpenCV(ffmpeg 백엔드)로 디코딩, 모든 코덱."""

    def __init__(self, path: str):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise ValueError(f"동영상 열기 실패: {path}")
        self.is_seekable = True
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        return frame if ok else None

    def seek(self, frame_idx: int) -> None:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))

    def release(self) -> None:
        self._cap.release()


def _resolve_stream_url(url: str) -> str:
    """YouTube 등 페이지 URL → 직접 스트림 URL (yt-dlp). 미설치 시 안내."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError as e:
        raise RuntimeError("yt-dlp 미설치 — pip install yt-dlp") from e
    with YoutubeDL({"quiet": True, "format": "best[ext=mp4]/best"}) as ydl:
        info = ydl.extract_info(url, download=False)
        return info["url"]


_PAGE_HOSTS = ("youtube.com", "youtu.be")


class UrlStreamSource(FrameSource):
    """직접 영상 URL·RTSP. YouTube 등 페이지 URL은 yt-dlp로 해석."""

    def __init__(self, url: str):
        open_url = url
        if url.startswith(("http://", "https://")) and any(h in url for h in _PAGE_HOSTS):
            open_url = _resolve_stream_url(url)
        self._cap = cv2.VideoCapture(open_url)
        if not self._cap.isOpened():
            raise ValueError(f"스트림 열기 실패: {url}")
        self.is_seekable = False           # 라이브/스트림은 seek 불가로 단순화
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = 0

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()
