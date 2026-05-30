import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_image(tmp_path):
    """64x48 단색 JPG (ASCII 경로)."""
    p = tmp_path / "img.jpg"
    img = np.full((48, 64, 3), (0, 128, 255), np.uint8)
    cv2.imwrite(str(p), img)
    return p


@pytest.fixture
def sample_mp4(tmp_path):
    """10프레임 mp4 (mp4v 코덱, ASCII 경로)."""
    p = tmp_path / "clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(p), fourcc, 10.0, (64, 48))
    for i in range(10):
        frame = np.full((48, 64, 3), i * 20 % 255, np.uint8)
        w.write(frame)
    w.release()
    return p
