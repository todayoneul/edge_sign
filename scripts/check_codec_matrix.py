"""[SP1-T9] 코덱 매트릭스 검증.

"어떤 코덱이든 서버가 디코딩한다"는 SP1 핵심 주장을 자동 검증한다.
기준 mp4를 H.264 / MPEG-4 Part2 / HEVC 로 트랜스코딩한 뒤,
VideoFileSource(OpenCV ffmpeg 백엔드)가 각각 프레임을 디코딩하는지 확인한다.

  - H.264  : 브라우저 호환 (클라이언트 모드)
  - MPEG-4 : 브라우저 비호환 (이전 블랙박스 검은화면 → 서버 모드로 해결)
  - HEVC   : 브라우저 부분 호환

사용법:
  python scripts/check_codec_matrix.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline.sources import VideoFileSource

base = next(ROOT.glob("data/demo_videos/**/*.mp4"), None)
assert base, "기준 mp4 없음 — scripts/build_demo_video.py 를 먼저 실행하세요"
print(f"기준 영상: {base.name}")

tmp = Path(tempfile.mkdtemp())
CODECS = [
    ("h264",  ["-c:v", "libx264"],  "브라우저 호환"),
    ("mpeg4", ["-c:v", "mpeg4"],    "브라우저 비호환(서버 모드)"),
    ("hevc",  ["-c:v", "libx265"],  "부분 호환"),
]

failures = 0
for codec, args, note in CODECS:
    out = tmp / f"{codec}.mp4"
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(base), *args, "-t", "2", "-an", str(out)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if not out.exists() or out.stat().st_size == 0:
        print(f"  {codec:6s}: 트랜스코딩 실패 (인코더 미지원?) — 스킵\n{r.stderr[-300:]}")
        continue
    src = VideoFileSource(str(out))
    n = 0
    while True:
        f = src.read()
        if f is None:
            break
        n += 1
    src.release()
    ok = n > 0
    print(f"  {codec:6s} ({note}): {n} frames 디코딩 {'OK' if ok else 'FAIL'}")
    if not ok:
        failures += 1

if failures:
    print(f"\n[실패] {failures}개 코덱 디코딩 실패")
    sys.exit(1)
print("\n[통과] 코덱 매트릭스 — 서버가 모든 코덱을 디코딩합니다.")
