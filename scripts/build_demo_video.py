"""
[Phase 7] test JPG 시퀀스 → 브라우저 호환 H.264 mp4 합성

학습에 사용하지 않은 `data/aihub_traffic/test/` 시퀀스를 동영상으로 합성하여
검증(validation) 및 웹 시연용으로 사용한다. 학습 데이터와 동일 분포라 검출기가
안정적으로 동작한다 (블랙박스 OOD 영상과 대조).

특징:
  - 파일명 숫자 prefix 기준 시간순 정렬 (타임스탬프)
  - ffmpeg concat demuxer로 H.264 yuv420p + faststart (브라우저 즉시 재생)
  - 학습에 안 쓰인 test split만 사용 → 정직한 시연

사용법:
  # 기본: 주간 시퀀스, 15fps
  python scripts/build_demo_video.py

  # 시퀀스/fps 지정
  python scripts/build_demo_video.py --seq d_validation_1920_1080_daylight_2 --fps 15
  python scripts/build_demo_video.py --list           # 사용 가능한 시퀀스 나열
  python scripts/build_demo_video.py --all             # 모든 test 시퀀스 합성
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
TEST_IMG_DIR = ROOT / "data" / "aihub_traffic" / "test" / "images"
OUT_DIR = ROOT / "data" / "demo_videos"


def list_sequences():
    if not TEST_IMG_DIR.exists():
        print(f"[오류] test 이미지 디렉토리 없음: {TEST_IMG_DIR}")
        return []
    seqs = sorted([d for d in TEST_IMG_DIR.iterdir() if d.is_dir()])
    for d in seqs:
        n = len(list(d.glob("*.jpg")))
        print(f"  {d.name}  ({n} frames)")
    return seqs


def sorted_frames(seq_dir: Path):
    """숫자 prefix는 정수 정렬, 그 외(a... 등)는 문자열 정렬로 뒤에 배치."""
    jpgs = list(seq_dir.glob("*.jpg"))
    numeric = sorted([p for p in jpgs if p.stem.isdigit()], key=lambda p: int(p.stem))
    other   = sorted([p for p in jpgs if not p.stem.isdigit()], key=lambda p: p.stem)
    return numeric + other


def split_into_runs(frames, gap=10):
    """타임스탬프(파일명 정수) 간격이 gap 이하인 연속 구간으로 분할.

    이 test 시퀀스는 여러 위치의 짧은 주행 스냅샷이 이어붙은 몽타주라,
    간격이 튀는 지점이 곧 '다른 위치'다. 같은 위치 = 하나의 연속 구간.
    """
    numeric = [p for p in frames if p.stem.isdigit()]
    if not numeric:
        return [frames]  # 타임스탬프 없으면 통째로 1개
    ts = [int(p.stem) for p in numeric]
    runs, start = [], 0
    for i in range(1, len(ts)):
        if ts[i] - ts[i - 1] > gap:
            runs.append(numeric[start:i])
            start = i
    runs.append(numeric[start:])
    return runs


def _encode(frames, out_path: Path, fps: int, scale_w: int, crf: int, tag: str) -> Path:
    """프레임 리스트 → H.264 mp4 (ffmpeg concat demuxer)."""
    if not frames:
        return None
    import cv2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dur = 1.0 / fps

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        list_path = Path(f.name)
        for p in frames:
            safe = str(p.resolve()).replace("'", r"'\''")
            f.write(f"file '{safe}'\n")
            f.write(f"duration {dur:.5f}\n")
        safe = str(frames[-1].resolve()).replace("'", r"'\''")
        f.write(f"file '{safe}'\n")  # 마지막 프레임 길이 보존

    # 너비 scale_w로 다운스케일(높이 비율 유지, 짝수). scale=W:-2 거부 빌드 대비 직접 계산.
    probe = cv2.imread(str(frames[0]))
    src_h, src_w = probe.shape[:2]
    out_h = int(round(src_h * scale_w / src_w / 2)) * 2
    vf = f"scale={scale_w}:{out_h}"
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-vsync", "vfr", "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    list_path.unlink(missing_ok=True)
    if res.returncode != 0:
        print(f"[오류] ffmpeg 실패 ({tag}):\n{res.stderr[-1200:]}")
        return None
    size_mb = out_path.stat().st_size / 1e6
    print(f"[완료] {out_path.name}  ({len(frames)} frames, {size_mb:.1f} MB, ~{len(frames)/fps:.0f}s)  [{tag}]")
    return out_path


def build(seq_dir: Path, fps: int, scale_w: int = 1280, crf: int = 24) -> Path:
    """시퀀스 전체를 단일 mp4로 합성 (몽타주 — 위치 점프 많음)."""
    frames = sorted_frames(seq_dir)
    if not frames:
        print(f"[건너뜀] 프레임 없음: {seq_dir.name}")
        return None
    out_path = OUT_DIR / f"{seq_dir.name}.mp4"
    print(f"[합성] {seq_dir.name}: {len(frames)} frames @ {fps}fps")
    return _encode(frames, out_path, fps, scale_w, crf, "full")


def build_clips(seq_dir: Path, fps: int, scale_w: int, crf: int,
                gap: int, min_len: int, top_n: int) -> int:
    """시퀀스를 '같은 위치' 연속 구간별 개별 mp4로 분할 저장.

    출력: data/demo_videos/<seq>_clips/clip_01.mp4 ...
    각 클립은 한 위치의 연속 프레임이라 박스가 안정적으로 추적된다.
    낮은 fps로 렌더 → 일시정지하고 Q&A 하기 좋음.
    """
    frames = sorted_frames(seq_dir)
    runs = split_into_runs(frames, gap=gap)
    runs = [r for r in runs if len(r) >= min_len]
    runs.sort(key=len, reverse=True)
    runs = runs[:top_n]
    if not runs:
        print(f"[건너뜀] {seq_dir.name}: 길이>={min_len} 연속 구간 없음 (gap={gap})")
        return 0

    clip_dir = OUT_DIR / f"{seq_dir.name}_clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    print(f"[클립] {seq_dir.name}: {len(runs)}개 위치 클립 (fps={fps}, min_len={min_len}) → {clip_dir.name}/")
    n = 0
    for i, run in enumerate(runs, 1):
        out = clip_dir / f"clip_{i:02d}_{run[0].stem}.mp4"
        if _encode(run, out, fps, scale_w, crf, f"clip {i}/{len(runs)}"):
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="d_validation_1920_1080_daylight_2",
                    help="test 시퀀스 디렉토리명")
    ap.add_argument("--fps", type=int, default=6,
                    help="출력 fps (클립 모드는 낮게 — 일시정지하며 Q&A 하기 좋게)")
    ap.add_argument("--scale", type=int, default=1280, help="출력 너비(px), 높이는 비율 유지")
    ap.add_argument("--crf", type=int, default=24, help="H.264 품질(낮을수록 고화질/대용량)")
    ap.add_argument("--list", action="store_true", help="사용 가능한 시퀀스 나열")
    ap.add_argument("--all", action="store_true", help="모든 test 시퀀스 합성")
    # 클립 분할 모드 (기본값) — 같은 위치 연속 구간을 개별 영상으로
    ap.add_argument("--full", action="store_true",
                    help="분할하지 않고 시퀀스 전체를 단일 영상으로 (몽타주, 위치 점프 많음)")
    ap.add_argument("--gap", type=int, default=10,
                    help="타임스탬프 간격이 이보다 크면 다른 위치로 분할")
    ap.add_argument("--min_len", type=int, default=8, help="클립 최소 프레임 수")
    ap.add_argument("--top_n", type=int, default=12, help="가장 긴 클립 N개만 추출")
    args = ap.parse_args()

    if args.list:
        print("사용 가능한 test 시퀀스:")
        list_sequences()
        return

    if not TEST_IMG_DIR.exists():
        print(f"[오류] test 이미지 디렉토리 없음: {TEST_IMG_DIR}")
        print("먼저 실행: python scripts/prepare_korean_traffic.py (또는 extract_frames.py)")
        sys.exit(1)

    targets = (sorted([d for d in TEST_IMG_DIR.iterdir() if d.is_dir()])
               if args.all else [TEST_IMG_DIR / args.seq])

    for seq_dir in targets:
        if not seq_dir.exists():
            print(f"[오류] 시퀀스 없음: {seq_dir}")
            print("사용 가능한 시퀀스:")
            list_sequences()
            sys.exit(1)
        if args.full:
            build(seq_dir, args.fps if args.fps else 15, args.scale, args.crf)
        else:
            build_clips(seq_dir, args.fps, args.scale, args.crf,
                        args.gap, args.min_len, args.top_n)


if __name__ == "__main__":
    main()
