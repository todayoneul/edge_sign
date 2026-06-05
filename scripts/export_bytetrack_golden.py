"""ByteTrack 골든 출력 생성 — TS 포팅(web_modern/src/lib/byteTrack.ts) 검증용.

서버 src/track/bytetrack.py 의 ByteTracker 를 결정적 합성 시퀀스에 돌려
프레임별 (입력 검출, 출력 트랙) 을 JSON으로 덤프한다. vitest(byteTrack.test.ts)가
같은 검출을 TS 추적기에 먹여 트랙 ID/박스가 일치하는지 대조한다.

시퀀스는 **공간적으로 명확히 분리**되게 설계해 Hungarian(scipy)과 greedy 매칭이
동일 결과를 내도록 한다 → 매칭 알고리즘 차이와 무관하게 Kalman·생명주기·연관을 검증.

시나리오 (모션은 박스 대비 충분히 느려 IoU≥0.8 유지 → 안정 추적):
  - 객체 A: 좌상단, 매 프레임 +2px(x) 이동, conf 0.9
  - 객체 B: 우하단, 매 프레임 +2px(y) 이동, conf 0.9 (단 프레임5는 저신뢰 0.3 → BYTE 2차)
  - 프레임 6,7: A 검출 누락 → Lost
  - 프레임 8: A 복귀(예측 근처) → re-activate, 같은 ID 유지

사용법:
  python scripts/export_bytetrack_golden.py
  → web_modern/src/lib/byteTrack.golden.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.track.bytetrack import ByteTracker  # noqa: E402

OUT = ROOT / "web_modern" / "src" / "lib" / "byteTrack.golden.json"


def build_sequence() -> list[list[list[float]]]:
    """프레임별 검출 리스트 [[x1,y1,x2,y2,conf,cls], ...] 생성 (결정적)."""
    frames: list[list[list[float]]] = []
    for f in range(10):
        dets: list[list[float]] = []
        # 객체 A (cls 0) — 프레임 6,7 누락
        if f not in (5, 6):  # 0-indexed → 프레임 6,7
            ax = 100 + 2 * f
            dets.append([ax, 100.0, ax + 40, 160.0, 0.9, 0.0])
        # 객체 B (cls 1) — 프레임 5(0-idx 4)는 저신뢰
        by = 300 + 2 * f
        b_conf = 0.3 if f == 4 else 0.9
        dets.append([400.0, by, 450.0, by + 80, b_conf, 1.0])
        frames.append(dets)
    return frames


def main() -> None:
    frames = build_sequence()
    tracker = ByteTracker(
        track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=30, low_thresh=0.1
    )

    golden = []
    for dets in frames:
        arr = np.array(dets, dtype=np.float32) if dets else np.empty((0, 6), dtype=np.float32)
        tracks = tracker.update(arr)
        golden.append(
            {
                "dets": [[round(float(v), 4) for v in d] for d in dets],
                "tracks": sorted(
                    (
                        {
                            "id": int(t.track_id),
                            "cls": int(t.cls),
                            "tlbr": [round(float(v), 3) for v in t.tlbr],
                        }
                        for t in tracks
                    ),
                    key=lambda x: x["id"],
                ),
            }
        )

    OUT.write_text(json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8")
    n_tracks = sum(len(g["tracks"]) for g in golden)
    print(f"골든 저장: {OUT.relative_to(ROOT)} ({len(golden)} 프레임, 누적 트랙출력 {n_tracks})")
    for i, g in enumerate(golden):
        ids = [t["id"] for t in g["tracks"]]
        print(f"  frame {i + 1}: dets={len(g['dets'])} → track ids={ids}")


if __name__ == "__main__":
    main()
