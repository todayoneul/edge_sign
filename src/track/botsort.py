"""
BoT-SORT 구현 — ByteTrack + Camera Motion Compensation + ReID 외형 매칭

Aharon et al., "BoT-SORT: Robust Associations Multi-Pedestrian Tracking", 2022.
https://arxiv.org/abs/2206.14651

ByteTrack 대비 추가 구성 요소:
  1. Camera Motion Compensation (CMC): ORB 희소 광류 → 호모그래피 → Kalman 상태 보정
  2. ReID 외형 특징: track 별 임베딩 버퍼 → IoU + 코사인 유사도 결합 매칭

사용법:
  from src.track.botsort import BoTSORTTracker, SimpleReIDNet

  reid_net = SimpleReIDNet(embed_dim=128)
  tracker  = BoTSORTTracker(reid_net=reid_net, track_thresh=0.5, match_thresh=0.8,
                             frame_rate=5, alpha=0.95)

  for frame_bgr in frames:
      dets   = detector.detect(frame_bgr)   # [N, 6]: x1 y1 x2 y2 conf cls
      tracks = tracker.update(dets, frame_bgr)
"""

from __future__ import annotations

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import List, Optional

# ByteTrack 기반 구성 요소 재사용
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.track.bytetrack import (
    KalmanFilter, STrack, TrackState,
    iou_batch, linear_assignment
)


def iou_distance(tlwhs_a: np.ndarray, tlwhs_b: np.ndarray) -> np.ndarray:
    """tlwh 형식 배열 → IoU 거리 행렬."""
    def _tlwh_to_tlbr(tlwhs):
        tlbr = tlwhs.copy()
        tlbr[:, 2] = tlwhs[:, 0] + tlwhs[:, 2]
        tlbr[:, 3] = tlwhs[:, 1] + tlwhs[:, 3]
        return tlbr
    if len(tlwhs_a) == 0 or len(tlwhs_b) == 0:
        return np.zeros((len(tlwhs_a), len(tlwhs_b)), dtype=np.float32)
    tlbr_a = _tlwh_to_tlbr(np.array(tlwhs_a, dtype=np.float32))
    tlbr_b = _tlwh_to_tlbr(np.array(tlwhs_b, dtype=np.float32))
    return 1.0 - iou_batch(tlbr_a, tlbr_b)


# ─────────────────────────────────────────────────────────────────
# 1. ReID 백본: SimpleReIDNet
# ─────────────────────────────────────────────────────────────────

class SimpleReIDNet(nn.Module):
    """
    경량 외형 특징 추출기.

    Input : (B, 3, 64, 64) RGB 크롭 (정규화됨)
    Output: (B, embed_dim) L2-정규화된 임베딩 벡터

    파라미터: ~128K (embed_dim=128 기준)
    OSNet-x0.25(~500K)보다 가볍지만 E6 ablation용으로 충분한 구조.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

        # Block 1: 3x64x64 -> 32x32x32
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)

        # Block 2: 32x32x32 -> 64x16x16
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)

        # Block 3 (depthwise-separable): 64x16x16 -> 128x8x8
        self.dw3   = nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False)
        self.pw3   = nn.Conv2d(64, 128, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(128)

        # Block 4 (depthwise-separable): 128x8x8 -> 128x4x4
        self.dw4   = nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False)
        self.pw4   = nn.Conv2d(128, 128, 1, bias=False)
        self.bn4   = nn.BatchNorm2d(128)

        self.pool  = nn.AdaptiveAvgPool2d((1, 1))
        self.fc    = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)                    # 32x32
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)                    # 16x16
        x = F.relu(self.bn3(self.pw3(self.dw3(x))))
        x = F.max_pool2d(x, 2)                    # 8x8
        x = F.relu(self.bn4(self.pw4(self.dw4(x))))
        x = F.max_pool2d(x, 2)                    # 4x4
        x = self.pool(x).flatten(1)               # 128
        x = self.fc(x)
        return F.normalize(x, dim=1)              # L2 정규화


# ─────────────────────────────────────────────────────────────────
# 2. ReID 특징 추출 유틸
# ─────────────────────────────────────────────────────────────────

_REID_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_REID_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_crop(img_bgr: np.ndarray, bbox: np.ndarray,
                    size: int = 64) -> np.ndarray:
    """bbox [x1,y1,x2,y2] 크롭 → 정규화된 float32 NCHW numpy."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 3, size, size), dtype=np.float32)
    crop = cv2.resize(img_bgr[y1:y2, x1:x2], (size, size))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    crop = (crop - _REID_MEAN) / _REID_STD
    return crop.transpose(2, 0, 1)[np.newaxis]    # [1, 3, H, W]


def extract_features(reid_model: nn.Module | None,
                     img_bgr: np.ndarray,
                     bboxes: np.ndarray) -> Optional[np.ndarray]:
    """검출 bbox 배치 → 임베딩 [N, embed_dim]."""
    if reid_model is None or len(bboxes) == 0:
        return None
    crops = np.concatenate(
        [preprocess_crop(img_bgr, bb) for bb in bboxes], axis=0
    )  # [N, 3, 64, 64]
    with torch.no_grad():
        feats = reid_model(torch.from_numpy(crops))
    return feats.cpu().numpy()   # [N, embed_dim]


# ─────────────────────────────────────────────────────────────────
# 3. Camera Motion Compensation (CMC)
# ─────────────────────────────────────────────────────────────────

class CMC:
    """
    ORB 희소 광류 기반 카메라 모션 보정.
    연속 프레임 간 전역 호모그래피 추정 → Kalman 상태 보정에 사용.
    """

    def __init__(self, max_features: int = 500, match_ratio: float = 0.7):
        self.orb     = cv2.ORB_create(nfeatures=max_features)
        self.bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.ratio   = match_ratio
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_kp:   Optional[list] = None
        self.prev_des:  Optional[np.ndarray] = None

    def reset(self):
        self.prev_gray = None
        self.prev_kp   = None
        self.prev_des  = None

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        """현재 프레임 처리 → 3×3 호모그래피 (없으면 단위행렬)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        H = np.eye(3, dtype=np.float32)   # 기본: 변환 없음

        if (self.prev_des is not None and des is not None
                and len(kp) >= 4 and len(self.prev_kp) >= 4):
            matches = self.bf.knnMatch(self.prev_des, des, k=2)
            good = [m for m, n in matches if m.distance < self.ratio * n.distance]

            if len(good) >= 4:
                src_pts = np.float32(
                    [self.prev_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H_est, mask = cv2.findHomography(
                    src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H_est is not None:
                    H = H_est

        self.prev_gray = gray
        self.prev_kp   = kp
        self.prev_des  = des
        return H


def warp_kalman_mean(mean: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Kalman 상태 [cx, cy, ar, h, ...] 에 CMC 호모그래피 적용."""
    cx, cy = mean[0], mean[1]
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(pt, H)
    new_mean = mean.copy()
    new_mean[0] = warped[0, 0, 0]
    new_mean[1] = warped[0, 0, 1]
    return new_mean


# ─────────────────────────────────────────────────────────────────
# 4. BoT-SORT 트랙 (STrack 확장)
# ─────────────────────────────────────────────────────────────────

class BoTTrack(STrack):
    """ReID 임베딩 버퍼를 추가한 STrack 확장."""

    def __init__(self, tlwh: np.ndarray, score: float, cls: int,
                 feat: Optional[np.ndarray] = None,
                 feat_history: int = 50,
                 alpha: float = 0.95):
        super().__init__(tlwh, score, cls)
        self.alpha          = alpha
        self.smooth_feat:   Optional[np.ndarray] = None
        self.feat_buf:      deque = deque(maxlen=feat_history)
        if feat is not None:
            self.update_features(feat)

    def update_features(self, feat: np.ndarray):
        """EMA 방식으로 외형 특징 갱신."""
        feat = feat / (np.linalg.norm(feat) + 1e-6)
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
            self.smooth_feat /= (np.linalg.norm(self.smooth_feat) + 1e-6)
        self.feat_buf.append(feat)


# ─────────────────────────────────────────────────────────────────
# 5. BoTSORTTracker
# ─────────────────────────────────────────────────────────────────

def cosine_distance(feats_a: np.ndarray, feats_b: np.ndarray) -> np.ndarray:
    """코사인 거리 행렬 [N_a, N_b]. 이미 L2 정규화된 벡터 가정."""
    return 1.0 - feats_a @ feats_b.T   # [N_a, N_b]


class BoTSORTTracker:
    """
    BoT-SORT: ByteTrack + CMC + ReID 외형 매칭.

    매칭 비용: λ × IoU거리 + (1-λ) × 코사인 거리
    """

    def __init__(
        self,
        reid_net: Optional[nn.Module] = None,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int   = 30,
        frame_rate: int     = 30,
        lam: float          = 0.5,   # IoU vs ReID 가중치 (0=ReID전용, 1=IoU전용)
        alpha: float        = 0.95,  # EMA 계수
        use_cmc: bool       = True,
    ):
        self.reid_net     = reid_net.eval() if reid_net is not None else None
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.buffer_size  = int(frame_rate / 30.0 * track_buffer)
        self.lam          = lam
        self.alpha        = alpha

        self.kalman = KalmanFilter()
        self.cmc    = CMC() if use_cmc else None

        self.tracked_stracks: List[BoTTrack] = []
        self.lost_stracks:    List[BoTTrack] = []
        self.removed_stracks: List[BoTTrack] = []

        self.frame_id = 0
        self._next_id = 1

    # ──────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _activate(self, track: BoTTrack):
        track.track_id = self._new_id()
        track.state    = TrackState.Tracked
        track.is_activated = True

    def _matching_cost(self, tracks: List[BoTTrack],
                       dets_tlwh: np.ndarray,
                       det_feats: Optional[np.ndarray]) -> np.ndarray:
        """IoU 거리 + (있으면) 코사인 거리 결합."""
        if len(tracks) == 0 or len(dets_tlwh) == 0:
            return np.zeros((len(tracks), len(dets_tlwh)), dtype=np.float32)

        # IoU 거리
        track_tlwhs = np.array([t.tlwh for t in tracks])
        iou_dist = iou_distance(track_tlwhs, dets_tlwh)

        if (det_feats is not None
                and any(t.smooth_feat is not None for t in tracks)):
            # ReID 코사인 거리
            t_feats = np.array([
                t.smooth_feat if t.smooth_feat is not None
                else np.zeros(det_feats.shape[1])
                for t in tracks
            ])
            cos_dist = cosine_distance(t_feats, det_feats)
            cost = self.lam * iou_dist + (1 - self.lam) * cos_dist
        else:
            cost = iou_dist

        return cost

    # ──────────────────────────────────────────
    # update()
    # ──────────────────────────────────────────

    def _make_temp_strack(self, tlwh: np.ndarray, score: float, cls: int) -> STrack:
        """매칭용 임시 STrack (update/re_activate 인터페이스 호환)."""
        t = STrack(tlwh, score, cls)
        # Kalman 없이도 tlwh 접근 가능하도록 _tlwh 직접 설정
        t._tlwh = tlwh.copy()
        return t

    def update(self, detections: np.ndarray,
               frame_bgr: Optional[np.ndarray] = None) -> List[BoTTrack]:
        """
        Args:
            detections: [N, 6] x1 y1 x2 y2 conf cls
            frame_bgr:  현재 BGR 프레임 (CMC + ReID 특징 추출용)
        Returns:
            활성 트랙 리스트
        """
        self.frame_id += 1

        # 1. CMC 호모그래피 계산
        H = np.eye(3, dtype=np.float32)
        if self.cmc is not None and frame_bgr is not None:
            H = self.cmc.apply(frame_bgr)

        # 2. 칼만 예측 + CMC 보정
        for t in self.tracked_stracks + self.lost_stracks:
            t.predict()
            if not np.allclose(H, np.eye(3)):
                t.mean = warp_kalman_mean(t.mean, H)

        # 3. 검출 분리 (고신뢰 / 저신뢰)
        if len(detections) == 0:
            dets_high = dets_low = np.zeros((0, 6))
        else:
            high_mask = detections[:, 4] >= self.track_thresh
            dets_high = detections[high_mask]
            dets_low  = detections[~high_mask]

        # 4. ReID 특징 추출 (고신뢰 검출)
        det_feats_high: Optional[np.ndarray] = None
        if self.reid_net is not None and frame_bgr is not None and len(dets_high):
            det_feats_high = extract_features(
                self.reid_net, frame_bgr, dets_high[:, :4])

        # 5. 1차 매칭 (활성 트랙 + 고신뢰 검출)
        confirmed   = [t for t in self.tracked_stracks if t.is_activated]
        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]

        activated_now: List[BoTTrack] = []
        refound:       List[BoTTrack] = []

        if len(confirmed) and len(dets_high):
            tlwh_high = np.array([
                [d[0], d[1], d[2]-d[0], d[3]-d[1]] for d in dets_high
            ])
            cost = self._matching_cost(confirmed, tlwh_high, det_feats_high)
            matches1, unmatched_t, unmatched_d = linear_assignment(
                cost, thresh=self.match_thresh)
        else:
            matches1 = []
            unmatched_t = list(range(len(confirmed)))
            unmatched_d = list(range(len(dets_high)))

        for ti, di in matches1:
            t = confirmed[ti]
            d = dets_high[di]
            tlwh = np.array([d[0], d[1], d[2]-d[0], d[3]-d[1]])
            tmp  = self._make_temp_strack(tlwh, d[4], int(d[5]))
            t.update(tmp, self.frame_id)
            if det_feats_high is not None:
                t.update_features(det_feats_high[di])
            activated_now.append(t)

        # 6. 2차 매칭 (미매칭 트랙 + 저신뢰 검출, BYTE 전략)
        unmatched_tracks_1 = [confirmed[i] for i in unmatched_t]
        if len(unmatched_tracks_1) and len(dets_low):
            tlwh_low = np.array([
                [d[0], d[1], d[2]-d[0], d[3]-d[1]] for d in dets_low
            ])
            iou_d2 = iou_distance(
                np.array([t.tlwh for t in unmatched_tracks_1]), tlwh_low)
            matches2, remain_t, _ = linear_assignment(iou_d2, thresh=0.5)
            for ti2, di2 in matches2:
                t   = unmatched_tracks_1[ti2]
                d   = dets_low[di2]
                tlwh = np.array([d[0], d[1], d[2]-d[0], d[3]-d[1]])
                tmp  = self._make_temp_strack(tlwh, d[4], int(d[5]))
                t.update(tmp, self.frame_id)
                activated_now.append(t)
            unmatched_t_final = [unmatched_tracks_1[i] for i in remain_t]
        else:
            unmatched_t_final = unmatched_tracks_1

        # 7. Lost 처리
        for t in unmatched_t_final:
            if t.state != TrackState.Lost:
                t.mark_lost()
                self.lost_stracks.append(t)

        # 8. Lost 재발견 (3차 매칭)
        if len(self.lost_stracks) and len(unmatched_d):
            det_arr_ud = dets_high[unmatched_d] if len(unmatched_d) else np.zeros((0, 6))
            feats_ud   = (det_feats_high[unmatched_d]
                          if det_feats_high is not None and len(unmatched_d) else None)
            tlwh_ud    = (np.array([[d[0], d[1], d[2]-d[0], d[3]-d[1]] for d in det_arr_ud])
                          if len(det_arr_ud) else np.zeros((0, 4)))
            cost3 = self._matching_cost(self.lost_stracks, tlwh_ud, feats_ud)
            matches3, remain_lost, remain_ud = linear_assignment(cost3, thresh=0.4)

            for li, di3 in matches3:
                t   = self.lost_stracks[li]
                d   = det_arr_ud[di3]
                tlwh = np.array([d[0], d[1], d[2]-d[0], d[3]-d[1]])
                tmp  = self._make_temp_strack(tlwh, d[4], int(d[5]))
                t.re_activate(tmp, self.frame_id, new_id=False)
                if feats_ud is not None:
                    t.update_features(feats_ud[di3])
                refound.append(t)

            self.lost_stracks = [self.lost_stracks[i] for i in remain_lost]
            unmatched_d_new   = [unmatched_d[i] for i in remain_ud]
        else:
            unmatched_d_new = list(unmatched_d)

        # 9. 미확정 트랙 처리
        if len(unconfirmed) and len(unmatched_d_new):
            det_arr2 = dets_high[unmatched_d_new]
            tlwh2    = (np.array([[d[0], d[1], d[2]-d[0], d[3]-d[1]] for d in det_arr2])
                        if len(det_arr2) else np.zeros((0, 4)))
            iou_d4   = iou_distance(np.array([t.tlwh for t in unconfirmed]), tlwh2)
            matches4, remain_unc, remain_d2 = linear_assignment(iou_d4, thresh=0.7)
            for ui, di4 in matches4:
                t   = unconfirmed[ui]
                d   = det_arr2[di4]
                tlwh = np.array([d[0], d[1], d[2]-d[0], d[3]-d[1]])
                tmp  = self._make_temp_strack(tlwh, d[4], int(d[5]))
                t.update(tmp, self.frame_id)
                activated_now.append(t)
            for ui in remain_unc:
                unconfirmed[ui].mark_removed()
                self.removed_stracks.append(unconfirmed[ui])
            unmatched_d_final = [unmatched_d_new[i] for i in remain_d2]
        else:
            for t in unconfirmed:
                t.mark_removed()
                self.removed_stracks.append(t)
            unmatched_d_final = list(unmatched_d_new)

        # 10. 새 트랙 생성
        new_tracks: List[BoTTrack] = []
        for di5 in unmatched_d_final:
            d = dets_high[di5]
            if d[4] < self.track_thresh:
                continue
            tlwh = np.array([d[0], d[1], d[2]-d[0], d[3]-d[1]])
            feat = det_feats_high[di5] if det_feats_high is not None else None
            t    = BoTTrack(tlwh, d[4], int(d[5]), feat=feat, alpha=self.alpha)
            t.activate(self.kalman, self.frame_id)   # track_id는 STrack 클래스 카운터
            new_tracks.append(t)

        # 11. lost 만료 제거 (frame_id = last seen)
        expire: List[BoTTrack] = []
        for t in self.lost_stracks:
            if self.frame_id - t.frame_id > self.buffer_size:
                t.mark_removed()
                expire.append(t)
        self.removed_stracks.extend(expire)
        self.lost_stracks = [t for t in self.lost_stracks if t not in expire]

        # 12. 상태 업데이트
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = _join(_join(activated_now, refound), new_tracks)
        self.lost_stracks    = [
            t for t in _sub(self.lost_stracks, self.tracked_stracks)
            if t.state != TrackState.Removed
        ]
        self.removed_stracks = self.removed_stracks[-500:]

        return [t for t in self.tracked_stracks if t.is_activated]


# ─────────────────────────────────────────────────────────────────
# 6. 헬퍼 함수
# ─────────────────────────────────────────────────────────────────

def _join(a: list, b: list) -> list:
    ids = {t.track_id for t in a}
    return a + [t for t in b if t.track_id not in ids]


def _sub(a: list, b: list) -> list:
    ids = {t.track_id for t in b}
    return [t for t in a if t.track_id not in ids]


# ─────────────────────────────────────────────────────────────────
# 7. ONNX ReID 래퍼 (추론 시 ONNX Runtime 사용)
# ─────────────────────────────────────────────────────────────────

class OnnxReIDNet:
    """ONNX Runtime으로 ReID 추론 (GPU DLL 없이 CPU 실행)."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"])
        self.input_name  = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = self.sess.run(
            [self.output_name], {self.input_name: x.numpy()})[0]
        return torch.from_numpy(out)

    # nn.Module 인터페이스 흉내 (eval() 호환)
    def eval(self):
        return self


# ─────────────────────────────────────────────────────────────────
# 8. 빠른 동작 테스트
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    reid = SimpleReIDNet(embed_dim=128)
    total_params = sum(p.numel() for p in reid.parameters())
    dummy_img = torch.randn(4, 3, 64, 64)
    out = reid(dummy_img)
    print(f"SimpleReIDNet: {total_params:,} params")
    print(f"  Input: {dummy_img.shape} -> Output: {out.shape}")
    print(f"  L2 norm (should be ~1.0): {out.norm(dim=1).mean():.4f}")

    # BoTSORTTracker 기본 동작
    tracker = BoTSORTTracker(reid_net=reid, track_thresh=0.5, frame_rate=5)
    dets = np.array([[100, 200, 200, 350, 0.9, 0],
                     [300, 100, 450, 280, 0.8, 1]], dtype=np.float32)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    tracks = tracker.update(dets, frame)
    print(f"\nBoTSORTTracker: {len(tracks)} tracks after frame 1")
    for t in tracks:
        print(f"  Track {t.track_id}: tlbr={t.tlbr.astype(int)}, cls={t.cls}")
