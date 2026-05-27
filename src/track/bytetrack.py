"""
ByteTrack 구현 — Edge-Sign v2 추적 모듈

Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022.
https://arxiv.org/abs/2110.06864

핵심 아이디어:
  - 고신뢰(high-conf) 검출 → 1차 IoU 매칭
  - 미매칭 트랙 + 저신뢰(low-conf) 검출 → 2차 IoU 매칭  (BYTE 전략)
  - 칼만 필터: 상태 [cx, cy, ar, h, vcx, vcy, var, vh] (8-dim constant-velocity)

사용법:
  from src.track import ByteTracker

  tracker = ByteTracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=30)

  for frame in frames:
      detections = yolo_model(frame)          # np.ndarray [N, 6]: x1 y1 x2 y2 conf cls
      tracks = tracker.update(detections)     # List[STrack]
      for t in tracks:
          x1, y1, x2, y2 = t.tlbr
          track_id = t.track_id
          cls = t.cls
"""

from __future__ import annotations

import numpy as np
from collections import OrderedDict
from typing import List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# 1. Kalman Filter  (constant-velocity, 8-dim state)
# ──────────────────────────────────────────────────────────────────────────────

class KalmanFilter:
    """
    8-차원 칼만 필터.

    상태 벡터 x = [cx, cy, ar, h, vcx, vcy, var, vh]
      cx, cy : 바운딩박스 중심 좌표
      ar     : aspect ratio (w/h)
      h      : 높이
      v*     : 대응 속도 성분

    관측 벡터 z = [cx, cy, ar, h]  (4-dim)
    """

    def __init__(self):
        dt = 1.0  # 프레임 당 시간 단위

        # 상태 전이 행렬 F (8×8) — 등속 운동 모델
        self._F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self._F[i, i + 4] = dt

        # 관측 행렬 H (4×8)
        self._H = np.eye(4, 8, dtype=np.float32)

        # 프로세스 노이즈 가중치 (position / velocity)
        self._std_weight_pos = 1.0 / 20.0
        self._std_weight_vel = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """측정값 [cx, cy, ar, h] → 초기 상태 (mean, covariance) 반환."""
        mean_pos = measurement.copy()
        mean_vel = np.zeros(4, dtype=np.float32)
        mean = np.concatenate([mean_pos, mean_vel])

        std = [
            2 * self._std_weight_pos * measurement[3],   # cx
            2 * self._std_weight_pos * measurement[3],   # cy
            1e-2,                                         # ar
            2 * self._std_weight_pos * measurement[3],   # h
            10 * self._std_weight_vel * measurement[3],  # vcx
            10 * self._std_weight_vel * measurement[3],  # vcy
            1e-5,                                         # var
            10 * self._std_weight_vel * measurement[3],  # vh
        ]
        covariance = np.diag(np.square(std, dtype=np.float32))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """시간 t → t+1 상태 예측."""
        std_pos = [
            self._std_weight_pos * mean[3],
            self._std_weight_pos * mean[3],
            1e-2,
            self._std_weight_pos * mean[3],
        ]
        std_vel = [
            self._std_weight_vel * mean[3],
            self._std_weight_vel * mean[3],
            1e-5,
            self._std_weight_vel * mean[3],
        ]
        Q = np.diag(np.square(std_pos + std_vel, dtype=np.float32))

        mean = self._F @ mean
        covariance = self._F @ covariance @ self._F.T + Q
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """상태 공간 → 관측 공간 투영 (측정 노이즈 포함)."""
        std = [
            self._std_weight_pos * mean[3],
            self._std_weight_pos * mean[3],
            1e-1,
            self._std_weight_pos * mean[3],
        ]
        R = np.diag(np.square(std, dtype=np.float32))

        mean_proj = self._H @ mean
        cov_proj = self._H @ covariance @ self._H.T + R
        return mean_proj, cov_proj

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """칼만 업데이트 (측정값으로 상태 보정)."""
        mean_proj, cov_proj = self.project(mean, covariance)

        # 칼만 이득 K
        chol = np.linalg.cholesky(cov_proj)
        K = np.linalg.solve(chol.T, np.linalg.solve(chol, (covariance @ self._H.T).T)).T

        innovation = measurement - mean_proj
        new_mean = mean + innovation @ K.T
        new_cov = covariance - K @ cov_proj @ K.T
        return new_mean, new_cov

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        """마할라노비스 거리 계산 (게이팅용)."""
        mean_proj, cov_proj = self.project(mean, covariance)
        if only_position:
            mean_proj = mean_proj[:2]
            cov_proj = cov_proj[:2, :2]
            measurements = measurements[:, :2]

        diff = measurements - mean_proj
        chol = np.linalg.cholesky(cov_proj)
        z = np.linalg.solve(chol, diff.T)
        return np.sum(z * z, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# 2. 트랙 상태 열거형
# ──────────────────────────────────────────────────────────────────────────────

class TrackState:
    New      = 0   # 방금 생성됨 (아직 confirmed 아님)
    Tracked  = 1   # 정상 추적 중
    Lost     = 2   # 일시적 미매칭 (버퍼 내)
    Removed  = 3   # 제거됨


# ──────────────────────────────────────────────────────────────────────────────
# 3. STrack — 단일 객체 트랙
# ──────────────────────────────────────────────────────────────────────────────

class STrack:
    """
    단일 객체 트랙.

    Attributes:
        track_id  : 전역 고유 ID
        cls       : 클래스 인덱스 (0=traffic_sign, 1=signboard)
        score     : 검출 신뢰도
        state     : TrackState
        frame_id  : 마지막 업데이트 프레임
        start_frame: 트랙 생성 프레임
        tracklet_len: 현재 tracklet 길이
    """

    _id_counter = 0  # 클래스 전역 카운터

    @classmethod
    def reset_id(cls):
        cls._id_counter = 0

    @classmethod
    def _next_id(cls) -> int:
        cls._id_counter += 1
        return cls._id_counter

    def __init__(self, tlwh: np.ndarray, score: float, cls: int):
        # tlwh: [x_top_left, y_top_left, width, height]
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score
        self.cls   = cls

        self.kalman_filter: Optional[KalmanFilter] = None
        self.mean:       Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None

        self.state       = TrackState.New
        self.track_id    = 0  # 아직 confirmed 아님
        self.frame_id    = 0
        self.start_frame = 0
        self.tracklet_len = 0
        self.is_activated = False

    # ── 좌표 변환 ────────────────────────────────────────────────────────────

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """[x1, y1, w, h] → [cx, cy, ar, h]"""
        ret = tlwh.copy()
        ret[:2] += ret[2:] / 2  # x1,y1 → cx,cy
        ret[2]  /= ret[3]       # w → ar = w/h
        return ret

    @staticmethod
    def xyah_to_tlwh(xyah: np.ndarray) -> np.ndarray:
        """[cx, cy, ar, h] → [x1, y1, w, h]"""
        ret = xyah.copy()
        ret[2] *= ret[3]        # ar → w
        ret[:2] -= ret[2:] / 2  # cx,cy → x1,y1
        return ret

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        return self.xyah_to_tlwh(self.mean[:4])

    @property
    def tlbr(self) -> np.ndarray:
        """[x1, y1, x2, y2]"""
        t = self.tlwh.copy()
        t[2:] += t[:2]
        return t

    @property
    def xyxy(self) -> np.ndarray:
        return self.tlbr

    # ── 칼만 필터 조작 ───────────────────────────────────────────────────────

    def activate(self, kalman_filter: KalmanFilter, frame_id: int):
        """첫 활성화 (New → Tracked)."""
        self.kalman_filter = kalman_filter
        self.track_id = self._next_id()
        self.mean, self.covariance = kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: "STrack", frame_id: int, new_id: bool = False):
        """Lost → Tracked 재활성화."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance,
            self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        self.cls   = new_track.cls
        if new_id:
            self.track_id = self._next_id()

    def update(self, new_track: "STrack", frame_id: int):
        """정상 업데이트 (매칭된 검출로 칼만 보정)."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.score = new_track.score
        self.cls   = new_track.cls
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance,
            self.tlwh_to_xyah(new_track.tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True

    def predict(self):
        """칼만 예측 (프레임 이동 전 호출)."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0  # 잃어버린 트랙은 h 속도 0으로 감쇄
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    def __repr__(self) -> str:
        return f"STrack(id={self.track_id}, cls={self.cls}, state={self.state}, frame={self.frame_id})"


# ──────────────────────────────────────────────────────────────────────────────
# 4. IoU 기반 비용 행렬 계산
# ──────────────────────────────────────────────────────────────────────────────

def iou_batch(tlbr_a: np.ndarray, tlbr_b: np.ndarray) -> np.ndarray:
    """
    두 박스 집합 간 IoU 행렬 계산.

    Args:
        tlbr_a: [M, 4] (x1,y1,x2,y2)
        tlbr_b: [N, 4] (x1,y1,x2,y2)
    Returns:
        iou: [M, N]
    """
    if len(tlbr_a) == 0 or len(tlbr_b) == 0:
        return np.zeros((len(tlbr_a), len(tlbr_b)), dtype=np.float32)

    tlbr_a = tlbr_a[:, None, :]  # [M,1,4]
    tlbr_b = tlbr_b[None, :, :]  # [1,N,4]

    inter_x1 = np.maximum(tlbr_a[..., 0], tlbr_b[..., 0])
    inter_y1 = np.maximum(tlbr_a[..., 1], tlbr_b[..., 1])
    inter_x2 = np.minimum(tlbr_a[..., 2], tlbr_b[..., 2])
    inter_y2 = np.minimum(tlbr_a[..., 3], tlbr_b[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    area_a = (tlbr_a[..., 2] - tlbr_a[..., 0]) * (tlbr_a[..., 3] - tlbr_a[..., 1])
    area_b = (tlbr_b[..., 2] - tlbr_b[..., 0]) * (tlbr_b[..., 3] - tlbr_b[..., 1])

    union = area_a + area_b - inter + 1e-7
    return (inter / union).squeeze() if inter.ndim > 2 else inter / union


def iou_cost(tracks: List[STrack], detections: List[STrack]) -> np.ndarray:
    """1 - IoU 비용 행렬 [M, N]."""
    if not tracks or not detections:
        return np.empty((len(tracks), len(detections)), dtype=np.float32)
    t_boxes = np.array([t.tlbr for t in tracks], dtype=np.float32)
    d_boxes = np.array([d.tlbr for d in detections], dtype=np.float32)
    return 1.0 - iou_batch(t_boxes, d_boxes)


# ──────────────────────────────────────────────────────────────────────────────
# 5. 헝가리안 매칭 (scipy 없을 때 간단 greedy fallback 포함)
# ──────────────────────────────────────────────────────────────────────────────

def linear_assignment(cost_matrix: np.ndarray, thresh: float):
    """
    비용 행렬에서 최적 매칭 반환.

    Returns:
        matches        : List[(track_idx, det_idx)]
        unmatched_tracks: List[int]
        unmatched_dets  : List[int]
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ImportError:
        # greedy fallback — 성능은 떨어지지만 scipy 없이도 동작
        row_ind, col_ind = _greedy_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_dets = [], [], []

    matched_mask_r = np.zeros(cost_matrix.shape[0], dtype=bool)
    matched_mask_c = np.zeros(cost_matrix.shape[1], dtype=bool)

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append((int(r), int(c)))
            matched_mask_r[r] = True
            matched_mask_c[c] = True

    unmatched_tracks = np.where(~matched_mask_r)[0].tolist()
    unmatched_dets   = np.where(~matched_mask_c)[0].tolist()

    return matches, unmatched_tracks, unmatched_dets


def _greedy_assignment(cost_matrix: np.ndarray):
    """비용 행렬에서 탐욕 매칭 (scipy 없을 때 fallback)."""
    n_rows, n_cols = cost_matrix.shape
    row_ind, col_ind = [], []
    used_rows = set()
    used_cols = set()

    flat_idx = np.argsort(cost_matrix.ravel())
    for idx in flat_idx:
        r, c = divmod(int(idx), n_cols)
        if r not in used_rows and c not in used_cols:
            row_ind.append(r)
            col_ind.append(c)
            used_rows.add(r)
            used_cols.add(c)

    return row_ind, col_ind


# ──────────────────────────────────────────────────────────────────────────────
# 6. ByteTracker — 메인 추적기
# ──────────────────────────────────────────────────────────────────────────────

class ByteTracker:
    """
    ByteTrack 추적기.

    Args:
        track_thresh : 고신뢰 검출 임계값 (default 0.5)
        match_thresh : 1차 IoU 매칭 임계값 (default 0.8)
        track_buffer : Lost 트랙 유지 프레임 수 (default 30)
        frame_rate   : 영상 fps (track_buffer 스케일링용, default 30)
        low_thresh   : 저신뢰 검출 하한 (default 0.1)
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        frame_rate: int = 30,
        low_thresh: float = 0.1,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.low_thresh   = low_thresh
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)  # fps 스케일

        self.kalman_filter = KalmanFilter()

        self.tracked_stracks:  List[STrack] = []  # 현재 추적 중
        self.lost_stracks:     List[STrack] = []  # 일시적 소실
        self.removed_stracks:  List[STrack] = []  # 제거됨

        self.frame_id = 0
        STrack.reset_id()

    # ── 내부 유틸 ────────────────────────────────────────────────────────────

    @staticmethod
    def _joint_stracks(list_a: List[STrack], list_b: List[STrack]) -> List[STrack]:
        """두 트랙 리스트 합집합 (track_id 기준 중복 제거)."""
        seen = {}
        for t in list_a:
            seen[t.track_id] = t
        for t in list_b:
            if t.track_id not in seen:
                seen[t.track_id] = t
        return list(seen.values())

    @staticmethod
    def _sub_stracks(list_a: List[STrack], list_b: List[STrack]) -> List[STrack]:
        """list_a - list_b (track_id 기준 차집합)."""
        ids_b = {t.track_id for t in list_b}
        return [t for t in list_a if t.track_id not in ids_b]

    @staticmethod
    def _remove_duplicate_stracks(list_a: List[STrack], list_b: List[STrack]):
        """두 리스트에 걸쳐 IoU>0.15 이고 더 짧은 tracklet을 제거."""
        if not list_a or not list_b:
            return list_a, list_b
        iou_dist = iou_cost(list_a, list_b)
        pairs = np.argwhere(iou_dist < 0.15)  # IoU > 0.85 중복
        dup_a, dup_b = set(), set()
        for r, c in pairs:
            ta, tb = list_a[r], list_b[c]
            if (ta.frame_id - ta.start_frame) > (tb.frame_id - tb.start_frame):
                dup_b.add(c)
            else:
                dup_a.add(r)
        res_a = [t for i, t in enumerate(list_a) if i not in dup_a]
        res_b = [t for i, t in enumerate(list_b) if i not in dup_b]
        return res_a, res_b

    # ── 메인 업데이트 ────────────────────────────────────────────────────────

    def update(self, detections: np.ndarray) -> List[STrack]:
        """
        한 프레임의 검출 결과를 받아 업데이트된 트랙 리스트를 반환.

        Args:
            detections: np.ndarray [N, 6]  — x1 y1 x2 y2 conf cls
                        또는 [N, 5]        — x1 y1 x2 y2 conf (cls=0으로 간주)
                        빈 배열도 허용.

        Returns:
            active_tracks: List[STrack]  (TrackState.Tracked 상태인 트랙만)
        """
        self.frame_id += 1

        # ─ 검출 결과 파싱 ─────────────────────────────────────────────────
        dets_high: List[STrack] = []
        dets_low:  List[STrack] = []

        if detections is not None and len(detections) > 0:
            detections = np.asarray(detections, dtype=np.float32)
            for det in detections:
                x1, y1, x2, y2, conf = det[:5]
                cls = int(det[5]) if det.shape[0] > 5 else 0
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                tlwh = np.array([x1, y1, w, h], dtype=np.float32)
                strack = STrack(tlwh, float(conf), cls)
                if conf >= self.track_thresh:
                    dets_high.append(strack)
                elif conf >= self.low_thresh:
                    dets_low.append(strack)

        # ─ 칼만 예측 ──────────────────────────────────────────────────────
        tracked_stracks  = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        strack_pool = self._joint_stracks(tracked_stracks, self.lost_stracks)
        for t in strack_pool:
            t.predict()

        # ─ 1차 매칭: 활성 트랙 ↔ 고신뢰 검출 ──────────────────────────
        cost1 = iou_cost(strack_pool, dets_high)
        matches1, unmatched_tracks1, unmatched_dets1 = linear_assignment(
            cost1, thresh=1.0 - self.match_thresh
        )

        activated_this_frame: List[STrack] = []   # 이번 프레임에 업데이트된 Tracked
        refound_this_frame:   List[STrack] = []   # Lost → re-activated

        for track_i, det_i in matches1:
            track = strack_pool[track_i]
            det   = dets_high[det_i]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_this_frame.append(track)
            else:
                # Lost 트랙 재활성화
                track.re_activate(det, self.frame_id, new_id=False)
                refound_this_frame.append(track)

        # ─ 2차 매칭 (BYTE): 미매칭 Tracked 트랙 ↔ 저신뢰 검출 ────────
        r_tracked = [strack_pool[i] for i in unmatched_tracks1
                     if strack_pool[i].state == TrackState.Tracked]
        cost2 = iou_cost(r_tracked, dets_low)
        matches2, unmatched_tracks2, _ = linear_assignment(
            cost2, thresh=0.5
        )

        for track_i, det_i in matches2:
            track = r_tracked[track_i]
            det   = dets_low[det_i]
            track.update(det, self.frame_id)
            activated_this_frame.append(track)

        # ─ 미매칭 트랙 → Lost ─────────────────────────────────────────
        unmatched_track2_ids = {r_tracked[i].track_id for i in unmatched_tracks2}
        for i in unmatched_tracks1:
            track = strack_pool[i]
            if track.state == TrackState.Tracked and track.track_id not in unmatched_track2_ids:
                continue  # 2차 매칭에서 처리됨
            if track.state != TrackState.Lost:
                track.mark_lost()

        # ─ 새 트랙 생성 (미매칭 고신뢰 검출) ─────────────────────────
        new_tracks: List[STrack] = []
        for i in unmatched_dets1:
            det = dets_high[i]
            if det.score >= self.track_thresh:
                det.activate(self.kalman_filter, self.frame_id)
                new_tracks.append(det)

        # ─ Lost 트랙 버퍼 초과 시 제거 ────────────────────────────────
        remaining_lost: List[STrack] = []
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                self.removed_stracks.append(track)
            else:
                remaining_lost.append(track)

        # ─ 리스트 갱신 ────────────────────────────────────────────────
        # tracked: 기존 + 이번 업데이트 + 재활성화 + 신규
        self.tracked_stracks = self._joint_stracks(
            self._joint_stracks(activated_this_frame, refound_this_frame),
            new_tracks,
        )
        # lost: 이번 프레임에 소실된 트랙 추가 (re-activated 제거)
        newly_lost = [t for t in strack_pool if t.state == TrackState.Lost]
        self.lost_stracks = self._sub_stracks(
            self._joint_stracks(remaining_lost, newly_lost),
            self.tracked_stracks,
        )

        # ─ 중복 제거 ──────────────────────────────────────────────────
        self.tracked_stracks, self.lost_stracks = self._remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        # ─ 활성 트랙만 반환 ───────────────────────────────────────────
        return [t for t in self.tracked_stracks if t.is_activated]

    def reset(self):
        """추적기 상태 초기화 (새 영상 처리 시 호출)."""
        self.tracked_stracks  = []
        self.lost_stracks     = []
        self.removed_stracks  = []
        self.frame_id = 0
        STrack.reset_id()
