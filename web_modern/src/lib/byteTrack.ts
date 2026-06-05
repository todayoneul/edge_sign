/**
 * byteTrack.ts — ByteTrack 추적기 (src/track/bytetrack.py 충실 포팅)
 *
 * Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box",
 * ECCV 2022. 8-dim constant-velocity Kalman + BYTE 2단계 IoU 매칭 + 트랙 생명주기.
 *
 * 브라우저 온디바이스 파이프라인(검출=ORT-Web, 추적=이 모듈, 인식=ORT-Web)의 추적 단계.
 * 서버 e2e_pipeline 과 동일한 트랙 ID/박스 거동을 목표로 하며, golden 테스트로 검증한다
 * (byteTrack.test.ts ↔ scripts/export_bytetrack_golden.py).
 *
 * 비고: Python은 float32 + scipy Hungarian(없으면 greedy). 여기선 float64 + greedy 매칭
 * (Python의 _greedy_assignment 와 동일 — 비용 오름차순 탐욕). 미세 수치차는 있으나
 * 트랙 ID/박스 거동은 동일.
 */

// ── 작은 선형대수 헬퍼 (행렬 = number[][], 벡터 = number[]) ──────────────────
type Mat = number[][];
type Vec = number[];

function matVec(A: Mat, x: Vec): Vec {
  const out = new Array(A.length).fill(0);
  for (let i = 0; i < A.length; i++) {
    let s = 0;
    const row = A[i];
    for (let j = 0; j < x.length; j++) s += row[j] * x[j];
    out[i] = s;
  }
  return out;
}

function matMul(A: Mat, B: Mat): Mat {
  const n = A.length;
  const m = B[0].length;
  const k = B.length;
  const out: Mat = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < k; p++) {
      const a = A[i][p];
      if (a === 0) continue;
      const brow = B[p];
      const orow = out[i];
      for (let j = 0; j < m; j++) orow[j] += a * brow[j];
    }
  }
  return out;
}

function transpose(A: Mat): Mat {
  const n = A.length;
  const m = A[0].length;
  const out: Mat = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) out[j][i] = A[i][j];
  return out;
}

function matAdd(A: Mat, B: Mat): Mat {
  return A.map((row, i) => row.map((v, j) => v + B[i][j]));
}

function matSub(A: Mat, B: Mat): Mat {
  return A.map((row, i) => row.map((v, j) => v - B[i][j]));
}

/** Gauss-Jordan 역행렬 (작은 well-conditioned 행렬용). */
function inverse(A: Mat): Mat {
  const n = A.length;
  const M = A.map((row, i) => [...row, ...row.map((_, j) => (i === j ? 1 : 0))]);
  for (let col = 0; col < n; col++) {
    // 부분 피벗
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (piv !== col) [M[col], M[piv]] = [M[piv], M[col]];
    const d = M[col][col] || 1e-12;
    for (let j = 0; j < 2 * n; j++) M[col][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col];
      if (f === 0) continue;
      for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[col][j];
    }
  }
  return M.map((row) => row.slice(n));
}

function diag(values: Vec): Mat {
  const n = values.length;
  const out: Mat = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) out[i][i] = values[i];
  return out;
}

// ── 1. Kalman Filter (8-dim, constant velocity) ──────────────────────────────
type MeanCov = { mean: Vec; cov: Mat };

class KalmanFilter {
  private F: Mat;
  private stdPos = 1.0 / 20.0;
  private stdVel = 1.0 / 160.0;

  constructor() {
    // F (8×8) 등속 모델: 위치 += 속도
    this.F = Array.from({ length: 8 }, (_, i) =>
      Array.from({ length: 8 }, (_, j) => (i === j ? 1 : i < 4 && j === i + 4 ? 1 : 0)),
    );
  }

  initiate(measurement: Vec): MeanCov {
    const h = measurement[3];
    const mean = [...measurement, 0, 0, 0, 0];
    const std = [
      2 * this.stdPos * h,
      2 * this.stdPos * h,
      1e-2,
      2 * this.stdPos * h,
      10 * this.stdVel * h,
      10 * this.stdVel * h,
      1e-5,
      10 * this.stdVel * h,
    ];
    return { mean, cov: diag(std.map((s) => s * s)) };
  }

  predict(mean: Vec, cov: Mat): MeanCov {
    const h = mean[3];
    const std = [
      this.stdPos * h,
      this.stdPos * h,
      1e-2,
      this.stdPos * h,
      this.stdVel * h,
      this.stdVel * h,
      1e-5,
      this.stdVel * h,
    ];
    const Q = diag(std.map((s) => s * s));
    const newMean = matVec(this.F, mean);
    const newCov = matAdd(matMul(matMul(this.F, cov), transpose(this.F)), Q);
    return { mean: newMean, cov: newCov };
  }

  /** 상태 → 관측(앞 4차원) 투영 + 측정 노이즈 R. H는 선택행렬이라 슬라이스로 처리. */
  private project(mean: Vec, cov: Mat): { meanProj: Vec; covProj: Mat } {
    const h = mean[3];
    const std = [this.stdPos * h, this.stdPos * h, 1e-1, this.stdPos * h];
    const R = diag(std.map((s) => s * s));
    const meanProj = mean.slice(0, 4);
    const covProj = matAdd(
      cov.slice(0, 4).map((row) => row.slice(0, 4)),
      R,
    );
    return { meanProj, covProj };
  }

  update(mean: Vec, cov: Mat, measurement: Vec): MeanCov {
    const { meanProj, covProj } = this.project(mean, cov);
    // cov @ H.T = cov의 앞 4개 열 (8×4)
    const covHt: Mat = cov.map((row) => row.slice(0, 4));
    // K = (cov @ H.T) @ inv(covProj)   (8×4)
    const K = matMul(covHt, inverse(covProj));
    const innovation = measurement.map((m, i) => m - meanProj[i]); // 4
    const newMean = mean.map((m, i) => m + matVec(K, innovation)[i]);
    // new_cov = cov - K @ covProj @ K.T
    const newCov = matSub(cov, matMul(matMul(K, covProj), transpose(K)));
    return { mean: newMean, cov: newCov };
  }
}

// ── 2. 트랙 상태 + STrack ─────────────────────────────────────────────────────
export const TrackState = { New: 0, Tracked: 1, Lost: 2, Removed: 3 } as const;

export interface Detection {
  /** [x1, y1, x2, y2] 픽셀 좌표 */
  bbox: [number, number, number, number];
  score: number;
  cls: number;
}

let _idCounter = 0;
function nextId(): number {
  _idCounter += 1;
  return _idCounter;
}
export function resetTrackId(): void {
  _idCounter = 0;
}

export class STrack {
  private _tlwh: Vec; // [x1, y1, w, h]
  score: number;
  cls: number;
  kf: KalmanFilter | null = null;
  mean: Vec | null = null;
  cov: Mat | null = null;
  state: number = TrackState.New;
  trackId = 0;
  frameId = 0;
  startFrame = 0;
  trackletLen = 0;
  isActivated = false;

  constructor(tlwh: Vec, score: number, cls: number) {
    this._tlwh = tlwh.slice();
    this.score = score;
    this.cls = cls;
  }

  static tlwhToXyah(tlwh: Vec): Vec {
    const [x, y, w, h] = tlwh;
    return [x + w / 2, y + h / 2, w / h, h];
  }
  static xyahToTlwh(xyah: Vec): Vec {
    const [cx, cy, ar, h] = xyah;
    const w = ar * h;
    return [cx - w / 2, cy - h / 2, w, h];
  }

  get tlwh(): Vec {
    if (this.mean === null) return this._tlwh.slice();
    return STrack.xyahToTlwh(this.mean.slice(0, 4));
  }
  get tlbr(): [number, number, number, number] {
    const [x, y, w, h] = this.tlwh;
    return [x, y, x + w, y + h];
  }

  activate(kf: KalmanFilter, frameId: number): void {
    this.kf = kf;
    this.trackId = nextId();
    const { mean, cov } = kf.initiate(STrack.tlwhToXyah(this._tlwh));
    this.mean = mean;
    this.cov = cov;
    this.trackletLen = 0;
    this.state = TrackState.Tracked;
    this.isActivated = true;
    this.frameId = frameId;
    this.startFrame = frameId;
  }

  reActivate(newTrack: STrack, frameId: number, newId = false): void {
    const { mean, cov } = this.kf!.update(this.mean!, this.cov!, STrack.tlwhToXyah(newTrack.tlwh));
    this.mean = mean;
    this.cov = cov;
    this.trackletLen = 0;
    this.state = TrackState.Tracked;
    this.isActivated = true;
    this.frameId = frameId;
    this.score = newTrack.score;
    this.cls = newTrack.cls;
    if (newId) this.trackId = nextId();
  }

  update(newTrack: STrack, frameId: number): void {
    this.frameId = frameId;
    this.trackletLen += 1;
    this.score = newTrack.score;
    this.cls = newTrack.cls;
    const { mean, cov } = this.kf!.update(this.mean!, this.cov!, STrack.tlwhToXyah(newTrack.tlwh));
    this.mean = mean;
    this.cov = cov;
    this.state = TrackState.Tracked;
    this.isActivated = true;
  }

  predict(): void {
    const meanState = this.mean!.slice();
    if (this.state !== TrackState.Tracked) meanState[7] = 0;
    const { mean, cov } = this.kf!.predict(meanState, this.cov!);
    this.mean = mean;
    this.cov = cov;
  }

  markLost(): void {
    this.state = TrackState.Lost;
  }
  markRemoved(): void {
    this.state = TrackState.Removed;
  }
}

// ── 3. IoU 비용 + greedy 매칭 ─────────────────────────────────────────────────
function iou(a: [number, number, number, number], b: [number, number, number, number]): number {
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter + 1e-7);
}

/** 1 - IoU 비용 행렬 [tracks × dets] */
function iouCost(tracks: STrack[], dets: STrack[]): Mat {
  return tracks.map((t) => dets.map((d) => 1 - iou(t.tlbr, d.tlbr)));
}

interface MatchResult {
  matches: [number, number][];
  unmatchedTracks: number[];
  unmatchedDets: number[];
}

/**
 * 탐욕 매칭(비용 오름차순) + thresh 게이팅. Python _greedy_assignment + linear_assignment 동등.
 * nR/nC를 명시 전달 — 트랙 0개·검출 N개(또는 그 반대)일 때 빈 cost 배열이 차원을 잃어
 * unmatched 인덱스를 누락하는 문제를 방지(Python은 shape (0,N)을 보존).
 */
function linearAssignment(cost: Mat, thresh: number, nR: number, nC: number): MatchResult {
  if (nR === 0 || nC === 0) {
    return {
      matches: [],
      unmatchedTracks: Array.from({ length: nR }, (_, i) => i),
      unmatchedDets: Array.from({ length: nC }, (_, i) => i),
    };
  }
  const cells: { r: number; c: number; v: number }[] = [];
  for (let r = 0; r < nR; r++) for (let c = 0; c < nC; c++) cells.push({ r, c, v: cost[r][c] });
  cells.sort((a, b) => a.v - b.v);

  const usedR = new Set<number>();
  const usedC = new Set<number>();
  const matches: [number, number][] = [];
  for (const { r, c, v } of cells) {
    if (usedR.has(r) || usedC.has(c)) continue;
    usedR.add(r);
    usedC.add(c);
    if (v <= thresh) matches.push([r, c]);
  }
  const unmatchedTracks: number[] = [];
  const unmatchedDets: number[] = [];
  for (let r = 0; r < nR; r++) if (!matches.some((m) => m[0] === r)) unmatchedTracks.push(r);
  for (let c = 0; c < nC; c++) if (!matches.some((m) => m[1] === c)) unmatchedDets.push(c);
  return { matches, unmatchedTracks, unmatchedDets };
}

// ── 4. ByteTracker ────────────────────────────────────────────────────────────
export interface ByteTrackerOptions {
  trackThresh?: number;
  matchThresh?: number;
  trackBuffer?: number;
  frameRate?: number;
  lowThresh?: number;
}

export class ByteTracker {
  private trackThresh: number;
  private matchThresh: number;
  private lowThresh: number;
  private maxTimeLost: number;
  private kf = new KalmanFilter();
  private tracked: STrack[] = [];
  private lost: STrack[] = [];
  private removed: STrack[] = [];
  frameId = 0;

  constructor(opts: ByteTrackerOptions = {}) {
    this.trackThresh = opts.trackThresh ?? 0.5;
    this.matchThresh = opts.matchThresh ?? 0.8;
    this.lowThresh = opts.lowThresh ?? 0.1;
    const frameRate = opts.frameRate ?? 30;
    const trackBuffer = opts.trackBuffer ?? 30;
    this.maxTimeLost = Math.floor((frameRate / 30.0) * trackBuffer);
    resetTrackId();
  }

  private static jointStracks(a: STrack[], b: STrack[]): STrack[] {
    const seen = new Map<number, STrack>();
    for (const t of a) seen.set(t.trackId, t);
    for (const t of b) if (!seen.has(t.trackId)) seen.set(t.trackId, t);
    return [...seen.values()];
  }
  private static subStracks(a: STrack[], b: STrack[]): STrack[] {
    const ids = new Set(b.map((t) => t.trackId));
    return a.filter((t) => !ids.has(t.trackId));
  }
  private static removeDuplicate(a: STrack[], b: STrack[]): [STrack[], STrack[]] {
    if (a.length === 0 || b.length === 0) return [a, b];
    const dupA = new Set<number>();
    const dupB = new Set<number>();
    for (let i = 0; i < a.length; i++) {
      for (let j = 0; j < b.length; j++) {
        if (1 - iou(a[i].tlbr, b[j].tlbr) < 0.15) {
          const la = a[i].frameId - a[i].startFrame;
          const lb = b[j].frameId - b[j].startFrame;
          if (la > lb) dupB.add(j);
          else dupA.add(i);
        }
      }
    }
    return [a.filter((_, i) => !dupA.has(i)), b.filter((_, j) => !dupB.has(j))];
  }

  /** 한 프레임 검출 → 업데이트된 활성 트랙 반환. */
  update(detections: Detection[]): STrack[] {
    this.frameId += 1;
    const detsHigh: STrack[] = [];
    const detsLow: STrack[] = [];
    for (const d of detections) {
      const [x1, y1, x2, y2] = d.bbox;
      const w = x2 - x1;
      const h = y2 - y1;
      if (w <= 0 || h <= 0) continue;
      const st = new STrack([x1, y1, w, h], d.score, d.cls);
      if (d.score >= this.trackThresh) detsHigh.push(st);
      else if (d.score >= this.lowThresh) detsLow.push(st);
    }

    const trackedNow = this.tracked.filter((t) => t.state === TrackState.Tracked);
    const pool = ByteTracker.jointStracks(trackedNow, this.lost);
    for (const t of pool) t.predict();

    // 1차: pool ↔ 고신뢰
    const cost1 = iouCost(pool, detsHigh);
    const m1 = linearAssignment(cost1, 1 - this.matchThresh, pool.length, detsHigh.length);
    const activated: STrack[] = [];
    const refound: STrack[] = [];
    for (const [ti, di] of m1.matches) {
      const track = pool[ti];
      const det = detsHigh[di];
      if (track.state === TrackState.Tracked) {
        track.update(det, this.frameId);
        activated.push(track);
      } else {
        track.reActivate(det, this.frameId, false);
        refound.push(track);
      }
    }

    // 2차(BYTE): 미매칭 Tracked ↔ 저신뢰
    const rTracked = m1.unmatchedTracks
      .map((i) => pool[i])
      .filter((t) => t.state === TrackState.Tracked);
    const cost2 = iouCost(rTracked, detsLow);
    const m2 = linearAssignment(cost2, 0.5, rTracked.length, detsLow.length);
    for (const [ti, di] of m2.matches) {
      rTracked[ti].update(detsLow[di], this.frameId);
      activated.push(rTracked[ti]);
    }

    // 미매칭 → Lost
    const unmatched2Ids = new Set(m2.unmatchedTracks.map((i) => rTracked[i].trackId));
    for (const i of m1.unmatchedTracks) {
      const track = pool[i];
      if (track.state === TrackState.Tracked && !unmatched2Ids.has(track.trackId)) continue;
      if (track.state !== TrackState.Lost) track.markLost();
    }

    // 새 트랙
    const newTracks: STrack[] = [];
    for (const i of m1.unmatchedDets) {
      const det = detsHigh[i];
      if (det.score >= this.trackThresh) {
        det.activate(this.kf, this.frameId);
        newTracks.push(det);
      }
    }

    // Lost 버퍼 초과 제거
    const remainingLost: STrack[] = [];
    for (const t of this.lost) {
      if (this.frameId - t.frameId > this.maxTimeLost) {
        t.markRemoved();
        this.removed.push(t);
      } else remainingLost.push(t);
    }

    this.tracked = ByteTracker.jointStracks(ByteTracker.jointStracks(activated, refound), newTracks);
    const newlyLost = pool.filter((t) => t.state === TrackState.Lost);
    this.lost = ByteTracker.subStracks(
      ByteTracker.jointStracks(remainingLost, newlyLost),
      this.tracked,
    );
    [this.tracked, this.lost] = ByteTracker.removeDuplicate(this.tracked, this.lost);

    return this.tracked.filter((t) => t.isActivated);
  }

  reset(): void {
    this.tracked = [];
    this.lost = [];
    this.removed = [];
    this.frameId = 0;
    resetTrackId();
  }
}
