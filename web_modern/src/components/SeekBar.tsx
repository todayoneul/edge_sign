/**
 * SeekBar.tsx — 통합 재생 트랜스포트 바 (모드①·② 공용, 뷰포트 내부 오버레이)
 *
 * 재생/일시정지 · 현재시간 · 탐색 슬라이더 · 총시간 · 5초 점프 · 재생속도까지
 * 한 줄에 모은 비디오 플레이어식 컨트롤. 뷰포트 위에 글래스 바로 떠 있다.
 *
 * 모드① (client): videoRef.current 를 직접 구동.
 *   - timeupdate 리스너 → cur/dur/val 갱신
 *   - input → video.currentTime 즉시 반영 (seeking 중 미리보기)
 *   - LIVE (webcam / 길이 불명) → range disabled + "LIVE"
 *
 * 모드② (server): SeekInfo(pos, total, fps, seekable)로 구동.
 *   - input 중 cur 미리보기, commit → onServerSeek(frameIdx) 콜백
 *   - seekable=false → range disabled + "LIVE"
 *
 * 클릭 탐색 버그 수정:
 *   - 커밋 값은 state(val) 대신 lastFracRef(즉시값)에서 읽는다.
 *     클릭은 input→pointerup 사이에 리렌더가 없어 val이 stale이고,
 *     그 stale 값으로 commit하면 방금 시킹한 위치가 옛 위치로 되돌아갔다.
 */

import { useEffect, useRef, useState } from "react";
import type { SeekInfo } from "../hooks/useSession";

export interface SeekBarProps {
  visible: boolean;
  /** 'client' = videoRef 구동, 'server' = seekInfo 구동 */
  mode: "client" | "server";
  playing: boolean;
  onTogglePlay: () => void;

  // 모드① only
  videoRef?: React.RefObject<HTMLVideoElement | null>;

  // 모드② only
  seekInfo?: SeekInfo;
  onServerSeek?: (frameIdx: number) => void;

  // 재생 속도 + 5초 점프 (트랜스포트 바에 통합)
  playbackRate: number;
  onPlaybackRate: (r: number) => void;
  onStepBack: () => void;
  onStepFwd: () => void;
}

/** 초 → m:ss 포맷 (app.js fmtTime) */
export function fmtTime(s: number): string {
  if (!isFinite(s) || s < 0) return "--:--";
  const m = Math.floor(s / 60);
  const ss = Math.floor(s % 60);
  return `${m}:${ss < 10 ? "0" : ""}${ss}`;
}

export default function SeekBar({
  visible,
  mode,
  playing,
  onTogglePlay,
  videoRef,
  seekInfo,
  onServerSeek,
  playbackRate,
  onPlaybackRate,
  onStepBack,
  onStepFwd,
}: SeekBarProps) {
  const [cur, setCur] = useState("0:00");
  const [dur, setDur] = useState("0:00");
  const [val, setVal] = useState(0);
  const [seeking, setSeeking] = useState(false);
  const [seekEnabled, setSeekEnabled] = useState(false);
  // 커밋 시 stale state 대신 읽을 즉시 frac (클릭 탐색 버그 수정의 핵심)
  const lastFracRef = useRef(0);

  // ── 모드① — video timeupdate ─────────────────────────────────────────────
  useEffect(() => {
    if (mode !== "client") return;
    const video = videoRef?.current;
    if (!video) return;
    const onTimeUpdate = () => {
      if (seeking) return;
      const d = video.duration;
      if (isFinite(d) && d > 0) {
        setVal(Math.round((video.currentTime / d) * 1000));
        setCur(fmtTime(video.currentTime));
        setDur(fmtTime(d));
        setSeekEnabled(true);
      } else {
        // webcam / 길이 불명 → LIVE
        setSeekEnabled(false);
        setDur("LIVE");
        setCur(fmtTime(video.currentTime));
      }
    };
    video.addEventListener("timeupdate", onTimeUpdate);
    return () => video.removeEventListener("timeupdate", onTimeUpdate);
  }, [mode, videoRef, seeking]);

  // ── 모드② — seekInfo 변경 시 갱신 (app.js handleServerFrame 통합 seek 바) ─
  useEffect(() => {
    if (mode !== "server") return;
    if (!seekInfo) return;
    if (seeking) return;                       // 드래그 중엔 갱신 안 함
    const { pos, total, fps, seekable } = seekInfo;
    if (seekable && total > 0) {
      setVal(Math.round(Math.min(1, pos / total) * 1000));
      setCur(fmtTime(pos / fps));
      setDur(fmtTime(total / fps));
      setSeekEnabled(true);
    } else {
      setSeekEnabled(false);
      setDur("LIVE");
    }
  }, [mode, seekInfo, seeking]);

  // visible 변경 시 리셋 (앱이 새 소스 열 때)
  useEffect(() => {
    if (!visible) {
      setVal(0);
      setCur("0:00");
      setDur("0:00");
      setSeeking(false);
    }
  }, [visible]);

  // ── drag/click input — 미리보기 + 즉시값 기록 ──────────────────────────────
  function handleInput(e: React.ChangeEvent<HTMLInputElement>) {
    setSeeking(true);
    const raw = Number(e.target.value);
    const frac = raw / 1000;
    lastFracRef.current = frac;
    setVal(raw);
    if (mode === "server" && seekInfo) {
      const sec = seekInfo.fps > 0 ? (frac * seekInfo.total) / seekInfo.fps : 0;
      setCur(fmtTime(sec));
    } else if (mode === "client") {
      const video = videoRef?.current;
      const d = video?.duration;
      if (video && isFinite(d!) && d! > 0) {
        video.currentTime = frac * d!;
        setCur(fmtTime(frac * d!));
      }
    }
  }

  // ── commit (pointerup/keyup) — lastFracRef(즉시값)로 확정 ──────────────────
  function handleCommit() {
    const frac = lastFracRef.current;
    if (mode === "server" && seekInfo?.seekable) {
      onServerSeek?.(Math.round(frac * seekInfo.total));
    } else if (mode === "client") {
      const video = videoRef?.current;
      const d = video?.duration;
      if (video && isFinite(d!) && d! > 0) video.currentTime = frac * d!;
    }
    setSeeking(false);
  }

  return (
    <div className={`seekbar${visible ? "" : " hidden"}`} id="seekbar">
      {/* 트랜스포트 클러스터: 5초 뒤로(←) · 재생/정지(Space) · 5초 앞으로(→) */}
      <button className="pc-btn step" id="step-back-btn" title="5초 뒤로 (←)" aria-label="5초 뒤로" onClick={onStepBack}>
        <svg viewBox="0 0 24 24" fill="currentColor" style={{ width: 20, height: 20 }}>
          <path d="M11 6v12L4 12zM19 6v12l-7-6z" />
        </svg>
      </button>

      <button
        className="pc-btn"
        id="play-toggle"
        type="button"
        aria-label="재생 / 일시정지"
        title="재생 / 일시정지 (Space)"
        onClick={onTogglePlay}
      >
        {playing ? (
          <svg viewBox="0 0 24 24" fill="currentColor" style={{ width: 24, height: 24 }}>
            <path d="M7 5h3v14H7zM14 5h3v14h-3z" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" fill="currentColor" style={{ width: 24, height: 24 }}>
            <path d="M8 5v14l11-7z" />
          </svg>
        )}
      </button>

      <button className="pc-btn step" id="step-fwd-btn" title="5초 앞으로 (→)" aria-label="5초 앞으로" onClick={onStepFwd}>
        <svg viewBox="0 0 24 24" fill="currentColor" style={{ width: 20, height: 20 }}>
          <path d="M13 6v12l7-6zM5 6v12l7-6z" />
        </svg>
      </button>

      {/* 현재 시간 */}
      <span className="seek-time" id="seek-cur">
        {cur}
      </span>

      {/* 탐색 슬라이더 */}
      <input
        type="range"
        id="seek-range"
        min={0}
        max={1000}
        value={val}
        step={1}
        disabled={!seekEnabled}
        aria-label="재생 위치"
        onChange={handleInput}
        onPointerUp={handleCommit}
        onKeyUp={handleCommit}
      />

      {/* 총 시간 */}
      <span className="seek-time" id="seek-dur">
        {dur}
      </span>

      {/* 재생 속도 */}
      <label className="speed-ctl" title="재생 속도">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
          strokeLinecap="round" strokeLinejoin="round" style={{ width: 18, height: 18 }} aria-hidden="true">
          <path d="M12 3a9 9 0 1 0 9 9" />
          <path d="M12 8v4l3 2" />
        </svg>
        <input
          type="range"
          id="speed-range"
          min="0.25"
          max="1.75"
          step="0.25"
          value={playbackRate}
          onChange={(e) => onPlaybackRate(parseFloat(e.target.value))}
          aria-label="재생 속도"
        />
        <span className="speed-val mono" id="speed-val">
          {playbackRate.toFixed(2)}×
        </span>
      </label>
    </div>
  );
}
