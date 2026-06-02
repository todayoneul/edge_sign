/**
 * SeekBar.tsx — 통합 재생 탐색 바 (모드①·② 공용)
 *
 * 모드① (client): videoRef.current 를 직접 구동.
 *   - timeupdate 리스너 → cur/dur/val 갱신
 *   - input → video.currentTime 즉시 반영 (seeking 중 미리보기)
 *   - LIVE (webcam / 길이 불명) → range disabled + "LIVE"
 *
 * 모드② (server): SeekInfo(pos, total, fps, seekable)로 구동.
 *   - input 중 cur 미리보기, change → onServerSeek(frameIdx) 콜백
 *   - seekable=false → range disabled + "LIVE"
 *
 * app.js 참조:
 *   fmtTime, showSeekbar, setSeekEnabled, resetSeekbar,
 *   seekRange input/change, playToggle click
 */

import { useEffect, useState } from "react";
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
}: SeekBarProps) {
  const [cur, setCur] = useState("0:00");
  const [dur, setDur] = useState("0:00");
  const [val, setVal] = useState(0);
  const [seeking, setSeeking] = useState(false);
  const [seekEnabled, setSeekEnabled] = useState(false);

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

  // ── drag input ────────────────────────────────────────────────────────────
  function handleInput(e: React.ChangeEvent<HTMLInputElement>) {
    setSeeking(true);
    const frac = Number(e.target.value) / 1000;
    setVal(Number(e.target.value));
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

  // ── drag end ──────────────────────────────────────────────────────────────
  function handleChange() {
    const frac = val / 1000;
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
      {/* 재생/일시정지 토글 */}
      <button
        className="pc-btn"
        id="play-toggle"
        type="button"
        aria-label="재생 / 일시정지"
        title="재생 / 일시정지 (Space)"
        onClick={onTogglePlay}
      >
        {playing ? (
          <svg viewBox="0 0 24 24" fill="currentColor" style={{ width: 20, height: 20 }}>
            <path d="M7 5h3v14H7zM14 5h3v14h-3z" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" fill="currentColor" style={{ width: 20, height: 20 }}>
            <path d="M8 5v14l11-7z" />
          </svg>
        )}
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
        onMouseUp={handleChange}
        onTouchEnd={handleChange}
      />

      {/* 총 시간 */}
      <span className="seek-time" id="seek-dur">
        {dur}
      </span>
    </div>
  );
}
