/**
 * Viewport.tsx — 비디오 + 오버레이 캔버스 + 히어로 + 컨트롤
 *
 * 모드①(클라 캡처): webcam / H.264 파일 → useStream.start(getFrame)
 *   - getFrame(): 현재 <video> 프레임을 오프스크린 캔버스로 캡처 → base64 JPEG 반환
 *   - useStream 훅이 setInterval로 getFrame()을 호출, WS 전송
 *
 * 오버레이 렌더:
 *   - useStore(s.tracks) 구독
 *   - rAF 루프에서 renderTracks(ctx, tracks, srcW, srcH, dispW, dispH) 호출
 *   - object-fit: contain 레터박스 오프셋(oX, oY) 계산 후 ctx.translate(oX, oY)
 *   - clear는 오프셋 포함 전체 캔버스 기준
 *
 * 모드②(서버 스트림): T7에서 완성; <img id="stream-img"> 자리만 확보
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useStore } from "../store";
import { useStream } from "../hooks/useStream";
import { renderTracks } from "../lib/draw";
import Hero from "./Hero";
import Controls from "./Controls";

// 샘플 클립 목록 (app.js SAMPLES)
const SAMPLES = ["samples/clip_01.mp4", "samples/clip_02.mp4"];

// 오프스크린 캔버스 (sendFrame용, 컴포넌트 라이프사이클 밖 싱글턴)
const _cap = document.createElement("canvas");
const _cctx = _cap.getContext("2d")!;

export default function Viewport() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamImgRef = useRef<HTMLImageElement>(null);
  const viewportRef = useRef<HTMLDivElement>(null);
  const animRef = useRef<number>(0);
  const sampleIdxRef = useRef(0);
  const streamRef = useRef<MediaStream | null>(null);
  const videoBlobRef = useRef<string | null>(null);

  const tracks = useStore((s) => s.tracks);
  const variant = useStore((s) => s.telemetry.variant);
  const playing = useStore((s) => s.playing);

  // Sent frame dimensions (for letterbox math source)
  const sentDimsRef = useRef({ w: 640, h: 480 });

  // Local UI state
  const [isPlaying, setIsPlaying] = useState(false);
  const [stageStatus, setStageStatus] = useState("서버 연결 대기 중…");
  const [stageStatusLive, setStageStatusLive] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1.0);

  // Seekbar state (basic for mode①; full seekbar in T7)
  const [seekbarVisible, setSeekbarVisible] = useState(false);

  const stream = useStream(10);

  // ── getFrame callback (Viewport → useStream) ────────────────────────────
  const getFrame = useCallback((): { data: string; variant?: string | null } | null => {
    const video = videoRef.current;
    if (!video || video.readyState < 2 || !isPlaying) return null;
    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;
    const tw = Math.min(vw, 1280);
    _cap.width = tw;
    _cap.height = Math.round((tw * vh) / vw);
    _cctx.drawImage(video, 0, 0, _cap.width, _cap.height);
    sentDimsRef.current = { w: _cap.width, h: _cap.height };
    return { data: _cap.toDataURL("image/jpeg", 0.8), variant: variant ?? null };
  }, [isPlaying, variant]);

  // ── Overlay render loop (rAF) ────────────────────────────────────────────
  const renderOverlay = useCallback(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cw = canvas.clientWidth;
    const ch = canvas.clientHeight;
    if (canvas.width !== cw * dpr || canvas.height !== ch * dpr) {
      canvas.width = cw * dpr;
      canvas.height = ch * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);

    if (tracks.length === 0) return;

    // ── object-fit: contain 레터박스 계산 ────────────────────────────────
    const { w: srcW, h: srcH } = sentDimsRef.current;
    const vAR = srcW / srcH;
    const cAR = cw / ch;
    let dW: number, dH: number, oX: number, oY: number;
    if (vAR > cAR) {
      dW = cw;
      dH = cw / vAR;
      oX = 0;
      oY = (ch - dH) / 2;
    } else {
      dH = ch;
      dW = ch * vAR;
      oY = 0;
      oX = (cw - dW) / 2;
    }

    // ctx.translate로 레터박스 오프셋 적용 후 renderTracks 호출
    ctx.save();
    ctx.translate(oX, oY);
    renderTracks(ctx, tracks, srcW, srcH, dW, dH);
    ctx.restore();
  }, [tracks]);

  // rAF 루프 — tracks 변경 or 재사이즈 시 재렌더
  useEffect(() => {
    cancelAnimationFrame(animRef.current);
    animRef.current = requestAnimationFrame(renderOverlay);
    return () => cancelAnimationFrame(animRef.current);
  }, [renderOverlay]);

  // ResizeObserver — 뷰포트 크기 변경 시 오버레이 재렌더
  useEffect(() => {
    const vp = viewportRef.current;
    if (!vp) return;
    const ro = new ResizeObserver(() => {
      cancelAnimationFrame(animRef.current);
      animRef.current = requestAnimationFrame(renderOverlay);
    });
    ro.observe(vp);
    return () => ro.disconnect();
  }, [renderOverlay]);

  // ── stopMedia ─────────────────────────────────────────────────────────────
  const stopMedia = useCallback(() => {
    stream.stop();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    const video = videoRef.current;
    if (video) {
      video.pause();
      video.srcObject = null;
      video.src = "";
      video.removeAttribute("src");
    }
    if (videoBlobRef.current) {
      URL.revokeObjectURL(videoBlobRef.current);
      videoBlobRef.current = null;
    }
    setIsPlaying(false);
    setSeekbarVisible(false);
    setStageStatusLive(false);
    setStageStatus("정지됨 — 영상을 시작하세요");
    useStore.getState().reset();
  }, [stream]);

  // ── startWebcam ───────────────────────────────────────────────────────────
  const startWebcam = useCallback(async () => {
    stopMedia();
    try {
      const ms = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });
      streamRef.current = ms;
      const video = videoRef.current!;
      video.srcObject = ms;
      await video.play();
      sentDimsRef.current = { w: 640, h: 480 };
      setIsPlaying(true);
      setSeekbarVisible(true);
      setStageStatus("웹캠 — 실시간 스트리밍 중");
      setStageStatusLive(true);
      stream.reset();
      stream.start(getFrame);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setStageStatus(`웹캠 접근 실패: ${msg}`);
    }
  }, [stopMedia, stream, getFrame]);

  // ── loadVideoFile ─────────────────────────────────────────────────────────
  const loadVideoFile = useCallback(
    (file: File) => {
      if (file.type.startsWith("image/")) {
        // T7: image → server ingest; stub
        setStageStatus(`이미지 입력(T7 미구현): ${file.name}`);
        return;
      }
      stopMedia();
      if (videoBlobRef.current) URL.revokeObjectURL(videoBlobRef.current);
      videoBlobRef.current = URL.createObjectURL(file);
      const video = videoRef.current!;
      video.srcObject = null;
      video.loop = false;
      video.src = videoBlobRef.current;
      video.playbackRate = playbackRate;

      let fellBack = false;
      const fallback = () => {
        if (fellBack) return;
        fellBack = true;
        video.onerror = null;
        // T7: server ingest fallback
        setStageStatus(`비호환 코덱 — 서버 인제스트(T7) 필요: ${file.name}`);
      };
      video.onerror = fallback;
      video.play().then(() => {
        setIsPlaying(true);
        setSeekbarVisible(true);
        setStageStatus(`재생 중 · ${file.name}`);
        setStageStatusLive(true);
        stream.reset();
        stream.start(getFrame);
      }).catch(fallback);
    },
    [stopMedia, stream, getFrame, playbackRate],
  );

  // ── loadSample ────────────────────────────────────────────────────────────
  const loadSample = useCallback(() => {
    const url = SAMPLES[sampleIdxRef.current % SAMPLES.length];
    sampleIdxRef.current++;
    stopMedia();
    const video = videoRef.current!;
    video.srcObject = null;
    video.loop = true;
    video.src = url;
    video.playbackRate = playbackRate;
    video.onerror = () => {
      video.onerror = null;
      // T7: server ingest fallback
      setStageStatus("샘플 재생 실패 — 서버 디코딩 시도(T7)");
    };
    video.play().then(() => {
      setIsPlaying(true);
      setSeekbarVisible(true);
      setStageStatus("샘플 영상 — FP32⇄INT8 토글을 바꿔보세요");
      setStageStatusLive(true);
      stream.reset();
      stream.start(getFrame);
    }).catch(() => {
      setStageStatus("샘플 재생 실패");
    });
  }, [stopMedia, stream, getFrame, playbackRate]);

  // ── togglePlay (seekbar) ──────────────────────────────────────────────────
  const togglePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      video.play().then(() => {
        setIsPlaying(true);
        stream.start(getFrame);
      }).catch(() => {});
    } else {
      video.pause();
      setIsPlaying(false);
      stream.stop();
    }
  }, [stream, getFrame]);

  // Sync playing state from store
  useEffect(() => {
    if (!playing && isPlaying) {
      // store reset was called externally
    }
  }, [playing, isPlaying]);

  // Video event listeners
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const onEnded = () => {
      setIsPlaying(false);
      setStageStatusLive(false);
      setStageStatus("재생 완료 — 다른 영상을 올려보세요");
      stream.stop();
    };
    const onPause = () => {
      setIsPlaying(false);
    };
    const onPlay = () => {
      if (video.src || video.srcObject) setIsPlaying(true);
    };
    video.addEventListener("ended", onEnded);
    video.addEventListener("pause", onPause);
    video.addEventListener("play", onPlay);
    return () => {
      video.removeEventListener("ended", onEnded);
      video.removeEventListener("pause", onPause);
      video.removeEventListener("play", onPlay);
    };
  }, [stream]);

  // Playback rate sync
  useEffect(() => {
    const video = videoRef.current;
    if (video) video.playbackRate = playbackRate;
  }, [playbackRate]);

  // Viewport drag-drop
  const handleViewportDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer.files[0];
      if (f) loadVideoFile(f);
    },
    [loadVideoFile],
  );

  // Stub handlers for T7
  const handleUrl = useCallback((url: string) => {
    setStageStatus(`URL 입력(T7 미구현): ${url}`);
  }, []);

  // Step ±5s
  const stepBack = useCallback(() => {
    const video = videoRef.current;
    if (video?.src) video.currentTime = Math.max(0, video.currentTime - 5);
  }, []);
  const stepFwd = useCallback(() => {
    const video = videoRef.current;
    if (video?.src && isFinite(video.duration))
      video.currentTime = Math.min(video.duration, video.currentTime + 5);
  }, []);

  return (
    <>
      {/* ── 뷰포트 ── */}
      <div
        className={`viewport${isPlaying ? " playing" : ""}`}
        id="viewport"
        ref={viewportRef}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleViewportDrop}
      >
        {/* 클라 모드 비디오 엘리먼트 */}
        <video
          ref={videoRef}
          id="video-el"
          muted
          playsInline
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            objectFit: "contain",
            display: isPlaying ? "block" : "none",
          }}
        />

        {/* 서버 스트림 모드②용 img (T7에서 활성화) */}
        <img
          ref={streamImgRef}
          id="stream-img"
          alt=""
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            objectFit: "contain",
            display: "none",
            background: "var(--viewport)",
          }}
        />

        {/* 오버레이 캔버스 */}
        <canvas
          ref={overlayRef}
          id="overlay-canvas"
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            pointerEvents: "none",
            zIndex: 2,
          }}
        />

        {/* 히어로 (재생 전) */}
        {!isPlaying && (
          <Hero
            onFile={() =>
              (document.getElementById("file-input") as HTMLInputElement | null)?.click()
            }
            onSample={loadSample}
            onWebcam={startWebcam}
          />
        )}
      </div>

      {/* ── 통합 재생 탐색 바 ── */}
      <SeekBar
        visible={seekbarVisible}
        videoRef={videoRef}
        playing={isPlaying}
        onTogglePlay={togglePlay}
      />

      {/* ── 성능 스트립 placeholder (T8에서 완성) ── */}
      <PerfStripPlaceholder />

      {/* ── 컨트롤 바 ── */}
      <Controls
        playing={isPlaying}
        onSample={loadSample}
        onWebcam={startWebcam}
        onFile={loadVideoFile}
        onStop={stopMedia}
        onUrl={handleUrl}
        stageStatus={stageStatus}
        stageStatusLive={stageStatusLive}
        playbackRate={playbackRate}
        onPlaybackRate={(r) => setPlaybackRate(r)}
        onStepBack={stepBack}
        onStepFwd={stepFwd}
      />
    </>
  );
}

// ── SeekBar ──────────────────────────────────────────────────────────────────
// 클라 모드①용 기본 seekbar (app.js 통합 seekbar 포팅, 서버 모드 부분은 T7)
function SeekBar({
  visible,
  videoRef,
  playing,
  onTogglePlay,
}: {
  visible: boolean;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  playing: boolean;
  onTogglePlay: () => void;
}) {
  const [cur, setCur] = useState("0:00");
  const [dur, setDur] = useState("0:00");
  const [val, setVal] = useState(0);
  const [seeking, setSeeking] = useState(false);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const onTimeUpdate = () => {
      if (seeking) return;
      const d = video.duration;
      if (isFinite(d) && d > 0) {
        setVal(Math.round((video.currentTime / d) * 1000));
        setCur(fmtTime(video.currentTime));
        setDur(fmtTime(d));
      } else {
        setDur("LIVE");
        setCur(fmtTime(video.currentTime));
      }
    };
    video.addEventListener("timeupdate", onTimeUpdate);
    return () => video.removeEventListener("timeupdate", onTimeUpdate);
  }, [videoRef, seeking]);

  function handleInput(e: React.ChangeEvent<HTMLInputElement>) {
    setSeeking(true);
    const frac = Number(e.target.value) / 1000;
    const video = videoRef.current;
    const d = video?.duration;
    if (video && isFinite(d!) && d! > 0) {
      video.currentTime = frac * d!;
      setCur(fmtTime(frac * d!));
    }
    setVal(Number(e.target.value));
  }

  function handleChange() {
    setSeeking(false);
  }

  return (
    <div className={`seekbar${visible ? "" : " hidden"}`} id="seekbar">
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
      <span className="seek-time" id="seek-cur">
        {cur}
      </span>
      <input
        type="range"
        id="seek-range"
        min={0}
        max={1000}
        value={val}
        step={1}
        aria-label="재생 위치"
        onChange={handleInput}
        onMouseUp={handleChange}
        onTouchEnd={handleChange}
      />
      <span className="seek-time" id="seek-dur">
        {dur}
      </span>
    </div>
  );
}

// ── PerfStrip placeholder (T8) ───────────────────────────────────────────────
function PerfStripPlaceholder() {
  const telemetry = useStore((s) => s.telemetry);
  const stageMs = telemetry.stageMs;

  const d = stageMs?.detect ?? 0;
  const t = stageMs?.track ?? 0;
  const r = stageMs?.recognize ?? 0;
  const tot = Math.max(d + t + r, 0.001);

  return (
    <div className="perf-strip" id="perf-strip">
      <div className="perf-block" style={{ flex: 1 }}>
        <span className="perf-tag">파이프라인 단계 · 프레임당 소요</span>
        <div className="pipeline-flow" id="pipeline-flow">
          {(
            [
              { key: "detect", label: "검출", ms: d, frac: d / tot },
              { key: "track", label: "추적", ms: t, frac: t / tot },
              { key: "recog", label: "인식", ms: r, frac: r / tot },
            ] as const
          ).map(({ key, label, ms, frac }) => (
            <div className="pstage" data-stage={key} key={key}>
              <div className="pstage-head">
                <span className="pstage-name">
                  {label}
                  <span className="bn">병목</span>
                </span>
                <span className="pstage-ms" id={`ms-${key}`}>
                  {ms > 0 ? (
                    <>
                      {ms.toFixed(1)}
                      <small>ms</small>
                    </>
                  ) : (
                    "—"
                  )}
                </span>
              </div>
              <div className="pbar">
                <i
                  id={`bar-${key}`}
                  style={{ width: `${Math.round(Math.min(1, frac) * 100)}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── 시간 포맷 ────────────────────────────────────────────────────────────────
function fmtTime(s: number): string {
  if (!isFinite(s) || s < 0) return "--:--";
  const m = Math.floor(s / 60);
  const ss = Math.floor(s % 60);
  return `${m}:${ss < 10 ? "0" : ""}${ss}`;
}
