/**
 * Viewport.tsx — 비디오 + 오버레이 캔버스 + 히어로 + 컨트롤
 *
 * 모드①(클라 캡처): webcam / H.264 파일 → useStream.start(getFrame)
 *   - getFrame(): 현재 <video> 프레임을 오프스크린 캔버스로 캡처 → base64 JPEG 반환
 *   - useStream 훅이 setInterval로 getFrame()을 호출, WS 전송
 *
 * 모드②(서버 스트림): ingest() → /ws/session → frameBlobUrl → <img id="stream-img">
 *   - URL 입력, 이미지 업로드, 비호환 코덱 영상 → 자동 폴백
 *   - 서버 w/h를 sentDimsRef에 기록 → letterbox overlay 기준으로 사용
 *
 * 오버레이 렌더:
 *   - useStore(s.tracks) 구독
 *   - rAF 루프에서 renderTracks(ctx, tracks, srcW, srcH, dispW, dispH) 호출
 *   - object-fit: contain 레터박스 오프셋(oX, oY) 계산 후 ctx.translate(oX, oY)
 *   - clear는 오프셋 포함 전체 캔버스 기준
 *
 * 통합 seekbar: SeekBar 컴포넌트 — mode① = videoRef, mode② = seekInfo
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useStore } from "../store";
import { useStream } from "../hooks/useStream";
import { useClientPipeline } from "../hooks/useClientPipeline";
import { useSession } from "../hooks/useSession";
import { useHotkeys } from "../hooks/useHotkeys";
import { renderTracks } from "../lib/draw";
import { SAMPLES } from "../lib/samples";
import SeekBar from "./SeekBar";
import Hero from "./Hero";
import Controls from "./Controls";
import PerfStrip from "./PerfStrip";

// 오프스크린 캔버스 (sendFrame용, 컴포넌트 라이프사이클 밖 싱글턴)
const _cap = document.createElement("canvas");
const _cctx = _cap.getContext("2d")!;

// 온디바이스 모드 전용 640×640 캡처 캔버스 (ORT-Web 입력)
const _cap640 = document.createElement("canvas");
_cap640.width = 640;
_cap640.height = 640;
const _cctx640 = _cap640.getContext("2d", { willReadFrequently: true })!;

// ROI 크롭 캔버스 (온디바이스 인식 — bbox 영역을 size×size RGBA로)
const _roi = document.createElement("canvas");
const _rctx = _roi.getContext("2d", { willReadFrequently: true })!;

/** 모드 구분 — app.js state.mode */
type Mode = "client" | "server";

export default function Viewport() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamImgRef = useRef<HTMLImageElement>(null);
  const viewportRef = useRef<HTMLDivElement>(null);
  const animRef = useRef<number>(0);
  const streamRef = useRef<MediaStream | null>(null);
  const videoBlobRef = useRef<string | null>(null);
  // 인제스트 폴백에서 재사용하기 위해 현재 로드된 File 보관 (app.js loadVideoFile)
  const pendingFileRef = useRef<File | null>(null);
  // 박스 화면 좌표 캐시 — mousemove 히트테스트용 (app.js _geo)
  const geoRef = useRef<Array<{ id: number; x: number; y: number; w: number; h: number }>>([]);

  const tracks = useStore((s) => s.tracks);
  const hoverId = useStore((s) => s.hoverId);
  const setHoverId = useStore((s) => s.setHoverId);
  // selectedVariant is set by PerfStrip; fall back to telemetry.variant from server
  const selectedVariant = useStore((s) => s.selectedVariant ?? s.telemetry.variant);

  // Sent frame dimensions (for letterbox math source)
  // 모드② 에서는 서버가 보내는 w/h로 덮어씀 (handleServerFrame 패턴)
  const sentDimsRef = useRef({ w: 640, h: 480 });

  // Local UI state
  const [mode, setMode] = useState<Mode>("client");
  const [isPlaying, setIsPlaying] = useState(false);
  // 소스가 로드된 상태(재생/일시정지/종료 모두 포함) — idle(Hero 표시)과 구분.
  // 종료 시 isPlaying은 false가 되지만 loaded는 true로 유지 → 마지막 프레임 보존.
  const [loaded, setLoaded] = useState(false);
  const [stageStatus, setStageStatus] = useState("서버 연결 대기 중…");
  const [stageStatusLive, setStageStatusLive] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const [seekbarVisible, setSeekbarVisible] = useState(false);

  const stream = useStream(10);
  const session = useSession();
  const client = useClientPipeline();
  const pipelineMode = useStore((s) => s.pipelineMode);
  const ondeviceModel = useStore((s) => s.ondeviceModel);
  const recordFps = useStore((s) => s.recordFps);
  const pushToast = useStore((s) => s.pushToast);
  // 온디바이스 추론 루프 (setInterval; busy 가드로 비중첩)
  const clientTimerRef = useRef<number | null>(null);
  const clientBusyRef = useRef(false);
  const fpsEmaRef = useRef(0);

  // 서버 프레임의 w/h를 sentDimsRef에 반영 (overlay letterbox 계산 기준)
  // useSession이 store.setFrame을 통해 트랙을 올리므로, seekInfo.pos 변화 시 체크
  const sourceKind = useStore((s) => s.sourceKind);
  useEffect(() => {
    // session.seekInfo는 프레임마다 갱신되며 w/h는 useSession이 직접 노출하지 않음.
    // 대신 useSession 내부에서 setFrame 호출 시 w/h를 같이 넘길 수 있도록
    // sentDimsRef 를 session 의 seekInfo 변화에 맞춰 갱신하는 대신,
    // Viewport 가 session 의 내부 w/h 를 얻으려면 useSession 을 확장해야 한다.
    // 여기서는 useSession 이 노출하는 frameDims 를 사용한다.
    if (session.frameDims) {
      sentDimsRef.current = session.frameDims;
    }
  }, [session.frameDims]);

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
    return { data: _cap.toDataURL("image/jpeg", 0.8), variant: selectedVariant ?? null };
  }, [isPlaying, selectedVariant]);

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

    // Build geo cache for mousemove hit-test (app.js _geo)
    const sx = dW / srcW, sy = dH / srcH;
    geoRef.current = tracks.map((t) => {
      const [x1, y1, x2, y2] = t.bbox;
      return {
        id: t.id,
        x: oX + x1 * sx,
        y: oY + y1 * sy,
        w: (x2 - x1) * sx,
        h: (y2 - y1) * sy,
      };
    });

    ctx.save();
    ctx.translate(oX, oY);
    renderTracks(ctx, tracks, srcW, srcH, dW, dH, hoverId ?? undefined);
    ctx.restore();
  }, [tracks, hoverId]);

  // rAF 루프 — tracks 변경 or 재사이즈 시 재렌더
  useEffect(() => {
    cancelAnimationFrame(animRef.current);
    animRef.current = requestAnimationFrame(renderOverlay);
    return () => cancelAnimationFrame(animRef.current);
  }, [renderOverlay]);

  // ResizeObserver
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

  // ── 온디바이스 추론 루프 (클라 캡처 모드 + pipelineMode==='ondevice') ──────
  const stopClientLoop = useCallback(() => {
    if (clientTimerRef.current != null) {
      clearInterval(clientTimerRef.current);
      clientTimerRef.current = null;
    }
  }, []);

  const startClientLoop = useCallback(() => {
    stopClientLoop();
    let last = performance.now();
    clientTimerRef.current = window.setInterval(() => {
      const video = videoRef.current;
      if (!video || video.readyState < 2 || video.paused || clientBusyRef.current) return;
      clientBusyRef.current = true;
      const vw = video.videoWidth || 640;
      const vh = video.videoHeight || 480;
      _cctx640.drawImage(video, 0, 0, 640, 640);
      const rgba = _cctx640.getImageData(0, 0, 640, 640).data;
      sentDimsRef.current = { w: vw, h: vh };
      // ROI 샘플러 — bbox(원본 좌표) 영역을 size×size RGBA로 크롭 (인식용)
      const roiSampler = (bbox: [number, number, number, number], size: number) => {
        const sw = bbox[2] - bbox[0];
        const sh = bbox[3] - bbox[1];
        if (sw <= 0 || sh <= 0) return null;
        _roi.width = size;
        _roi.height = size;
        _rctx.drawImage(video, bbox[0], bbox[1], sw, sh, 0, 0, size, size);
        return _rctx.getImageData(0, 0, size, size).data;
      };
      void client
        .processFrame(rgba, vw, vh, roiSampler)
        .then((r) => {
          if (r) {
            useStore.getState().setFrame(r);
            const now = performance.now();
            const fps = 1000 / Math.max(1, now - last);
            last = now;
            fpsEmaRef.current = fpsEmaRef.current ? fpsEmaRef.current * 0.8 + fps * 0.2 : fps;
            recordFps(Math.round(fpsEmaRef.current * 10) / 10);
          }
        })
        .catch(() => {})
        .finally(() => {
          clientBusyRef.current = false;
        });
    }, 33);
  }, [client, stopClientLoop, recordFps]);

  /** 클라 캡처 추론 시작 — 모드에 따라 서버 WS(stream) 또는 온디바이스 루프. */
  const startCaptureInference = useCallback(async () => {
    if (pipelineMode === "ondevice") {
      const modelUrl =
        ondeviceModel === "fp16"
          ? "/models/yolov8s_signs_v3_fp16.onnx"
          : "/models/yolov8s_signs_v3_fp32.onnx";
      setStageStatus("온디바이스 모델 로딩… (최초 1회 ~수초)");
      try {
        await client.ensureLoaded(modelUrl);
      } catch {
        setStageStatus("온디바이스 로드 실패 — 서버 모드로 폴백");
        stream.reset();
        stream.start(getFrame);
        return;
      }
      setStageStatusLive(true);
      startClientLoop();
    } else {
      stream.reset();
      stream.start(getFrame);
    }
  }, [pipelineMode, ondeviceModel, client, startClientLoop, stream, getFrame]);

  // ── 공통 stopAll — 두 모드 모두 정리 ─────────────────────────────────────
  const stopAll = useCallback(() => {
    stream.stop();
    stopClientLoop();
    client.reset();
    session.stop();
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
      video.onerror = null;
    }
    if (videoBlobRef.current) {
      URL.revokeObjectURL(videoBlobRef.current);
      videoBlobRef.current = null;
    }
    pendingFileRef.current = null;
    // img 숨기기
    const img = streamImgRef.current;
    if (img) { img.removeAttribute("src"); }
    setMode("client");
    setIsPlaying(false);
    setLoaded(false);
    setSeekbarVisible(false);
    setStageStatusLive(false);
    setStageStatus("정지됨 — 영상을 시작하세요");
    useStore.getState().reset();
  }, [stream, session, stopClientLoop, client]);

  // ── 서버 인제스트 공통 진입점 (app.js ingest + startServerStream) ───────
  const startIngest = useCallback(
    async (kind: "image" | "video" | "url", fileOrUrl: File | string, label: string) => {
      // 클라 스트림만 먼저 정지 (session은 useSession.ingest가 내부에서 처리)
      stream.stop();
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
      const video = videoRef.current;
      if (video) {
        video.pause();
        video.src = "";
        video.srcObject = null;
        video.onerror = null;
      }

      setStageStatus("서버 디코딩 준비 중…");

      const fd = new FormData();
      fd.append("kind", kind);
      if (kind === "url") {
        fd.append("url", fileOrUrl as string);
      } else {
        fd.append("file", fileOrUrl as File);
      }

      await session.ingest(fd);
      // ingest가 오류 없이 반환되면 WS가 열린 것 (session 내부에서 sourceKind="session" 세팅)
      setMode("server");
      setIsPlaying(true);
      setLoaded(true);
      setSeekbarVisible(true);
      setStageStatus(`서버 스트림 · ${label}`);
      setStageStatusLive(true);

      // img 표시, video 숨기기
      const img = streamImgRef.current;
      if (img) img.style.display = "block";
      if (video) video.style.display = "none";
    },
    [stream, session],
  );

  // session.frameBlobUrl 변화 → img.src 갱신 (app.js startServerStream binary 분기)
  useEffect(() => {
    const img = streamImgRef.current;
    if (!img) return;
    if (session.frameBlobUrl) {
      img.src = session.frameBlobUrl;
    }
  }, [session.frameBlobUrl]);

  // session.ended → 상태 반영
  useEffect(() => {
    if (session.ended && mode === "server") {
      setStageStatus("재생 완료 — 다른 입력을 시도하세요");
      setStageStatusLive(false);
      setIsPlaying(false);
    }
  }, [session.ended, mode]);

  // 온디바이스 모델 로드 상태 → 토스트 (준비 완료 / 실패 폴백)
  useEffect(() => {
    if (client.status.loaded) {
      pushToast(`온디바이스 추론 준비 완료 · ${client.status.ep.toUpperCase()}`, "ok");
    }
  }, [client.status.loaded, client.status.ep, pushToast]);
  useEffect(() => {
    if (client.status.error) pushToast("온디바이스 로드 실패 — 서버 모드로 폴백", "warn");
  }, [client.status.error, pushToast]);

  // sourceKind가 session으로 바뀌었을 때 img 표시 보정
  useEffect(() => {
    const img = streamImgRef.current;
    const video = videoRef.current;
    if (sourceKind === "session") {
      if (img) img.style.display = "block";
      if (video) video.style.display = "none";
    } else {
      if (img) img.style.display = "none";
      if (video) video.style.display = "";
    }
  }, [sourceKind]);


  // ── loadVideoFile ─────────────────────────────────────────────────────────
  const loadVideoFile = useCallback(
    (file: File) => {
      // 이미지 → 무조건 서버 인제스트 (app.js loadVideoFile 이미지 분기)
      if (file.type.startsWith("image/")) {
        void startIngest("image", file, file.name);
        return;
      }

      // 비디오: 먼저 클라 디코딩 시도 (app.js loadVideoFile)
      stopAll();
      if (videoBlobRef.current) URL.revokeObjectURL(videoBlobRef.current);
      videoBlobRef.current = URL.createObjectURL(file);
      pendingFileRef.current = file;

      const video = videoRef.current!;
      video.srcObject = null;
      video.loop = false;
      video.src = videoBlobRef.current;
      video.style.display = "";
      video.playbackRate = playbackRate;

      let fellBack = false;
      // 폴백: 비호환 코덱 or play() 거부 → 서버 인제스트 (app.js fallback)
      // 브라우저가 디코딩 못 하는 코덱(MPEG-4 Part2/HEVC 등)은 온디바이스 경로가
      // 불가능(온디바이스도 <video> 디코딩에 의존)하므로 서버 디코딩이 유일한 길.
      // 공개 데모 서버는 CPU 전용이라 느릴 수 있어, 폴백 사실을 명확히 알린다.
      const fallback = () => {
        if (fellBack) return;
        fellBack = true;
        video.onerror = null;
        pushToast(
          "이 코덱은 브라우저에서 직접 재생 불가 → 서버 디코딩(CPU)으로 전환. " +
            "부드러운 체험은 샘플·웹캠·H.264 mp4를 권장합니다.",
          "warn",
        );
        const f = pendingFileRef.current;
        if (f) void startIngest("video", f, f.name);
      };
      video.onerror = fallback;           // MEDIA_ERR_SRC_NOT_SUPPORTED 등

      video.play().then(() => {
        setMode("client");
        setIsPlaying(true);
        setLoaded(true);
        setSeekbarVisible(true);
        setStageStatus(`재생 중 · ${file.name}`);
        setStageStatusLive(true);
        void startCaptureInference();
      }).catch(fallback);
    },
    [stopAll, startIngest, startCaptureInference, playbackRate, pushToast],
  );

  // ── loadSample — 내장 동(同)도메인 샘플 클립 로드 (웹캠 대체, 누구나 바로 체험) ──
  // file 미지정 시 첫 샘플(주간 도심). 모두 H.264라 온디바이스로 매끄럽게 돈다.
  const loadSample = useCallback(
    (file: string = SAMPLES[0].file) => {
      setStageStatus("샘플 영상 불러오는 중…");
      fetch(import.meta.env.BASE_URL + "sample/" + file)
        .then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.blob();
        })
        .then((b) => loadVideoFile(new File([b], file, { type: "video/mp4" })))
        .catch((e) => setStageStatus(`샘플 로드 실패: ${e}`));
    },
    [loadVideoFile],
  );

  // ── handleUrl (Controls URL 입력 → 서버 인제스트) ────────────────────────
  const handleUrl = useCallback(
    (url: string) => {
      if (!url) return;
      void startIngest("url", url, url);
    },
    [startIngest],
  );

  // ── togglePlay — 두 모드 공통 (app.js togglePlay) ────────────────────────
  const togglePlay = useCallback(() => {
    if (mode === "server") {
      if (session.ended) {
        // 끝난 서버 스트림 → 처음부터 재생
        session.restart();
        setIsPlaying(true);
        setStageStatus("서버 스트림 — 처음부터 재생");
        setStageStatusLive(true);
      } else {
        session.togglePlay();
        setIsPlaying(session.serverPlaying);  // 반전 후 값은 useEffect로 보정
      }
    } else {
      const video = videoRef.current;
      if (!video) return;
      if (video.ended) {
        // 끝난 영상 → 처음부터 재생 (트랙 ID도 초기화)
        video.currentTime = 0;
        video.play().then(() => {
          setIsPlaying(true);
          setStageStatus("다시 재생 중");
          setStageStatusLive(true);
          // 온디바이스 모드는 클라 루프가 video.paused로 자가 게이팅 → 서버 스트림 미시작
          if (pipelineMode !== "ondevice") {
            stream.reset();
            stream.start(getFrame);
          }
        }).catch(() => {});
      } else if (video.paused) {
        video.play().then(() => {
          setIsPlaying(true);
          if (pipelineMode !== "ondevice") stream.start(getFrame);
        }).catch(() => {});
      } else {
        video.pause();
        setIsPlaying(false);
        stream.stop();
      }
    }
  }, [mode, session, stream, getFrame, pipelineMode]);

  // session.serverPlaying 변화를 isPlaying에 반영 (서버 pause/play 제어 후)
  useEffect(() => {
    if (mode === "server") setIsPlaying(session.serverPlaying);
  }, [session.serverPlaying, mode]);

  // ── stepBack/stepFwd — 두 모드 (app.js step-back/fwd 버튼) ──────────────
  const stepBack = useCallback(() => {
    if (mode === "server") {
      if (session.ended) {
        // 끝난 뒤 되감기 → 처음부터 재생
        session.restart();
        setIsPlaying(true);
        setStageStatus("서버 스트림 — 처음부터 재생");
        setStageStatusLive(true);
        return;
      }
      const { pos, fps } = session.seekInfo;
      session.seek(Math.max(0, pos - Math.round(5 * fps)));
    } else {
      const video = videoRef.current;
      if (!video?.src) return;
      if (video.ended) {
        // 끝난 뒤 되감기 → 처음부터 재생
        video.currentTime = 0;
        video.play().then(() => {
          setIsPlaying(true);
          setStageStatus("다시 재생 중");
          setStageStatusLive(true);
          if (pipelineMode !== "ondevice") {
            stream.reset();
            stream.start(getFrame);
          }
        }).catch(() => {});
        return;
      }
      video.currentTime = Math.max(0, video.currentTime - 5);
    }
  }, [mode, session, stream, getFrame, pipelineMode]);

  const stepFwd = useCallback(() => {
    if (mode === "server") {
      const { pos, total, fps } = session.seekInfo;
      session.seek(Math.min(total, pos + Math.round(5 * fps)));
    } else {
      const video = videoRef.current;
      if (video?.src && isFinite(video.duration))
        video.currentTime = Math.min(video.duration, video.currentTime + 5);
    }
  }, [mode, session]);

  // ── 재생 속도 (app.js speedRange) ────────────────────────────────────────
  const handlePlaybackRate = useCallback(
    (r: number) => {
      setPlaybackRate(r);
      if (videoRef.current) videoRef.current.playbackRate = r;
      if (mode === "server") session.setSpeed(r);      // 서버 스트림 속도 제어
    },
    [mode, session],
  );

  // ── video 이벤트 리스너 ────────────────────────────────────────────────────
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const onEnded = () => {
      setIsPlaying(false);
      setStageStatusLive(false);
      setStageStatus("재생 완료 — 다른 영상을 올려보세요");
      stream.stop();
    };
    const onPause = () => { if (mode === "client") setIsPlaying(false); };
    const onPlay = () => { if ((video.src || video.srcObject) && mode === "client") setIsPlaying(true); };
    video.addEventListener("ended", onEnded);
    video.addEventListener("pause", onPause);
    video.addEventListener("play", onPlay);
    return () => {
      video.removeEventListener("ended", onEnded);
      video.removeEventListener("pause", onPause);
      video.removeEventListener("play", onPlay);
    };
  }, [stream, mode]);

  // Playback rate sync to video element
  useEffect(() => {
    const video = videoRef.current;
    if (video && mode === "client") video.playbackRate = playbackRate;
  }, [playbackRate, mode]);

  // Viewport drag-drop
  const handleViewportDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer.files[0];
      if (f) loadVideoFile(f);
    },
    [loadVideoFile],
  );

  // Viewport mousemove → canvas 박스 히트테스트 → setHoverId (app.js viewport mousemove)
  const handleViewportMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const canvas = overlayRef.current;
      if (!canvas) return;
      const geo = geoRef.current;
      if (!geo.length) return;
      const r = canvas.getBoundingClientRect();
      const mx = e.clientX - r.left;
      const my = e.clientY - r.top;
      let hit: number | null = null;
      for (const g of geo) {
        if (mx >= g.x && mx <= g.x + g.w && my >= g.y && my <= g.y + g.h) {
          hit = g.id;
          break;
        }
      }
      if (hit !== useStore.getState().hoverId) setHoverId(hit);
    },
    [setHoverId],
  );

  const handleViewportMouseLeave = useCallback(() => {
    if (useStore.getState().hoverId != null) setHoverId(null);
  }, [setHoverId]);

  // ── 서버 seek 콜백 (SeekBar → session.seek) ──────────────────────────────
  const handleServerSeek = useCallback(
    (frameIdx: number) => {
      session.seek(frameIdx);
    },
    [session],
  );

  // ── 재생 단축키 (Space 재생/정지, ←/→ 5초 점프) ──────────────────────────
  // App의 전역 useHotkeys는 ?/·테마 등 앱 레벨 키만 처리(재생키는 noop).
  // 재생 제어는 mode/session/videoRef 컨텍스트를 가진 Viewport가 직접 소유.
  useHotkeys({
    onTogglePlay: togglePlay,
    onStepBack: stepBack,
    onStepForward: stepFwd,
  });

  return (
    <>
      {/* ── 뷰포트 ── */}
      <div
        className={`viewport${loaded ? " playing" : ""}`}
        id="viewport"
        ref={viewportRef}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleViewportDrop}
        onMouseMove={handleViewportMouseMove}
        onMouseLeave={handleViewportMouseLeave}
      >
        {/* 클라 모드① 비디오 엘리먼트 */}
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
            display: mode === "client" && loaded ? "block" : "none",
          }}
        />

        {/* 서버 스트림 모드② img (app.js stream-img) */}
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
            display: "none",               // sourceKind effect로 토글
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

        {/* 히어로 (소스 로드 전 idle 상태에서만) — 종료/일시정지 시엔 마지막 프레임 유지 */}
        {!loaded && (
          <Hero
            onFile={() =>
              (document.getElementById("file-input") as HTMLInputElement | null)?.click()
            }
            onSample={() => loadSample()}
          />
        )}

        {/* 온디바이스 모델 로딩 오버레이 (최초 1회 다운로드+컴파일) */}
        {client.status.loading && (
          <div className="od-loading" role="status" aria-live="polite">
            <span className="od-spinner" aria-hidden="true" />
            <span>
              온디바이스 모델 로딩…
              <small>최초 1회 · WebGPU 컴파일</small>
            </span>
          </div>
        )}

        {/* ── 재생 트랜스포트 바 (뷰포트 내부 오버레이) ── */}
        <SeekBar
          visible={seekbarVisible}
          mode={mode}
          playing={isPlaying}
          onTogglePlay={togglePlay}
          videoRef={videoRef}
          seekInfo={session.seekInfo}
          onServerSeek={handleServerSeek}
          playbackRate={playbackRate}
          onPlaybackRate={handlePlaybackRate}
          onStepBack={stepBack}
          onStepFwd={stepFwd}
        />
      </div>

      {/* ── 성능 스트립 (양자화 A/B + 단계 레이턴시) ── */}
      <PerfStrip />

      {/* ── 소스 선택 바 ── */}
      <Controls
        playing={isPlaying}
        onSample={loadSample}
        onFile={loadVideoFile}
        onStop={stopAll}
        onUrl={handleUrl}
        stageStatus={stageStatus}
        stageStatusLive={stageStatusLive}
      />
    </>
  );
}

