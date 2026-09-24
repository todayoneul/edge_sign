# Hugging Face Space v3 runtime: run selection

The **primary** runs are `20260924_int8_final/` and `20260924_fp32_final/`.
Both reset the shared tracker before sending frames, pace client enqueue at
10 FPS, use 10 warm-up frames and 50 measured frames, and retain 60 ordered
per-frame responses. The Space source commit is
`9fc2593358a678a5b1597e978a63778bc909fc31` on `cpu-basic`.

All other directories here are **diagnostic pilots**, excluded from the
reported comparison:

| Directory | Why excluded |
|---|---|
| `pilot_int8/` | Two measured frames, sequential request/response client. |
| `pilot_open_int8/` | Three measured frames; pacing clock began before WebSocket handshake. |
| `20260924_int8_open10/`, `20260924_fp32_open10/` | First long open-loop runs; preconnection clock caused an initial send burst. |
| `pilot_corrected_int8/` | Four measured frames; WebSocket send backpressure delayed the client timer. |
| `pilot_browser_queue_int8/` | Five measured frames; browser-like enqueue was checked, but tracker was not reset. |
| `20260924_int8_open10_corrected/`, `20260924_fp32_open10_corrected/` | Timer and queue fixed, but the global ByteTrack state persisted across runs. |
| `pilot_reset_int8/` | Tracker-reset and pacing smoke test with three measured frames. |

The input is the public `seoul_daylight.mp4` demo clip from the Space
(15 frames, 5.28 source FPS, SHA256 in each `config.json`). The client loops
this short clip for sustained throughput measurement. These runs are **not**
an accuracy evaluation or a natural 10 FPS video sequence.

`round_trip_ms` includes client queuing when the 10 FPS offer exceeds Space
capacity. `result_delivery_fps` is derived from the interval between first
and last measured responses; `end_to_end_completion_fps` additionally counts
the time from the first measured send to the final measured response.
`server_inference_ms` comes from the deployed app and covers detector,
ByteTrack, and recognizer. The trace field `client_jpeg_encode_ms` also
includes base64 and JSON payload preparation, not JPEG encoding alone.
Browser canvas rendering is outside this benchmark.
