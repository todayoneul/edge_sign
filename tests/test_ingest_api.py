import io
from fastapi.testclient import TestClient
from src.pipeline.app import app


def test_ingest_image_creates_session(sample_image):
    client = TestClient(app)
    with open(sample_image, "rb") as f:
        resp = client.post("/api/ingest",
                           files={"file": ("img.jpg", f.read(), "image/jpeg")},
                           data={"kind": "image"})
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "session"


def test_ingest_bad_url_returns_error():
    client = TestClient(app)
    resp = client.post("/api/ingest", data={"kind": "url", "url": "rtsp://0.0.0.0:1/x"})
    assert resp.status_code == 400
    assert "error" in resp.json()


import json

def test_ws_session_streams_annotated_frames(sample_mp4):
    with TestClient(app) as client:                      # context manager → startup (pipeline) 실행
        with open(sample_mp4, "rb") as f:
            client.post("/api/ingest",
                        files={"file": ("clip.mp4", f.read(), "video/mp4")},
                        data={"kind": "video"})
        with client.websocket_connect("/ws/session") as ws:
            msg = ws.receive_json()
            assert msg["type"] in ("frame", "ended")
            if msg["type"] == "frame":
                assert "tracks" in msg
                jpeg = ws.receive_bytes()
                assert jpeg[:2] == b"\xff\xd8"          # JPEG SOI
