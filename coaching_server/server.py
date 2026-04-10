"""
coaching_server/server.py

FastAPI application. Runs on the laptop/server.

- Subscribes to ZeroMQ keypoint stream from Coral publisher
- Counts reps via RepCounter
- Schedules VLM coaching calls via VLMCoach (Gemini via Vertex AI)
- Serves MJPEG video feed from RTSP source via ffmpeg
- Broadcasts live state to browser clients via WebSocket
- Serves the dashboard HTML
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from contextlib import asynccontextmanager

import msgpack
import uvicorn
import zmq
import zmq.asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from rep_counter import RepCounter
from vlm_coach import VLMCoach

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [server] %(message)s")
log = logging.getLogger(__name__)

# --- Config from environment ---
CORAL_IP = os.getenv("CORAL_IP", "localhost")
CORAL_ZMQ_PORT = int(os.getenv("CORAL_ZMQ_PORT", "5555"))
RTSP_URL = os.getenv("RTSP_URL", "")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
VLM_INTERVAL = float(os.getenv("VLM_INTERVAL_SECONDS", "5"))
HOST = os.getenv("COACHING_HOST", "0.0.0.0")
PORT = int(os.getenv("COACHING_PORT", "8000"))

# Minimum keypoint confidence to consider a pose valid
POSE_CONFIDENCE_THRESHOLD = 0.5


# --- Shared state ---
class AppState:
    def __init__(self) -> None:
        self.rep_counter = RepCounter()
        self.vlm_coach = VLMCoach(
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_LOCATION,
            interval_seconds=VLM_INTERVAL,
        )
        self.no_pose = True
        self.clients: list[WebSocket] = []
        self.latest_keypoints: list[dict] = []
        self.tip_history: list[str] = []


state = AppState()
templates = Jinja2Templates(directory="templates")


# --- Lifespan: background tasks ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks = [
        asyncio.create_task(zmq_listener()),
        asyncio.create_task(vlm_scheduler()),
    ]
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- ZeroMQ listener ---
async def zmq_listener() -> None:
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{CORAL_IP}:{CORAL_ZMQ_PORT}")
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    log.info("ZeroMQ SUB connected to tcp://%s:%d", CORAL_IP, CORAL_ZMQ_PORT)

    try:
        while True:
            raw = await sock.recv()
            msg = msgpack.unpackb(raw, raw=False)
            await process_frame(msg)
    except asyncio.CancelledError:
        pass
    finally:
        sock.close()
        ctx.term()


async def process_frame(msg: dict) -> None:
    keypoints: list[dict] = msg.get("keypoints", [])
    snapshot: bytes | None = msg.get("snapshot_jpeg")

    # Determine no-pose state
    confident = [kp for kp in keypoints if kp.get("score", 0) >= POSE_CONFIDENCE_THRESHOLD]
    state.no_pose = len(confident) < 3

    if not state.no_pose:
        state.latest_keypoints = keypoints
        state.rep_counter.update(keypoints)
        state.vlm_coach.ingest_frame(keypoints, snapshot)

    rc = state.rep_counter
    vlm = state.vlm_coach.last_result

    await broadcast({
        "type": "frame",
        "keypoints": keypoints,
        "no_pose": state.no_pose,
        # Live estimate: pending reps under the current motion hint
        "pending_reps": rc.pending_reps,
        "motion_hint": rc.exercise,
        # Confirmed totals: authoritative per-exercise rep counts from VLM
        "confirmed_totals": rc.confirmed_totals,
        "total_reps": rc.total_reps,
        "vlm": {
            "exercise": vlm.exercise,
            "form_score": vlm.form_score,
            "tip": vlm.tip,
        },
    })


# --- VLM scheduler ---
async def vlm_scheduler() -> None:
    try:
        while True:
            await asyncio.sleep(1)
            if state.no_pose:
                continue
            result = await state.vlm_coach.maybe_call(
                pending_reps=state.rep_counter.pending_reps,
            )
            if result:
                # Retroactively assign pending reps to VLM's exercise label
                state.rep_counter.confirm_window(result.exercise)
                if result.tip and (not state.tip_history or state.tip_history[-1] != result.tip):
                    state.tip_history.append(result.tip)
                log.info(
                    "VLM confirmed: exercise=%s form=%d tip=%s",
                    result.exercise, result.form_score, result.tip,
                )
    except asyncio.CancelledError:
        pass



# --- WebSocket broadcast ---
async def broadcast(data: dict) -> None:
    dead = []
    for ws in state.clients:
        try:
            await ws.send_text(json.dumps(data))
        except Exception:
            dead.append(ws)
    for ws in dead:
        state.clients.remove(ws)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    state.clients.append(ws)
    try:
        while True:
            await ws.receive_text()  # keep connection alive; client sends pings
    except WebSocketDisconnect:
        state.clients.remove(ws)


# --- MJPEG video feed ---
def mjpeg_frames(rtsp_url: str):
    """Stream MJPEG frames from ffmpeg subprocess."""
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-f", "mjpeg",
        "-q:v", "5",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    buf = b""
    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            # Split on JPEG boundaries (SOI = 0xFFD8, EOI = 0xFFD9)
            while True:
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9", start + 2) if start != -1 else -1
                if start == -1 or end == -1:
                    break
                jpg = buf[start: end + 2]
                buf = buf[end + 2:]
                yield boundary + jpg + b"\r\n"
    finally:
        proc.terminate()


@app.get("/video_feed")
def video_feed() -> StreamingResponse:
    if not RTSP_URL:
        return StreamingResponse(iter([]), media_type="text/plain")
    return StreamingResponse(
        mjpeg_frames(RTSP_URL),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# --- Dashboard ---
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "ws_url": f"ws://{request.headers.get('host', f'localhost:{PORT}')}/ws",
            "video_url": "/video_feed",
        },
    )


if __name__ == "__main__":
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False)
