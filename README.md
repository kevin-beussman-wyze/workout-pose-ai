# workout-pose-ai

Real-time calisthenics workout coaching using edge AI on a Google Coral Dev Board + VLM form analysis.

## Architecture

```
[RTSP Camera]
      ↓
[Coral Dev Board — coral_publisher/]
  PoseNet @ ~30fps (EdgeTPU) → ZeroMQ + msgpack on :5555
  (Coral board handles ML inference only — no video transcoding)

[Laptop — coaching_server/]
  ZeroMQ SUB → rep counting (joint Y tracking, 10fps)
  VLM scheduler (GPT-4o) → exercise auto-detection + form coaching every 5s
  ffmpeg RTSP → MJPEG video feed (~1-2s latency)
  FastAPI + WebSockets → browser dashboard

[Browser Dashboard]
  MJPEG video + Canvas skeleton overlay
  Rep counter · Exercise type · Form score · Coaching text
```

## Setup

### Coral Dev Board

```bash
cd coral_publisher
pip install -r requirements.txt
python publisher.py --rtsp rtsp://<camera-ip>/stream
```

### Laptop / Coaching Server

```bash
cd coaching_server
pip install -r requirements.txt

# Set your config
cp .env.example .env
# Edit .env: OPENAI_API_KEY, CORAL_IP, RTSP_URL

python server.py
# Open http://localhost:8000 in your browser
```

### Pre-recorded Video (Demo Fallback)

Pass a local video file instead of an RTSP URL:

```bash
# Coral publisher side
python publisher.py --rtsp /path/to/demo.mp4

# Coaching server side — set in .env:
RTSP_URL=/path/to/demo.mp4
```

## Project Structure

```
workout-pose-ai/
├── coral_publisher/        # Runs on the Coral Dev Board
│   ├── publisher.py        # PoseNet inference → ZeroMQ publish
│   ├── pose_engine.py      # Coral EdgeTPU PoseNet wrapper
│   └── requirements.txt
├── coaching_server/        # Runs on laptop/server
│   ├── server.py           # FastAPI app: ZeroMQ sub, rep counting, VLM, WebSocket, MJPEG
│   ├── rep_counter.py      # Per-exercise rep counting logic
│   ├── vlm_coach.py        # GPT-4o integration and prompt logic
│   ├── templates/
│   │   └── dashboard.html  # Browser dashboard
│   ├── static/js/
│   │   └── dashboard.js    # WebSocket client + canvas skeleton overlay
│   └── requirements.txt
├── docs/
│   └── design.md           # Architecture design document
├── .env.example
├── .gitignore
└── README.md
```

## Supported Exercises

Automatically detected by the VLM — no configuration required:
- Squat
- Push-up
- Jumping jack
- Sit-up
- Plank (time-based, not rep-based)
- Lunge

## Hardware

- **Google Coral Dev Board** (NXP i.MX 8M + Edge TPU)
- **RTSP IP camera** (any — Wyze, Reolink, etc.)
- **Laptop/desktop** running the coaching server and browser dashboard
