"""
coral_publisher/publisher.py

Runs on the Google Coral Dev Board.
Reads a video source (RTSP stream or local file), runs PoseNet inference
via the Edge TPU, and publishes keypoints + periodic JPEG snapshots over
ZeroMQ (msgpack-encoded) at ~10fps.

Usage:
    python publisher.py --rtsp rtsp://<ip>/stream
    python publisher.py --rtsp /path/to/video.mp4   # pre-recorded fallback
"""

import argparse
import logging
import time
import cv2
import zmq
import msgpack
import numpy as np

from pose_engine import PoseEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [publisher] %(message)s")
log = logging.getLogger(__name__)

# PoseNet model path — update if your model file differs
MODEL_PATH = "models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite"

# Publish at 10fps (inference runs at full speed; we downsample for the network)
PUBLISH_FPS = 10

# Include a JPEG snapshot every N published frames (~0.5s at 10fps)
SNAPSHOT_EVERY_N = 5

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def build_message(frame_id: int, keypoints: list[dict], frame: np.ndarray | None) -> bytes:
    """Encode a ZeroMQ message as msgpack bytes."""
    msg: dict = {
        "ts": time.time(),
        "frame_id": frame_id,
        "keypoints": keypoints,
    }
    if frame is not None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            msg["snapshot_jpeg"] = buf.tobytes()
    return msgpack.packb(msg, use_bin_type=True)


def run(rtsp_url: str, zmq_port: int) -> None:
    engine = PoseEngine(MODEL_PATH)
    log.info("PoseNet engine loaded: %s", MODEL_PATH)

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {rtsp_url}")
    log.info("Video source opened: %s", rtsp_url)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://0.0.0.0:{zmq_port}")
    log.info("ZeroMQ PUB bound on tcp://0.0.0.0:%d", zmq_port)

    frame_id = 0
    published = 0
    interval = 1.0 / PUBLISH_FPS
    last_publish = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("End of stream or read error — attempting reconnect")
                cap.release()
                time.sleep(1.0)
                cap = cv2.VideoCapture(rtsp_url)
                continue

            frame_id += 1

            # Run PoseNet inference on every frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            poses, _ = engine.DetectPosesInImage(rgb)

            now = time.time()
            if now - last_publish < interval:
                continue
            last_publish = now

            # Use highest-confidence pose if multiple detected
            if not poses:
                # No person detected: publish empty keypoints so subscriber
                # can enter no-pose state
                keypoints = []
            else:
                pose = max(poses, key=lambda p: p.score)
                h, w = frame.shape[:2]
                keypoints = [
                    {
                        "name": KEYPOINT_NAMES[i],
                        "x": float(kp.yx[1]) / w,
                        "y": float(kp.yx[0]) / h,
                        "score": float(kp.score),
                    }
                    for i, kp in enumerate(pose.keypoints)
                ]

            # Include snapshot on every Nth published frame
            snapshot_frame = frame if (published % SNAPSHOT_EVERY_N == 0) else None

            msg = build_message(frame_id, keypoints, snapshot_frame)
            sock.send(msg)
            published += 1

            if published % 100 == 0:
                log.info("Published %d frames (frame_id=%d)", published, frame_id)

    except KeyboardInterrupt:
        log.info("Shutting down")
    finally:
        cap.release()
        sock.close()
        ctx.term()


def main() -> None:
    parser = argparse.ArgumentParser(description="Coral PoseNet ZeroMQ publisher")
    parser.add_argument("--rtsp", required=True, help="RTSP URL or local video file path")
    parser.add_argument("--port", type=int, default=5555, help="ZeroMQ PUB port (default: 5555)")
    args = parser.parse_args()
    run(args.rtsp, args.port)


if __name__ == "__main__":
    main()
