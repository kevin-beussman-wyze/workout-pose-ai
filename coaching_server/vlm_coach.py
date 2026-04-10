"""
coaching_server/vlm_coach.py

Schedules periodic Gemini calls with a keypoint time series + single grounding image.
The LLM classifies exercise segments and counts reps from the structured pose data.
Returns structured coaching results via Gemini's JSON schema enforcement.

Uses the Google Gen AI SDK (google-genai) with a Vertex AI backend.
Authentication is handled via Application Default Credentials (ADC):
  gcloud auth application-default login
"""

import asyncio
import json
import logging
import time
from collections import deque
from typing import Annotated, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

MODEL = "gemini-2.0-flash-lite"

# Keypoint frames to buffer (at 10fps, 60 frames = 6s window)
KP_BUFFER_SIZE = 60
# Subsample to this many frames for the LLM prompt (~3fps over 5s = 15 frames)
KP_SEND_COUNT = 15

# Canonical exercise names used by both LLM output and RepCounter.
ExerciseType = Literal[
    "squat", "push-up", "jumping_jack", "sit-up", "plank", "lunge", "other"
]


class ExerciseSegment(BaseModel):
    exercise: ExerciseType
    reps: Annotated[int, Field(ge=0)]
    form_score: Annotated[int, Field(ge=1, le=10)]
    tip: str


class CoachingResult(BaseModel):
    segments: list[ExerciseSegment]


SYSTEM_PROMPT = """\
You are an expert fitness coach analyzing a calisthenics workout session.

You receive:
1. A time series of body keypoint frames (normalized 0.0–1.0 coordinates, sampled at ~3fps).
   Each frame is {"t": <relative_seconds>, "kp": {joint_name: [x, y], ...}}.
2. A single grounding image of the most recent frame.

Your job:
- Identify one or more exercise segments in the keypoint sequence.
- For each segment, count completed reps by analyzing joint motion patterns.
  Rep patterns: push-up = shoulder Y oscillates; squat/lunge = hip/knee Y oscillates;
  jumping_jack = wrists spread apart and return; sit-up = shoulder Y rises toward hip.
  plank = body held still in low position.
- Score form and give a coaching tip for each segment.

If the exercise was consistent across the whole window, output a single segment.
If the exercise changed mid-window, output multiple segments in time order.
"""

USER_PROMPT_TEMPLATE = """\
Keypoint time series ({frame_count} frames, ~{window_seconds:.1f}s window):
{keypoints_series}

Classify exercises and count reps from the motion patterns above.\
"""


class VLMCoach:
    def __init__(self, project: str, location: str, interval_seconds: float = 5.0) -> None:
        self._client = genai.Client(vertexai=True, project=project, location=location)
        self._interval = interval_seconds
        self._last_call: float = 0.0
        self._last_result: CoachingResult = CoachingResult(
            segments=[ExerciseSegment(
                exercise="other", reps=0, form_score=5, tip="Waiting for analysis..."
            )]
        )
        # Rolling buffer of (timestamp, keypoints) tuples
        self._kp_buffer: deque[tuple[float, list[dict]]] = deque(maxlen=KP_BUFFER_SIZE)
        # Single grounding image (latest snapshot)
        self._latest_snapshot: bytes | None = None

    def ingest_frame(self, keypoints: list[dict], snapshot: bytes | None = None) -> None:
        if keypoints:
            self._kp_buffer.append((time.time(), keypoints))
        if snapshot is not None:
            self._latest_snapshot = snapshot

    @property
    def last_result(self) -> CoachingResult:
        return self._last_result

    @property
    def current_exercise(self) -> ExerciseType:
        """Exercise from the most recent segment."""
        if self._last_result.segments:
            return self._last_result.segments[-1].exercise
        return "other"

    async def maybe_call(self, pending_reps: int) -> CoachingResult | None:
        """
        Call Gemini if the interval has elapsed and data is available.
        Returns a validated CoachingResult, or None if no call was made.
        """
        now = time.time()
        if now - self._last_call < self._interval:
            return None
        if not self._kp_buffer:
            return None

        self._last_call = now
        try:
            result = await self._call_vlm()
            self._last_result = result
            return result
        except Exception as exc:
            log.warning("VLM call failed: %s — retaining last result", exc)
            return None

    def _build_keypoint_series(self) -> tuple[str, float]:
        """
        Subsample KP_SEND_COUNT evenly-spaced frames from the buffer.
        Returns (compact JSON string, window duration in seconds).
        """
        buf = list(self._kp_buffer)
        if not buf:
            return "[]", 0.0

        # Evenly-spaced indices, always including first and last
        n = len(buf)
        if n <= KP_SEND_COUNT:
            indices = list(range(n))
        else:
            step = (n - 1) / (KP_SEND_COUNT - 1)
            indices = sorted({round(i * step) for i in range(KP_SEND_COUNT)})
            indices[-1] = n - 1  # ensure last frame always included

        t0 = buf[0][0]
        frames = []
        for i in indices:
            ts, kps = buf[i]
            confident = {
                kp["name"]: [round(kp["x"], 3), round(kp["y"], 3)]
                for kp in kps
                if kp.get("score", 0) >= 0.5
            }
            frames.append({"t": round(ts - t0, 2), "kp": confident})

        window_seconds = buf[-1][0] - t0
        return json.dumps(frames, separators=(",", ":")), window_seconds

    async def _call_vlm(self) -> CoachingResult:
        keypoints_series, window_seconds = self._build_keypoint_series()

        prompt = USER_PROMPT_TEMPLATE.format(
            frame_count=min(len(self._kp_buffer), KP_SEND_COUNT),
            window_seconds=window_seconds,
            keypoints_series=keypoints_series,
        )

        # Grounding image first (if available), then the keypoint text prompt
        contents: list[types.Part] = []
        if self._latest_snapshot is not None:
            contents.append(
                types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=self._latest_snapshot))
            )
        contents.append(types.Part(text=prompt))

        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=CoachingResult,
                max_output_tokens=400,
                temperature=0.2,
            ),
        )

        return CoachingResult.model_validate_json(response.text)


