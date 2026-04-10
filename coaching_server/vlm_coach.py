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
import logging
import time
from collections import deque
from typing import Annotated, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

MODEL = "gemini-2.0-flash-lite"

# Abbreviated joint names for the compact table format (eyes/ears omitted —
# not useful for rep counting and waste column space).
_JOINT_SHORT: dict[str, str] = {
    "nose":           "nose",
    "left_shoulder":  "l_sho",
    "right_shoulder": "r_sho",
    "left_elbow":     "l_elb",
    "right_elbow":    "r_elb",
    "left_wrist":     "l_wri",
    "right_wrist":    "r_wri",
    "left_hip":       "l_hip",
    "right_hip":      "r_hip",
    "left_knee":      "l_kne",
    "right_knee":     "r_kne",
    "left_ankle":     "l_ank",
    "right_ankle":    "r_ank",
}
_JOINT_ORDER = list(_JOINT_SHORT.keys())
_TABLE_HEADER = "t | " + " | ".join(_JOINT_SHORT.values())
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
1. A pipe-delimited keypoint table sampled at ~3fps. Columns are joint names
   (abbreviated); values are "x,y" (normalized 0–1, origin top-left) or "-" if
   the joint was not detected. First column "t" is seconds since window start.
2. A single grounding image of the most recent frame.

Count completed reps by analyzing joint motion patterns:
  push-up    → l_sho / r_sho Y oscillates (down then up)
  squat      → l_hip / r_hip Y oscillates (down then up)
  lunge      → l_kne / r_kne Y oscillates
  jumping_jack → l_wri / r_wri X spreads apart then returns
  sit-up     → l_sho Y rises toward l_hip, then returns
  plank      → body held low and still

If the exercise was consistent across the whole window, output one segment.
If the exercise changed mid-window, output multiple segments in time order.
"""

USER_PROMPT_TEMPLATE = """\
Keypoint table ({frame_count} frames, {window_seconds:.1f}s window):
{table}

Classify exercises and count reps from the joint motion patterns above.\
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

    def _build_keypoint_table(self) -> tuple[str, float]:
        """
        Subsample KP_SEND_COUNT evenly-spaced frames and format as a compact
        pipe-delimited table. Joint names appear once in the header; values are
        "x,y" with leading zeros stripped, or "-" if not detected.
        Returns (table string, window duration in seconds).
        """
        buf = list(self._kp_buffer)
        if not buf:
            return _TABLE_HEADER + "\n(no data)", 0.0

        n = len(buf)
        if n <= KP_SEND_COUNT:
            indices = list(range(n))
        else:
            step = (n - 1) / (KP_SEND_COUNT - 1)
            indices = sorted({round(i * step) for i in range(KP_SEND_COUNT)})
            indices[-1] = n - 1

        t0 = buf[0][0]
        rows = [_TABLE_HEADER]
        for i in indices:
            ts, kps = buf[i]
            kp_map = {
                kp["name"]: kp for kp in kps if kp.get("score", 0) >= 0.5
            }
            cells = [f"{ts - t0:.2f}".lstrip("0") or "0"]
            for joint in _JOINT_ORDER:
                if joint in kp_map:
                    x = f"{kp_map[joint]['x']:.2f}".lstrip("0") or "0"
                    y = f"{kp_map[joint]['y']:.2f}".lstrip("0") or "0"
                    cells.append(f"{x},{y}")
                else:
                    cells.append("-")
            rows.append(" | ".join(cells))

        window_seconds = buf[-1][0] - t0
        return "\n".join(rows), window_seconds

    async def _call_vlm(self) -> CoachingResult:
        table, window_seconds = self._build_keypoint_table()

        prompt = USER_PROMPT_TEMPLATE.format(
            frame_count=min(len(self._kp_buffer), KP_SEND_COUNT),
            window_seconds=window_seconds,
            table=table,
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


