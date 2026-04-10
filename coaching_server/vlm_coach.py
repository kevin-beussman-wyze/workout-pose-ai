"""
coaching_server/vlm_coach.py

Schedules periodic Gemini calls with a rolling window of keypoint snapshots.
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

# Number of snapshots to buffer and send per VLM call.
# At one snapshot every 0.5s, SNAPSHOT_BUFFER_SIZE=10 covers the full 5s window.
# We subsample to SNAPSHOT_SEND_COUNT evenly-spaced frames to keep token cost low.
SNAPSHOT_BUFFER_SIZE = 12   # store up to 6 seconds of snapshots
SNAPSHOT_SEND_COUNT  = 5    # send this many frames per call (~1s apart in a 5s window)

# Canonical exercise names — used by both VLM output and RepCounter joint map.
# "other" catches exercises outside the supported set without hallucinating a label.
ExerciseType = Literal[
    "squat", "push-up", "jumping_jack", "sit-up", "plank", "lunge", "other"
]


class CoachingResult(BaseModel):
    exercise: ExerciseType
    form_score: Annotated[int, Field(ge=1, le=10)]
    tip: str


SYSTEM_PROMPT = (
    "You are an expert fitness coach analyzing a calisthenics workout session. "
    "You receive a sequence of video frames (oldest to newest) covering the last ~5 seconds, "
    "plus the current body keypoints from the most recent frame. "
    "Use the full sequence to understand motion and detect exercise transitions. "
    "Respond concisely and accurately."
)

USER_PROMPT_TEMPLATE = """\
Pending reps counted this window: {rep_count}

Body keypoints from the most recent frame (normalized 0.0–1.0, confident joints only):
{keypoints_compact}

The {frame_count} images above are equally-spaced frames from the last ~5 seconds (oldest → newest).

Based on this sequence:
1. What calisthenics exercise is this person performing in the MOST RECENT frame?
   (If the exercise changed during the sequence, classify the current state.)
2. Rate their current form 1–10 (10 = textbook perfect).
3. Give ONE specific, actionable coaching tip (max 2 sentences).\
"""


class VLMCoach:
    def __init__(self, project: str, location: str, interval_seconds: float = 5.0) -> None:
        self._client = genai.Client(vertexai=True, project=project, location=location)
        self._interval = interval_seconds
        self._last_call: float = 0.0
        self._last_result: CoachingResult = CoachingResult(
            exercise="other",
            form_score=5,
            tip="Waiting for analysis...",
        )
        self._latest_keypoints: list[dict] = []
        # Rolling buffer of (timestamp, jpeg_bytes) tuples
        self._snapshot_buffer: deque[tuple[float, bytes]] = deque(maxlen=SNAPSHOT_BUFFER_SIZE)

    def ingest_frame(self, keypoints: list[dict], snapshot: bytes | None = None) -> None:
        if keypoints:
            self._latest_keypoints = keypoints
        if snapshot is not None:
            self._snapshot_buffer.append((time.time(), snapshot))

    @property
    def last_result(self) -> CoachingResult:
        return self._last_result

    async def maybe_call(self, pending_reps: int) -> CoachingResult | None:
        """
        Call Gemini if the interval has elapsed and data is available.
        Returns a validated CoachingResult, or None if no call was made.
        """
        now = time.time()
        if now - self._last_call < self._interval:
            return None
        if not self._snapshot_buffer or not self._latest_keypoints:
            return None

        self._last_call = now
        try:
            result = await self._call_vlm(pending_reps)
            self._last_result = result
            return result
        except Exception as exc:
            log.warning("VLM call failed: %s — retaining last result", exc)
            return None

    def _select_snapshots(self) -> list[bytes]:
        """
        Pick SNAPSHOT_SEND_COUNT evenly-spaced frames from the buffer.
        Always includes the most recent frame.
        """
        buf = list(self._snapshot_buffer)
        if len(buf) <= SNAPSHOT_SEND_COUNT:
            return [jpeg for _, jpeg in buf]
        # Evenly-spaced indices across the buffer, always including the last
        step = (len(buf) - 1) / (SNAPSHOT_SEND_COUNT - 1)
        indices = {round(i * step) for i in range(SNAPSHOT_SEND_COUNT)}
        indices.add(len(buf) - 1)
        return [buf[i][1] for i in sorted(indices)]

    async def _call_vlm(self, pending_reps: int) -> CoachingResult:
        snapshots = self._select_snapshots()

        keypoints_compact = json.dumps(
            {
                kp["name"]: [round(kp["x"], 3), round(kp["y"], 3)]
                for kp in self._latest_keypoints
                if kp.get("score", 0) >= 0.5
            },
            indent=None,
        )

        prompt = USER_PROMPT_TEMPLATE.format(
            rep_count=pending_reps,
            keypoints_compact=keypoints_compact,
            frame_count=len(snapshots),
        )

        # Build contents: images first (oldest → newest), then the text prompt
        contents: list[types.Part] = [
            types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=jpeg))
            for jpeg in snapshots
        ]
        contents.append(types.Part(text=prompt))

        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=CoachingResult,
                max_output_tokens=200,
                temperature=0.2,
            ),
        )

        return CoachingResult.model_validate_json(response.text)

