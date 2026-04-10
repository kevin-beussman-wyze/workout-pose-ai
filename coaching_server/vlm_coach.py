"""
coaching_server/vlm_coach.py

Schedules periodic Gemini calls with the latest keypoint snapshot.
Returns structured coaching results via Gemini's JSON schema enforcement.

Uses the Google Gen AI SDK (google-genai) with a Vertex AI backend.
Authentication is handled via Application Default Credentials (ADC):
  gcloud auth application-default login
"""

import asyncio
import json
import logging
import time
from typing import Annotated, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

MODEL = "gemini-2.0-flash-lite"

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
    "You receive a body keypoint map and a video frame image. "
    "Respond concisely and accurately."
)

USER_PROMPT_TEMPLATE = """\
Current rep count this window: {rep_count}

Current body keypoints (normalized 0.0–1.0, origin top-left, only confident joints):
{keypoints_compact}

Based on the attached image and keypoints:
1. What calisthenics exercise is this person performing right now?
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
        self._latest_snapshot: bytes | None = None

    def ingest_frame(self, keypoints: list[dict], snapshot: bytes | None = None) -> None:
        if keypoints:
            self._latest_keypoints = keypoints
        if snapshot is not None:
            self._latest_snapshot = snapshot

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
        if self._latest_snapshot is None or not self._latest_keypoints:
            return None

        self._last_call = now
        try:
            result = await self._call_vlm(pending_reps)
            self._last_result = result
            return result
        except Exception as exc:
            log.warning("VLM call failed: %s — retaining last result", exc)
            return None

    async def _call_vlm(self, pending_reps: int) -> CoachingResult:
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
        )

        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=MODEL,
            contents=[
                types.Part(
                    inline_data=types.Blob(mime_type="image/jpeg", data=self._latest_snapshot)
                ),
                types.Part(text=prompt),
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=CoachingResult,
                max_output_tokens=200,
                temperature=0.2,
            ),
        )

        return CoachingResult.model_validate_json(response.text)


