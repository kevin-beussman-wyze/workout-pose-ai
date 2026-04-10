"""
coaching_server/vlm_coach.py

Schedules periodic LLM calls with a keypoint time series + optional grounding image.
The LLM classifies exercise segments and counts reps from the structured pose data.

Two backends are supported, selected via LLM_BACKEND env var:

  LLM_BACKEND=gemini (default)
    Uses the google-genai SDK with Vertex AI.
    Auth: gcloud auth application-default login
    Config: GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GEMINI_MODEL

  LLM_BACKEND=openai
    Uses the openai SDK pointing at any OpenAI-compatible endpoint.
    Works with vLLM, Ollama, LM Studio, or any local server.
    Config: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL
"""

import abc
import asyncio
import base64
import logging
import time
from collections import deque
from typing import Annotated, Literal

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ── Table format ─────────────────────────────────────────────────────────────
# Eyes/ears omitted — not useful for rep counting and waste column space.
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

KP_BUFFER_SIZE = 60   # store ~6s at 10fps
KP_SEND_COUNT  = 15   # subsample to ~3fps for the LLM prompt

# ── Schema ────────────────────────────────────────────────────────────────────
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


# ── Prompts ───────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are an expert fitness coach analyzing a calisthenics workout session.

You receive:
1. A pipe-delimited keypoint table sampled at ~3fps. Columns are joint names
   (abbreviated); values are "x,y" (normalized 0–1, origin top-left) or "-" if
   the joint was not detected. First column "t" is seconds since window start.
2. Optionally, a single grounding image of the most recent frame.

Count completed reps by analyzing joint motion patterns:
  push-up     → l_sho / r_sho Y oscillates (down then up)
  squat       → l_hip / r_hip Y oscillates (down then up)
  lunge       → l_kne / r_kne Y oscillates
  jumping_jack → l_wri / r_wri X spreads apart then returns
  sit-up      → l_sho Y rises toward l_hip, then returns
  plank       → body held low and still

If the exercise was consistent, output one segment.
If the exercise changed mid-window, output multiple segments in time order.
"""

# For OpenAI-compatible backends: append schema description so json_object
# mode (which enforces valid JSON but not schema) still produces correct output.
_SCHEMA_HINT = """
Respond with JSON exactly matching this structure:
{
  "segments": [
    {
      "exercise": "<one of: squat, push-up, jumping_jack, sit-up, plank, lunge, other>",
      "reps": <integer >= 0>,
      "form_score": <integer 1-10>,
      "tip": "<one actionable coaching tip, max 2 sentences>"
    }
  ]
}
"""

_USER_PROMPT = """\
Keypoint table ({frame_count} frames, {window_seconds:.1f}s window):
{table}

Classify exercises and count reps from the joint motion patterns above.\
"""


# ── Backend abstraction ───────────────────────────────────────────────────────
class _Backend(abc.ABC):
    @abc.abstractmethod
    async def generate(
        self,
        table: str,
        window_seconds: float,
        frame_count: int,
        snapshot: bytes | None,
    ) -> CoachingResult: ...


class _GeminiBackend(_Backend):
    def __init__(self, project: str, location: str, model: str) -> None:
        from google import genai
        from google.genai import types as gtypes
        self._client = genai.Client(vertexai=True, project=project, location=location)
        self._model = model
        self._types = gtypes

    async def generate(self, table, window_seconds, frame_count, snapshot) -> CoachingResult:
        from google.genai import types as gtypes

        prompt = _USER_PROMPT.format(
            frame_count=frame_count,
            window_seconds=window_seconds,
            table=table,
        )
        contents: list = []
        if snapshot is not None:
            contents.append(
                gtypes.Part(inline_data=gtypes.Blob(mime_type="image/jpeg", data=snapshot))
            )
        contents.append(gtypes.Part(text=prompt))

        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self._model,
            contents=contents,
            config=gtypes.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=CoachingResult,
                max_output_tokens=400,
                temperature=0.2,
            ),
        )
        return CoachingResult.model_validate_json(response.text)


class _OpenAIBackend(_Backend):
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    async def generate(self, table, window_seconds, frame_count, snapshot) -> CoachingResult:
        prompt = _USER_PROMPT.format(
            frame_count=frame_count,
            window_seconds=window_seconds,
            table=table,
        )

        user_content: list = []
        if snapshot is not None:
            b64 = base64.b64encode(snapshot).decode()
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        user_content.append({"type": "text", "text": prompt})

        # Attempt structured output via json_schema (vLLM, LM Studio, OpenAI).
        # Fall back to json_object + schema hint in system prompt for servers
        # that support valid-JSON-only mode but not full schema enforcement.
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "CoachingResult",
                        "schema": CoachingResult.model_json_schema(),
                        "strict": False,
                    },
                },
                max_tokens=400,
                temperature=0.2,
            )
        except Exception:
            # Server doesn't support json_schema — fall back to json_object
            log.debug("json_schema unsupported, falling back to json_object mode")
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT + _SCHEMA_HINT},
                    {"role": "user",   "content": user_content},
                ],
                response_format={"type": "json_object"},
                max_tokens=400,
                temperature=0.2,
            )

        return CoachingResult.model_validate_json(response.choices[0].message.content)


# ── Factory ───────────────────────────────────────────────────────────────────
def make_backend(settings: dict) -> _Backend:
    """
    Instantiate the correct backend from settings (typically loaded from env).

    Gemini (default):
      settings["backend"] = "gemini"
      settings["project"], settings["location"], settings["model"]

    OpenAI-compatible:
      settings["backend"] = "openai"
      settings["base_url"], settings["api_key"], settings["model"]
    """
    backend = settings.get("backend", "gemini").lower()
    model = settings.get("model", "")

    if backend == "gemini":
        return _GeminiBackend(
            project=settings["project"],
            location=settings["location"],
            model=model or "gemini-2.0-flash-lite",
        )
    if backend == "openai":
        return _OpenAIBackend(
            base_url=settings["base_url"],
            api_key=settings.get("api_key", "local"),
            model=model,
        )
    raise ValueError(f"Unknown LLM_BACKEND: {backend!r}. Choose 'gemini' or 'openai'.")


# ── VLMCoach ──────────────────────────────────────────────────────────────────
class VLMCoach:
    def __init__(self, backend: _Backend, interval_seconds: float = 5.0) -> None:
        self._backend = backend
        self._interval = interval_seconds
        self._last_call: float = 0.0
        self._last_result: CoachingResult = CoachingResult(
            segments=[ExerciseSegment(
                exercise="other", reps=0, form_score=5, tip="Waiting for analysis...",
            )]
        )
        self._kp_buffer: deque[tuple[float, list[dict]]] = deque(maxlen=KP_BUFFER_SIZE)
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
        if self._last_result.segments:
            return self._last_result.segments[-1].exercise
        return "other"

    async def maybe_call(self, pending_reps: int) -> CoachingResult | None:
        now = time.time()
        if now - self._last_call < self._interval:
            return None
        if not self._kp_buffer:
            return None
        self._last_call = now
        try:
            result = await self._backend.generate(
                *self._build_keypoint_table(),
                snapshot=self._latest_snapshot,
            )
            self._last_result = result
            return result
        except Exception as exc:
            log.warning("LLM call failed: %s — retaining last result", exc)
            return None

    def _build_keypoint_table(self) -> tuple[str, float, int]:
        """Returns (table, window_seconds, frame_count)."""
        buf = list(self._kp_buffer)
        if not buf:
            return _TABLE_HEADER + "\n(no data)", 0.0, 0

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
            kp_map = {kp["name"]: kp for kp in kps if kp.get("score", 0) >= 0.5}
            cells = [f"{ts - t0:.2f}".lstrip("0") or "0"]
            for joint in _JOINT_ORDER:
                if joint in kp_map:
                    x = f"{kp_map[joint]['x']:.2f}".lstrip("0") or "0"
                    y = f"{kp_map[joint]['y']:.2f}".lstrip("0") or "0"
                    cells.append(f"{x},{y}")
                else:
                    cells.append("-")
            rows.append(" | ".join(cells))

        return "\n".join(rows), buf[-1][0] - t0, len(indices)


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


