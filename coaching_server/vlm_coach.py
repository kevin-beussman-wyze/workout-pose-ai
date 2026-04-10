"""
coaching_server/vlm_coach.py

Schedules periodic GPT-4o calls with the latest keypoint window + JPEG snapshot.
Returns exercise type, form score, and coaching tip as structured data.
"""

import base64
import json
import logging
import time
from collections import deque

from openai import AsyncOpenAI

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert fitness coach analyzing calisthenics exercises from video frames. "
    "You receive body keypoint data and an image. Respond concisely and accurately."
)

USER_PROMPT_TEMPLATE = """\
Current rep count: {rep_count}
Exercise detected so far: {current_exercise}

Body keypoints (normalized 0.0-1.0 coordinates, origin top-left, ~last 5 seconds of movement):
{keypoints_json}

Based on the attached image and keypoints above:
1. What exercise is this person performing?
   Choose exactly one: squat, push-up, jumping_jack, sit-up, plank, lunge, or unknown
2. Rate their current form 1-10 (10 = textbook perfect).
3. Give ONE specific, actionable coaching tip (max 2 sentences).

Respond ONLY with valid JSON, no markdown:
{{"exercise": "...", "form_score": <integer 1-10>, "tip": "..."}}"""


class VLMCoach:
    def __init__(self, api_key: str, interval_seconds: float = 5.0) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._interval = interval_seconds
        self._last_call: float = 0.0
        self._last_result: dict = {
            "exercise": "unknown",
            "form_score": 0,
            "tip": "Waiting for analysis...",
        }
        # Rolling buffer of recent keypoint frames for context
        self._keypoint_buffer: deque[list[dict]] = deque(maxlen=50)
        self._latest_snapshot: bytes | None = None

    def ingest_frame(self, keypoints: list[dict], snapshot: bytes | None = None) -> None:
        """Feed a keypoint frame (and optional snapshot) into the buffer."""
        if keypoints:
            self._keypoint_buffer.append(keypoints)
        if snapshot is not None:
            self._latest_snapshot = snapshot

    @property
    def last_result(self) -> dict:
        return self._last_result

    async def maybe_call(self, rep_count: int, current_exercise: str) -> dict | None:
        """
        Call GPT-4o if the interval has elapsed and a snapshot is available.
        Returns the new result dict, or None if no call was made.
        """
        now = time.time()
        if now - self._last_call < self._interval:
            return None
        if self._latest_snapshot is None:
            return None
        if not self._keypoint_buffer:
            return None

        self._last_call = now
        try:
            result = await self._call_vlm(rep_count, current_exercise)
            self._last_result = result
            return result
        except Exception as exc:
            log.warning("VLM call failed: %s — retaining last result", exc)
            return None

    async def _call_vlm(self, rep_count: int, current_exercise: str) -> dict:
        keypoints_json = json.dumps(list(self._keypoint_buffer)[-50:], indent=None)
        image_b64 = base64.b64encode(self._latest_snapshot).decode()

        response = await self._client.chat.completions.create(
            model="gpt-4o",
            max_tokens=200,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "low",
                            },
                        },
                        {
                            "type": "text",
                            "content": USER_PROMPT_TEMPLATE.format(
                                rep_count=rep_count,
                                current_exercise=current_exercise,
                                keypoints_json=keypoints_json,
                            ),
                        },
                    ],
                },
            ],
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if model wraps response despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw)
        # Validate expected fields; raise on malformed so caller retains last result
        assert "exercise" in parsed and "form_score" in parsed and "tip" in parsed
        parsed["form_score"] = int(parsed["form_score"])
        return parsed
