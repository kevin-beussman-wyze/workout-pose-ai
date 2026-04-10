"""
coaching_server/rep_counter.py

Counts exercise repetitions by tracking joint Y-coordinate peaks/valleys
over a sliding window of keypoint frames.
"""

from collections import deque
from dataclasses import dataclass, field

# Map from exercise name → the keypoint name(s) to track
# When two names given, track their midpoint Y
EXERCISE_JOINT_MAP: dict[str, list[str]] = {
    "squat":        ["left_hip", "right_hip"],
    "push-up":      ["left_shoulder", "right_shoulder"],
    "jumping_jack": ["left_wrist", "right_wrist"],
    "sit-up":       ["left_shoulder", "right_shoulder"],
    "plank":        ["left_shoulder", "right_shoulder"],
    "lunge":        ["left_knee", "right_knee"],
    "unknown":      ["left_hip", "right_hip"],  # fallback
}

# Minimum Y-axis movement to count as a rep cycle (0.0-1.0 normalized)
MIN_AMPLITUDE = 0.05

# Sliding window size in frames (10fps → 60 frames = 6 seconds)
WINDOW_SIZE = 60


@dataclass
class RepCounter:
    exercise: str = "squat"
    rep_count: int = 0
    archived_sets: list[dict] = field(default_factory=list)

    _window: deque = field(default_factory=lambda: deque(maxlen=WINDOW_SIZE))
    _state: str = "idle"   # "idle" | "down" | "up"
    _peak: float = 0.0
    _valley: float = 1.0

    def set_exercise(self, new_exercise: str) -> None:
        """Switch exercise type. Archives current set and resets counter."""
        if new_exercise == self.exercise:
            return
        if self.rep_count > 0:
            self.archived_sets.append({
                "exercise": self.exercise,
                "reps": self.rep_count,
            })
        self.exercise = new_exercise
        self.rep_count = 0
        self._window.clear()
        self._state = "idle"
        self._peak = 0.0
        self._valley = 1.0

    def update(self, keypoints: list[dict]) -> int:
        """
        Feed a new keypoint frame. Returns updated rep count.
        keypoints: list of {name, x, y, score} dicts.
        """
        y = self._get_tracked_y(keypoints)
        if y is None:
            return self.rep_count

        self._window.append(y)
        self._tick(y)
        return self.rep_count

    def _get_tracked_y(self, keypoints: list[dict]) -> float | None:
        """Extract the mean Y of the joints to track. Returns None if no pose."""
        if not keypoints:
            return None

        joint_names = EXERCISE_JOINT_MAP.get(self.exercise, EXERCISE_JOINT_MAP["unknown"])
        kp_map = {kp["name"]: kp for kp in keypoints}

        ys = []
        for name in joint_names:
            kp = kp_map.get(name)
            if kp and kp["score"] >= 0.5:
                ys.append(kp["y"])

        return sum(ys) / len(ys) if ys else None

    def _tick(self, y: float) -> None:
        """Simple peak/valley state machine for rep counting."""
        # In image coordinates, Y increases downward.
        # "Down" in the exercise (e.g., squat down) → Y increases (lower body in frame).
        # We track: idle → goes down (Y rises) → comes back up (Y falls) = 1 rep.

        if self._state == "idle":
            self._valley = y
            self._peak = y
            self._state = "watching"
            return

        if self._state == "watching":
            if y > self._peak:
                self._peak = y
            if y < self._valley:
                self._valley = y
            # Enter "down" once we've moved significantly downward from the valley
            if self._peak - self._valley >= MIN_AMPLITUDE and y >= self._peak - MIN_AMPLITUDE / 2:
                self._state = "down"
                return

        if self._state == "down":
            if y > self._peak:
                self._peak = y
            # Rep complete when position returns near the valley
            if self._peak - y >= MIN_AMPLITUDE:
                self.rep_count += 1
                self._state = "watching"
                self._valley = y
                self._peak = y
