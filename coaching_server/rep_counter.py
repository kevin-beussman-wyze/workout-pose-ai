"""
coaching_server/rep_counter.py

Counts exercise repetitions by tracking joint Y-coordinate peaks/valleys.

Rep counting is split into two layers:

  pending_reps   — reps accumulated since the last VLM confirmation window.
                   Counted in real-time using the motion hint to pick the right
                   joint. Displayed as a live estimate.

  confirmed_totals — dict of exercise → rep count, populated by confirm_window()
                   when the VLM returns an authoritative exercise label. Reps are
                   retroactively assigned to whatever exercise the VLM identified,
                   regardless of what the motion hint thought during counting.

This means rep counts are always attributed to the correct exercise label, at the
cost of ~5s of latency before they move from "pending" to "confirmed".
"""

from collections import deque
from dataclasses import dataclass, field

# Map from exercise name → the keypoint name(s) to track for rep counting
EXERCISE_JOINT_MAP: dict[str, list[str]] = {
    "squat":        ["left_hip", "right_hip"],
    "push-up":      ["left_shoulder", "right_shoulder"],
    "jumping_jack": ["left_wrist", "right_wrist"],
    "sit-up":       ["left_shoulder", "right_shoulder"],
    "plank":        ["left_shoulder", "right_shoulder"],
    "lunge":        ["left_knee", "right_knee"],
    "other":        ["left_hip", "right_hip"],
}

# Motion hint: ordered by how distinctive the movement signature is.
MOTION_HINT_GROUPS: list[tuple[str, list[str]]] = [
    ("jumping_jack", ["left_wrist",    "right_wrist"]),
    ("squat",        ["left_hip",      "right_hip"]),
    ("lunge",        ["left_knee",     "right_knee"]),
    ("push-up",      ["left_shoulder", "right_shoulder"]),
]

MIN_AMPLITUDE  = 0.05   # minimum Y range to count as a rep cycle
HINT_WINDOW    = 15     # frames of Y history per joint (~1.5s at 10fps)
MIN_MOTION_AMP = 0.04   # min Y range for a group to be considered "active"
HINT_CONFIRM   = 8      # frames hint must be stable before switching joint (~0.8s)


@dataclass
class RepCounter:
    # Motion hint's current exercise guess (controls which joint to track)
    exercise: str = "squat"

    # Reps accumulated since the last VLM confirm_window() call
    pending_reps: int = 0

    # Authoritative per-exercise totals, populated by confirm_window()
    confirmed_totals: dict = field(default_factory=dict)

    # Log of every confirmed VLM window
    archived_sets: list[dict] = field(default_factory=list)

    # Peak/valley state machine
    _state: str = "idle"
    _peak: float = 0.0
    _valley: float = 1.0

    # Motion hint state
    _joint_history: dict = field(default_factory=dict)
    _current_hint: str = "squat"
    _hint_stable_frames: int = 0

    def __post_init__(self) -> None:
        for _, joints in MOTION_HINT_GROUPS:
            for j in joints:
                self._joint_history[j] = deque(maxlen=HINT_WINDOW)

    # ------------------------------------------------------------------ #
    # VLM interface                                                        #
    # ------------------------------------------------------------------ #

    def confirm_window(self, vlm_exercise: str) -> None:
        """
        Called by the VLM scheduler with an authoritative exercise label.
        Retroactively assigns all pending_reps to vlm_exercise, then resets
        the window. This is the only place confirmed_totals is written.
        """
        if self.pending_reps > 0:
            self.confirmed_totals[vlm_exercise] = (
                self.confirmed_totals.get(vlm_exercise, 0) + self.pending_reps
            )
            self.archived_sets.append({
                "exercise": vlm_exercise,
                "reps": self.pending_reps,
            })
        self.pending_reps = 0
        # Align hint so it needs HINT_CONFIRM frames to disagree with VLM
        self._current_hint = vlm_exercise
        self._hint_stable_frames = 0

    @property
    def total_reps(self) -> int:
        """Total confirmed reps across all exercises."""
        return sum(self.confirmed_totals.values())

    # ------------------------------------------------------------------ #
    # Per-frame update                                                     #
    # ------------------------------------------------------------------ #

    def update(self, keypoints: list[dict]) -> None:
        """Feed a new keypoint frame. Updates pending_reps in place."""
        kp_map = {kp["name"]: kp for kp in keypoints if kp.get("score", 0) >= 0.5}
        self._update_motion_hint(kp_map)
        y = self._get_tracked_y(kp_map)
        if y is not None:
            self._tick(y)

    # ------------------------------------------------------------------ #
    # Motion hint                                                          #
    # ------------------------------------------------------------------ #

    def _update_motion_hint(self, kp_map: dict) -> None:
        for joint, hist in self._joint_history.items():
            if joint in kp_map:
                hist.append(kp_map[joint]["y"])

        hint = self._dominant_group()

        if hint == self._current_hint:
            self._hint_stable_frames += 1
        else:
            self._current_hint = hint
            self._hint_stable_frames = 1

        # Auto-switch joint tracking when motion hint is stable and different
        if (
            self._hint_stable_frames >= HINT_CONFIRM
            and hint != self.exercise
            and hint != "other"
        ):
            self.exercise = hint
            # Reset state machine so we start fresh on the new joint
            # (pending_reps is NOT reset — VLM handles attribution)
            self._reset_state()

    def _dominant_group(self) -> str:
        best_exercise = "other"
        best_amp = MIN_MOTION_AMP

        for exercise, joints in MOTION_HINT_GROUPS:
            amps = []
            for joint in joints:
                hist = self._joint_history.get(joint, deque())
                if len(hist) >= 3:
                    amps.append(max(hist) - min(hist))
            if not amps:
                continue
            amp = sum(amps) / len(amps)
            if amp > best_amp:
                best_amp = amp
                best_exercise = exercise

        return best_exercise

    # ------------------------------------------------------------------ #
    # Rep counting state machine                                           #
    # ------------------------------------------------------------------ #

    def _get_tracked_y(self, kp_map: dict) -> float | None:
        joint_names = EXERCISE_JOINT_MAP.get(self.exercise, EXERCISE_JOINT_MAP["other"])
        ys = [kp_map[name]["y"] for name in joint_names if name in kp_map]
        return sum(ys) / len(ys) if ys else None

    def _reset_state(self) -> None:
        self._state = "idle"
        self._peak = 0.0
        self._valley = 1.0

    def _tick(self, y: float) -> None:
        """Peak/valley state machine. Increments pending_reps on each detected rep.

        Image coordinates: Y increases downward.
        "Going down" in an exercise → Y increases. A rep = Y rises by
        MIN_AMPLITUDE then falls back by MIN_AMPLITUDE.
        """
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
            if (self._peak - self._valley >= MIN_AMPLITUDE
                    and y >= self._peak - MIN_AMPLITUDE / 2):
                self._state = "down"
                return

        if self._state == "down":
            if y > self._peak:
                self._peak = y
            if self._peak - y >= MIN_AMPLITUDE:
                self.pending_reps += 1
                self._state = "watching"
                self._valley = y
                self._peak = y


