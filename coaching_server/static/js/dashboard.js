/**
 * dashboard.js
 *
 * WebSocket client for the Workout Pose AI dashboard.
 * Handles:
 *  - Live keypoint skeleton rendering on a canvas overlaid on the MJPEG feed
 *  - Rep counter, exercise badge, form score updates
 *  - Coaching tip feed (newest at top)
 */

const WS_URL = document.currentScript.dataset.wsUrl;

// PoseNet 17-keypoint connections for skeleton rendering
const CONNECTIONS = [
  ["nose", "left_eye"], ["nose", "right_eye"],
  ["left_eye", "left_ear"], ["right_eye", "right_ear"],
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"],
  ["left_shoulder", "left_hip"], ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"], ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"], ["right_knee", "right_ankle"],
];

const JOINT_COLOR = "rgba(64, 192, 112, 0.9)";
const BONE_COLOR  = "rgba(64, 192, 112, 0.6)";
const FADED_COLOR = "rgba(64, 192, 112, 0.25)";

// DOM refs
const videoEl   = document.getElementById("video-feed");
const canvas    = document.getElementById("skeleton-canvas");
const ctx       = canvas.getContext("2d");
const noPoseBadge    = document.getElementById("no-pose-badge");
const exerciseEl     = document.getElementById("exercise-name");
const pendingCountEl = document.getElementById("pending-count");
const pendingLabelEl = document.getElementById("pending-label");
const totalsListEl   = document.getElementById("totals-list");
const formScoreEl    = document.getElementById("form-score");
const formBarFill    = document.getElementById("form-bar-fill");
const tipsList       = document.getElementById("tips-list");
const statusDot      = document.getElementById("status-dot");

// Resize canvas to match video element size
function syncCanvasSize() {
  const rect = videoEl.getBoundingClientRect();
  canvas.width  = rect.width;
  canvas.height = rect.height;
}
window.addEventListener("resize", syncCanvasSize);
videoEl.addEventListener("load", syncCanvasSize);
syncCanvasSize();

// ---- Skeleton rendering ----
function drawSkeleton(keypoints, noPose) {
  syncCanvasSize();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (noPose || !keypoints || keypoints.length === 0) return;

  // Build name → pixel position map
  const kpMap = {};
  for (const kp of keypoints) {
    kpMap[kp.name] = {
      x: kp.x * canvas.width,
      y: kp.y * canvas.height,
      score: kp.score,
    };
  }

  // Draw bones
  ctx.lineWidth = 2.5;
  for (const [a, b] of CONNECTIONS) {
    const pa = kpMap[a], pb = kpMap[b];
    if (!pa || !pb) continue;
    const confident = pa.score >= 0.5 && pb.score >= 0.5;
    ctx.strokeStyle = confident ? BONE_COLOR : FADED_COLOR;
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }

  // Draw joints
  for (const kp of keypoints) {
    const px = kp.x * canvas.width;
    const py = kp.y * canvas.height;
    ctx.fillStyle = kp.score >= 0.5 ? JOINT_COLOR : FADED_COLOR;
    ctx.beginPath();
    ctx.arc(px, py, kp.score >= 0.5 ? 5 : 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ---- Metrics updates ----
function updateExercise(motionHint, vlmExercise) {
  // Show motion hint as real-time indicator; VLM label in parens if different
  let label = (motionHint || "—").replace(/_/g, " ");
  if (vlmExercise && vlmExercise !== "other" && vlmExercise !== motionHint) {
    label += ` <span style="color:#555;font-size:14px;font-weight:400">(VLM: ${vlmExercise.replace(/_/g, " ")})</span>`;
  }
  exerciseEl.innerHTML = label;
}

function updatePendingReps(count, motionHint) {
  pendingCountEl.textContent = count;
  pendingLabelEl.textContent = motionHint
    ? `(live · ${motionHint.replace(/_/g, " ")})`
    : "(estimating…)";
}

function updateConfirmedTotals(totals) {
  const entries = Object.entries(totals || {}).filter(([, v]) => v > 0);
  if (entries.length === 0) {
    totalsListEl.innerHTML = '<div class="totals-empty">No confirmed reps yet</div>';
    return;
  }
  totalsListEl.innerHTML = entries
    .sort((a, b) => b[1] - a[1])
    .map(([exercise, reps]) => `
      <div class="totals-row">
        <span class="totals-exercise">${exercise.replace(/_/g, " ")}</span>
        <span class="totals-reps">${reps}</span>
      </div>`)
    .join("");
}

function updateFormScore(score) {
  if (!score || score === 0) {
    formScoreEl.textContent = "—";
    formScoreEl.className = "medium";
    formBarFill.style.width = "0%";
    return;
  }
  formScoreEl.textContent = `${score}/10`;
  const pct = (score / 10) * 100;
  formBarFill.style.width = `${pct}%`;

  if (score <= 4) {
    formScoreEl.className = "low";
    formBarFill.style.background = "#e05050";
  } else if (score <= 7) {
    formScoreEl.className = "medium";
    formBarFill.style.background = "#e0a030";
  } else {
    formScoreEl.className = "high";
    formBarFill.style.background = "#40c070";
  }
}

let knownTips = new Set();

function addTip(tip) {
  if (!tip || knownTips.has(tip)) return;
  knownTips.add(tip);

  // Remove "waiting" placeholder if present
  const placeholder = tipsList.querySelector(".tip-item");
  if (placeholder && placeholder.textContent === "Waiting for analysis…") {
    placeholder.remove();
  }

  // Remove "latest" class from all existing tips
  for (const el of tipsList.querySelectorAll(".tip-item.latest")) {
    el.classList.remove("latest");
  }

  const div = document.createElement("div");
  div.className = "tip-item latest";
  div.textContent = tip;
  tipsList.insertBefore(div, tipsList.firstChild);
}

// ---- WebSocket ----
function connect() {
  const ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    statusDot.className = "connected";
    // Send periodic pings to keep connection alive
    const ping = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) ws.send("ping");
      else clearInterval(ping);
    }, 10000);
  };

  ws.onclose = () => {
    statusDot.className = "";
    // Reconnect after 2 seconds
    setTimeout(connect, 2000);
  };

  ws.onerror = () => ws.close();

  ws.onmessage = (event) => {
    let data;
    try { data = JSON.parse(event.data); } catch { return; }

    if (data.type !== "frame") return;

    drawSkeleton(data.keypoints, data.no_pose);
    noPoseBadge.style.display = data.no_pose ? "block" : "none";

    const vlm = data.vlm || {};
    updateExercise(data.motion_hint, vlm.exercise);
    updatePendingReps(data.pending_reps ?? 0, data.motion_hint);
    updateConfirmedTotals(data.confirmed_totals);

    if (vlm.form_score) updateFormScore(vlm.form_score);
    if (vlm.tip) addTip(vlm.tip);
  };
}

connect();
