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
const repCountEl     = document.getElementById("rep-count");
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
function updateExercise(name) {
  exerciseEl.textContent = name.replace(/_/g, " ") || "—";
}

function updateRepCount(count) {
  repCountEl.textContent = count;
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
    updateRepCount(data.rep_count ?? 0);
    updateExercise(data.exercise ?? "unknown");

    if (data.vlm) {
      updateFormScore(data.vlm.form_score);
      addTip(data.vlm.tip);
    }
  };
}

connect();
