const camera = document.querySelector("#camera");
const cameraEmpty = document.querySelector("#camera-empty");
const cameraToggle = document.querySelector("#camera-toggle");
const systemStatus = document.querySelector("#system-status");
const sampleSelect = document.querySelector("#sample-select");
const runSample = document.querySelector("#run-sample");
const refreshFailures = document.querySelector("#refresh-failures");
const avatarCanvas = document.querySelector("#avatar-canvas");
const avatarContext = avatarCanvas.getContext("2d");
const avatarClipCache = new Map();
let avatarClipIndex;
let activeAvatarFrames = [];
let activeAvatarFrame = 0;
let avatarPlaybackId = 0;
let avatarPlaying = false;
let avatarSpeed = 1;
let lastAvatarFrameAt = 0;
let stream;
let captureTimer;
let frameCanvas;
let frameBuffer = [];
let frameBusy = false;
const sequenceLength = 24;

function setText(selector, value) {
  document.querySelector(selector).textContent = value;
}

async function loadAvatarClip(gloss) {
  if (!avatarClipIndex) {
    const response = await fetch("/avatar/clips/index.json");
    if (!response.ok) throw new Error("Avatar clip index unavailable");
    avatarClipIndex = (await response.json()).gloss_to_clip;
  }
  const filename = avatarClipIndex[gloss] || avatarClipIndex[gloss.toLowerCase()];
  if (!filename) return null;
  if (!avatarClipCache.has(filename)) {
    const response = await fetch(`/avatar/clips/${encodeURIComponent(filename)}`);
    if (!response.ok) throw new Error(`Avatar clip unavailable for ${gloss}`);
    avatarClipCache.set(filename, await response.json());
  }
  return avatarClipCache.get(filename);
}

const poseEdges = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24],
  [23, 24], [23, 25], [25, 27], [27, 29], [29, 31], [24, 26], [26, 28],
  [28, 30], [30, 32],
];
const handEdges = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15],
  [15, 16], [0, 17], [17, 18], [18, 19], [19, 20], [5, 9], [9, 13],
  [13, 17],
];

function drawAvatarFrame(frame) {
  const bounds = avatarCanvas.getBoundingClientRect();
  const width = bounds.width;
  const height = Math.max(220, width * 0.56);
  const scale = window.devicePixelRatio || 1;
  if (avatarCanvas.width !== Math.round(width * scale) || avatarCanvas.height !== Math.round(height * scale)) {
    avatarCanvas.width = Math.round(width * scale);
    avatarCanvas.height = Math.round(height * scale);
  }
  avatarContext.setTransform(scale, 0, 0, scale, 0, 0);
  avatarContext.clearRect(0, 0, width, height);
  avatarContext.fillStyle = "#183438";
  avatarContext.fillRect(0, 0, width, height);
  if (!frame) return;

  const allPoints = [
    ...frame.left_hand.map(([x, y], index) => ({ x, y, visible: Boolean(frame.mask[index]) })),
    ...frame.right_hand.map(([x, y], index) => ({ x, y, visible: Boolean(frame.mask[21 + index]) })),
    ...frame.pose.map(([x, y], index) => ({ x, y, visible: Boolean(frame.mask[42 + index]) })),
  ];
  const visible = allPoints.filter((point) => point.visible);
  if (!visible.length) return;
  const minX = Math.min(...visible.map((point) => point.x));
  const maxX = Math.max(...visible.map((point) => point.x));
  const minY = Math.min(...visible.map((point) => point.y));
  const maxY = Math.max(...visible.map((point) => point.y));
  const pad = 28;
  const rangeX = Math.max(maxX - minX, 0.05);
  const rangeY = Math.max(maxY - minY, 0.05);
  const fit = Math.min((width - pad * 2) / rangeX, (height - pad * 2) / rangeY);
  const offsetX = (width - rangeX * fit) / 2;
  const offsetY = (height - rangeY * fit) / 2;
  const point = (collection, index) => {
    const item = collection[index];
    if (!item?.visible) return null;
    return [offsetX + (item.x - minX) * fit, offsetY + (item.y - minY) * fit];
  };
  const pose = allPoints.slice(42);
  const leftHand = allPoints.slice(0, 21);
  const rightHand = allPoints.slice(21, 42);
  const drawEdges = (points, edges, color) => {
    avatarContext.strokeStyle = color;
    avatarContext.lineWidth = 2.5;
    edges.forEach(([from, to]) => {
      const start = point(points, from);
      const end = point(points, to);
      if (!start || !end) return;
      avatarContext.beginPath();
      avatarContext.moveTo(...start);
      avatarContext.lineTo(...end);
      avatarContext.stroke();
    });
    avatarContext.fillStyle = color;
    points.forEach((_, index) => {
      const position = point(points, index);
      if (!position) return;
      avatarContext.beginPath();
      avatarContext.arc(position[0], position[1], 3.2, 0, Math.PI * 2);
      avatarContext.fill();
    });
  };
  drawEdges(pose, poseEdges, "#f4c95d");
  drawEdges(leftHand, handEdges, "#60d3c0");
  drawEdges(rightHand, handEdges, "#f18b70");
}

function startAvatarPlayback() {
  if (!activeAvatarFrames.length) return;
  avatarPlaying = true;
  const playbackId = avatarPlaybackId;
  const advance = (now) => {
    if (!avatarPlaying || playbackId !== avatarPlaybackId) return;
    if (!lastAvatarFrameAt || now - lastAvatarFrameAt >= 1000 / (12 * avatarSpeed)) {
      drawAvatarFrame(activeAvatarFrames[activeAvatarFrame]);
      activeAvatarFrame += 1;
      lastAvatarFrameAt = now;
      if (activeAvatarFrame >= activeAvatarFrames.length) {
        avatarPlaying = false;
        return;
      }
    }
    requestAnimationFrame(advance);
  };
  requestAnimationFrame(advance);
}

async function playGlosses(glosses, initiallyUnmatched = []) {
  const playbackId = ++avatarPlaybackId;
  avatarPlaying = false;
  lastAvatarFrameAt = 0;
  const clips = [];
  const unmatched = [...initiallyUnmatched];
  for (const gloss of glosses) {
    try {
      const clip = await loadAvatarClip(gloss);
      if (playbackId !== avatarPlaybackId) return;
      if (clip) clips.push(clip);
      else unmatched.push(gloss);
    } catch {
      unmatched.push(gloss);
    }
  }
  if (playbackId !== avatarPlaybackId) return;
  activeAvatarFrames = clips.flatMap((clip) => clip.frames);
  activeAvatarFrame = 0;
  if (!activeAvatarFrames.length) {
    drawAvatarFrame(null);
    setText("#avatar-note", `No clip available${unmatched.length ? ` for: ${unmatched.join(", ")}` : "."}`);
    return;
  }
  const skipped = unmatched.length ? ` No clip for: ${unmatched.join(", ")}.` : "";
  setText("#avatar-note", `Playing ${clips.map((clip) => clip.gloss).join(", ")}.${skipped}`);
  startAvatarPlayback();
}

function showResult(payload) {
  const prediction = payload.prediction;
  const route = payload.route;
  const confidence = Number(prediction.confidence);
  setText("#gloss", prediction.gloss);
  setText("#confidence", `${(confidence * 100).toFixed(1)}%`);
  setText("#timestamp", new Date(prediction.timestamp * 1000).toLocaleTimeString());
  setText("#response", route.response_text || route.message || "No response returned.");
  setText("#avatar-status", payload.avatar.status === "ready" ? "Skeleton clip ready" : "No clip available");
  playGlosses(payload.avatar.glosses || [], payload.avatar.unmatched || []);
  document.querySelector("#confidence-fill").style.width = `${confidence * 100}%`;
  const badge = document.querySelector("#confidence-badge");
  badge.textContent = route.status === "accepted" ? "Accepted" : "Clarification";
  badge.className = `confidence-badge ${route.status === "accepted" ? "is-good" : "is-warn"}`;
  loadFailureEvents();
}

async function loadSamples() {
  const response = await fetch("/api/samples");
  if (!response.ok) throw new Error("Sample list unavailable");
  const data = await response.json();
  data.samples.slice(0, 12).forEach((sample) => {
    const option = document.createElement("option");
    option.value = sample.sample_id;
    option.textContent = `${sample.label} · ${sample.sample_id}`;
    sampleSelect.append(option);
  });
}

async function loadFailureEvents() {
  const body = document.querySelector("#failure-events");
  const note = document.querySelector("#failure-note");
  try {
    const response = await fetch("/api/failures");
    if (!response.ok) throw new Error("Failure log unavailable");
    const data = await response.json();
    body.replaceChildren();
    data.events.slice(-100).reverse().forEach((event) => {
      const row = document.createElement("tr");
      [
        event.reason,
        event.gloss,
        `${(Number(event.confidence) * 100).toFixed(1)}%`,
        new Date(Number(event.timestamp) * 1000).toLocaleString(),
      ].forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      });
      body.append(row);
    });
    note.textContent = data.events.length ? `Showing the latest ${data.events.length} event(s).` : "No clarification events logged yet.";
  } catch (error) {
    note.textContent = error.message;
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    systemStatus.textContent = `${data.labels} labels / ready`;
  } catch {
    systemStatus.textContent = "Runtime unavailable";
  }
}

async function captureFrame() {
  if (!stream || frameBusy || !camera.videoWidth) return;
  frameBusy = true;
  const context = frameCanvas.getContext("2d");
  context.drawImage(camera, 0, 0, frameCanvas.width, frameCanvas.height);
  try {
    const frameResponse = await fetch("/api/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_base64: frameCanvas.toDataURL("image/jpeg", 0.72),
        timestamp_ms: Date.now(),
      }),
    });
    if (!frameResponse.ok) throw new Error("Landmark extraction unavailable");
    const frame = await frameResponse.json();
    if (!frame.has_landmarks) {
      setText("#camera-note", "No landmarks detected; move into view.");
      return;
    }
    frameBuffer.push(frame);
    setText("#camera-note", `Landmark frames: ${frameBuffer.length}/${sequenceLength}`);
    if (frameBuffer.length >= sequenceLength) {
      const sequence = frameBuffer.splice(0, sequenceLength);
      const inferenceResponse = await fetch("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          features: sequence.map((frame) => frame.features),
          mask: sequence.map((frame) => frame.mask),
          timestamp: Date.now() / 1000,
        }),
      });
      if (!inferenceResponse.ok) throw new Error("Camera inference failed");
      showResult(await inferenceResponse.json());
    }
  } catch (error) {
    setText("#camera-note", error.message);
    clearInterval(captureTimer);
  } finally {
    frameBusy = false;
  }
}

runSample.addEventListener("click", async () => {
  runSample.disabled = true;
  runSample.textContent = "Running…";
  try {
    const response = await fetch(`/api/infer?sample_id=${encodeURIComponent(sampleSelect.value)}`);
    if (!response.ok) throw new Error("Inference request failed");
    showResult(await response.json());
  } catch (error) {
    setText("#response", error.message);
  } finally {
    runSample.disabled = false;
    runSample.textContent = "Run inference";
  }
});

refreshFailures.addEventListener("click", loadFailureEvents);
document.querySelector("#avatar-play").addEventListener("click", startAvatarPlayback);
document.querySelector("#avatar-pause").addEventListener("click", () => {
  avatarPlaying = false;
});
document.querySelector("#avatar-speed").addEventListener("change", (event) => {
  avatarSpeed = Number(event.target.value);
});
window.addEventListener("resize", () => {
  if (activeAvatarFrames.length) drawAvatarFrame(activeAvatarFrames[activeAvatarFrame]);
});

cameraToggle.addEventListener("click", async () => {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    clearInterval(captureTimer);
    stream = undefined;
    camera.srcObject = null;
    frameBuffer = [];
    cameraEmpty.hidden = false;
    cameraToggle.textContent = "Enable camera";
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    camera.srcObject = stream;
    frameCanvas = document.createElement("canvas");
    frameCanvas.width = 640;
    frameCanvas.height = 360;
    frameBuffer = [];
    cameraEmpty.hidden = true;
    cameraToggle.textContent = "Stop camera";
    captureTimer = setInterval(captureFrame, 180);
  } catch {
    setText("#camera-note", "Camera permission was not granted.");
  }
});

Promise.all([checkHealth(), loadSamples(), loadFailureEvents()]);
