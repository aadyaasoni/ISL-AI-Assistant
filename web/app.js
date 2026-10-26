const camera = document.querySelector("#camera");
const cameraFrame = document.querySelector("#camera-frame");
const cameraEmpty = document.querySelector("#camera-empty");
const cameraEnable = document.querySelector("#camera-enable");
const cameraStop = document.querySelector("#camera-stop");
const cameraNote = document.querySelector("#camera-note");
const systemStatus = document.querySelector("#system-status");
const sampleSelect = document.querySelector("#sample-select");
const runSample = document.querySelector("#run-sample");
const avatarCanvas = document.querySelector("#avatar-canvas");
const avatarContext = avatarCanvas.getContext("2d");
const avatarClipCache = new Map();
const sequenceLength = 24;
const poseUpperBodyIndices = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24];
const poseEdges = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [11, 23],
  [12, 24],
  [23, 24],
];
const handEdges = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

let avatarClipIndex;
let activeAvatarFrames = [];
let activeFrameGlosses = [];
let activeAvatarFrame = 0;
let avatarPlaybackId = 0;
let avatarPlaying = false;
let avatarSpeed = 1;
let playbackPosition = 0;
let previousAnimationTime = 0;
let stream;
let captureTimer;
let frameCanvas;
let frameBuffer = [];
let frameBusy = false;
let sampleRows = [];

function setText(selector, value) {
  document.querySelector(selector).textContent = value;
}

function friendlyWord(label) {
  return label.replace(/^\s*\d+\.\s*/, "");
}

function cameraErrorMessage(error) {
  switch (error.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Camera permission was denied. Allow camera access for this site in Chrome, then try again.";
    case "NotFoundError":
      return "No camera was found. Connect or enable a camera, then try again.";
    case "NotReadableError":
      return "The camera is busy or unavailable. Close other apps or tabs using it, then try again.";
    case "OverconstrainedError":
      return "The camera cannot provide the requested video mode. Check your camera settings and try again.";
    default:
      return `Could not start the camera${error.message ? `: ${error.message}` : "."}`;
  }
}

function waitForVideoMetadata(video) {
  if (video.readyState >= HTMLMediaElement.HAVE_METADATA) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timeoutId = window.setTimeout(() => {
      cleanup();
      reject(new Error("The camera connected, but no video metadata arrived. Try restarting the camera."));
    }, 10000);
    const cleanup = () => {
      window.clearTimeout(timeoutId);
      video.removeEventListener("loadedmetadata", loaded);
      video.removeEventListener("error", failed);
    };
    const loaded = () => {
      cleanup();
      resolve();
    };
    const failed = () => {
      cleanup();
      reject(new Error("The browser could not read video from this camera."));
    };
    video.addEventListener("loadedmetadata", loaded, { once: true });
    video.addEventListener("error", failed, { once: true });
  });
}

function updateCameraState(enabled) {
  cameraFrame.classList.toggle("is-live", enabled);
  cameraEmpty.hidden = enabled;
  cameraEnable.hidden = enabled;
  cameraStop.hidden = !enabled;
}

function stopCamera(message = "Camera is off. Enable it when you're ready to sign.") {
  if (captureTimer) window.clearInterval(captureTimer);
  captureTimer = undefined;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = undefined;
  camera.pause();
  camera.srcObject = null;
  frameBuffer = [];
  frameBusy = false;
  updateCameraState(false);
  cameraNote.textContent = message;
}

async function enableCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    cameraNote.textContent = "Camera access is unavailable in this browser. Open the app over HTTPS in a current browser.";
    return;
  }
  cameraEnable.disabled = true;
  cameraNote.textContent = "Connecting to your camera…";
  try {
    const nextStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
    stream = nextStream;
    camera.muted = true;
    camera.autoplay = true;
    camera.playsInline = true;
    const metadataReady = waitForVideoMetadata(camera);
    camera.srcObject = stream;
    await metadataReady;
    if (!camera.videoWidth || !camera.videoHeight) {
      throw new Error("The camera connected but returned an empty video frame. Try another camera.");
    }
    await camera.play();
    frameCanvas = document.createElement("canvas");
    frameCanvas.width = 640;
    frameCanvas.height = 360;
    frameBuffer = [];
    updateCameraState(true);
    cameraNote.textContent = `Live mirrored preview · ${camera.videoWidth} × ${camera.videoHeight}. Sign when ready.`;
    captureTimer = window.setInterval(captureFrame, 180);
  } catch (error) {
    stopCamera(cameraErrorMessage(error));
  } finally {
    cameraEnable.disabled = false;
  }
}

async function loadAvatarClip(gloss) {
  if (!avatarClipIndex) {
    const response = await fetch("/avatar/clips/index.json");
    if (!response.ok) throw new Error("Avatar clips are unavailable.");
    avatarClipIndex = (await response.json()).gloss_to_clip;
  }
  const filename = avatarClipIndex[gloss] || avatarClipIndex[gloss.toLowerCase()];
  if (!filename) return null;
  if (!avatarClipCache.has(filename)) {
    const response = await fetch(`/avatar/clips/${encodeURIComponent(filename)}`);
    if (!response.ok) throw new Error(`Could not load the ${friendlyWord(gloss)} skeleton.`);
    avatarClipCache.set(filename, await response.json());
  }
  return avatarClipCache.get(filename);
}

function visiblePoint(frame, group, index) {
  const maskIndex = group === "left_hand"
    ? index
    : group === "right_hand"
      ? 21 + index
      : 42 + index;
  if (!frame.mask[maskIndex]) return null;
  const coordinates = frame[group][index];
  return { x: coordinates[0], y: coordinates[1] };
}

function upperBodyPoints(frame) {
  const points = [];
  ["left_hand", "right_hand"].forEach((group) => {
    for (let index = 0; index < 21; index += 1) {
      const item = visiblePoint(frame, group, index);
      if (item) points.push(item);
    }
  });
  poseUpperBodyIndices.forEach((index) => {
    const item = visiblePoint(frame, "pose", index);
    if (item) points.push(item);
  });
  return points;
}

function calculateClipBounds(frames) {
  const points = frames.flatMap(upperBodyPoints);
  if (!points.length) return null;
  return {
    minX: Math.min(...points.map((point) => point.x)),
    maxX: Math.max(...points.map((point) => point.x)),
    minY: Math.min(...points.map((point) => point.y)),
    maxY: Math.max(...points.map((point) => point.y)),
  };
}

function interpolateFrame(first, second, amount) {
  if (!second || amount <= 0) return first;
  const lerp = (a, b) => a + (b - a) * amount;
  const interpolatePoints = (pointsA, pointsB) => pointsA.map((point, index) =>
    point.map((value, axis) => lerp(value, pointsB[index][axis])),
  );
  return {
    left_hand: interpolatePoints(first.left_hand, second.left_hand),
    right_hand: interpolatePoints(first.right_hand, second.right_hand),
    pose: interpolatePoints(first.pose, second.pose),
    mask: first.mask.map((value, index) => value || second.mask[index]),
  };
}

function drawAvatarFrame(frame, bounds) {
  const rect = avatarCanvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  if (!width || !height) return;
  const pixelRatio = window.devicePixelRatio || 1;
  const pixelWidth = Math.round(width * pixelRatio);
  const pixelHeight = Math.round(height * pixelRatio);
  if (avatarCanvas.width !== pixelWidth || avatarCanvas.height !== pixelHeight) {
    avatarCanvas.width = pixelWidth;
    avatarCanvas.height = pixelHeight;
  }
  avatarContext.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  avatarContext.clearRect(0, 0, width, height);
  avatarContext.fillStyle = "#111821";
  avatarContext.fillRect(0, 0, width, height);
  if (!frame || !bounds) return;

  const widthRange = Math.max(bounds.maxX - bounds.minX, 0.05);
  const heightRange = Math.max(bounds.maxY - bounds.minY, 0.05);
  const paddingX = Math.max(34, width * 0.11);
  const paddingY = Math.max(32, height * 0.1);
  const fit = Math.min(
    (width - paddingX * 2) / widthRange,
    (height - paddingY * 2) / heightRange,
  );
  const originX = (width - widthRange * fit) / 2;
  const originY = (height - heightRange * fit) / 2;
  const mapPoint = (item) => item && [
    originX + (item.x - bounds.minX) * fit,
    originY + (item.y - bounds.minY) * fit,
  ];
  const pose = Array.from({ length: 33 }, (_, index) => visiblePoint(frame, "pose", index));
  const leftHand = Array.from({ length: 21 }, (_, index) => visiblePoint(frame, "left_hand", index));
  const rightHand = Array.from({ length: 21 }, (_, index) => visiblePoint(frame, "right_hand", index));

  const drawEdges = (points, edges, color, lineWidth, jointRadius, pointIndices = points.map((_, index) => index)) => {
    avatarContext.strokeStyle = color;
    avatarContext.lineWidth = lineWidth;
    avatarContext.lineCap = "round";
    avatarContext.lineJoin = "round";
    edges.forEach(([startIndex, endIndex]) => {
      const start = mapPoint(points[startIndex]);
      const end = mapPoint(points[endIndex]);
      if (!start || !end) return;
      avatarContext.beginPath();
      avatarContext.moveTo(start[0], start[1]);
      avatarContext.lineTo(end[0], end[1]);
      avatarContext.stroke();
    });
    avatarContext.fillStyle = color;
    pointIndices.forEach((index) => {
      const item = points[index];
      const position = mapPoint(item);
      if (!position) return;
      avatarContext.beginPath();
      avatarContext.arc(position[0], position[1], jointRadius, 0, Math.PI * 2);
      avatarContext.fill();
    });
  };

  drawEdges(
    pose,
    poseEdges,
    "#edf2f7",
    Math.max(4, width * 0.008),
    Math.max(4, width * 0.009),
    poseUpperBodyIndices,
  );
  drawEdges(leftHand, handEdges, "#55d6be", Math.max(3.5, width * 0.007), Math.max(3, width * 0.007));
  drawEdges(rightHand, handEdges, "#55d6be", Math.max(3.5, width * 0.007), Math.max(3, width * 0.007));

  const leftEar = mapPoint(pose[7]);
  const rightEar = mapPoint(pose[8]);
  const nose = mapPoint(pose[0]);
  if (leftEar && rightEar && nose) {
    const centerX = (leftEar[0] + rightEar[0]) / 2;
    const centerY = (leftEar[1] + rightEar[1]) / 2;
    const radius = Math.max(Math.hypot(rightEar[0] - leftEar[0], rightEar[1] - leftEar[1]) * 1.15, 15);
    avatarContext.strokeStyle = "#edf2f7";
    avatarContext.lineWidth = Math.max(4, width * 0.008);
    avatarContext.beginPath();
    avatarContext.ellipse(centerX, centerY, radius, radius * 1.18, 0, 0, Math.PI * 2);
    avatarContext.stroke();
    avatarContext.fillStyle = "#edf2f7";
    avatarContext.beginPath();
    avatarContext.arc(nose[0], nose[1], 4, 0, Math.PI * 2);
    avatarContext.fill();
  }
}

function startAvatarPlayback() {
  if (!activeAvatarFrames.length) return;
  if (playbackPosition >= activeAvatarFrames.length - 1) playbackPosition = 0;
  avatarPlaying = true;
  previousAnimationTime = 0;
  const playbackId = avatarPlaybackId;
  const bounds = calculateClipBounds(activeAvatarFrames);
  const animate = (now) => {
    if (!avatarPlaying || playbackId !== avatarPlaybackId) return;
    if (previousAnimationTime) playbackPosition += ((now - previousAnimationTime) / 1000) * 15 * avatarSpeed;
    previousAnimationTime = now;
    if (playbackPosition >= activeAvatarFrames.length - 1) {
      playbackPosition = activeAvatarFrames.length - 1;
      avatarPlaying = false;
    }
    activeAvatarFrame = Math.floor(playbackPosition);
    const nextIndex = Math.min(activeAvatarFrame + 1, activeAvatarFrames.length - 1);
    const frame = interpolateFrame(
      activeAvatarFrames[activeAvatarFrame],
      activeAvatarFrames[nextIndex],
      playbackPosition - activeAvatarFrame,
    );
    drawAvatarFrame(frame, bounds);
    setText("#avatar-caption", friendlyWord(activeFrameGlosses[activeAvatarFrame] || ""));
    if (avatarPlaying) window.requestAnimationFrame(animate);
  };
  window.requestAnimationFrame(animate);
}

async function playGlosses(glosses, initiallyUnmatched = []) {
  const playbackId = ++avatarPlaybackId;
  avatarPlaying = false;
  playbackPosition = 0;
  const clips = [];
  const unmatched = [...initiallyUnmatched];
  const clipErrors = [];
  for (const gloss of glosses) {
    try {
      const clip = await loadAvatarClip(gloss);
      if (playbackId !== avatarPlaybackId) return;
      if (clip) clips.push(clip);
      else unmatched.push(gloss);
    } catch (error) {
      unmatched.push(gloss);
      clipErrors.push(error.message);
    }
  }
  if (playbackId !== avatarPlaybackId) return;
  activeAvatarFrames = clips.flatMap((clip) => clip.frames);
  activeFrameGlosses = clips.flatMap((clip) => clip.frames.map(() => clip.gloss));
  const note = document.querySelector("#avatar-note");
  if (!activeAvatarFrames.length) {
    drawAvatarFrame(null, null);
    setText("#avatar-caption", "No clip available");
    const missing = unmatched.length ? ` for: ${unmatched.map(friendlyWord).join(", ")}.` : ".";
    const loadErrors = clipErrors.length ? ` ${clipErrors.join(" ")}` : "";
    note.textContent = `No skeleton available${missing}${loadErrors}`;
    return;
  }
  const skipped = unmatched.length ? ` Skipped: ${unmatched.map(friendlyWord).join(", ")}.` : "";
  const loadErrors = clipErrors.length ? ` ${clipErrors.join(" ")}` : "";
  note.textContent = `Playing ${clips.map((clip) => friendlyWord(clip.gloss)).join(", ")}.${skipped}${loadErrors}`;
  startAvatarPlayback();
}

function showResult(payload) {
  const prediction = payload.prediction;
  const route = payload.route;
  const confidence = Number(prediction.confidence);
  setText("#gloss", friendlyWord(prediction.gloss));
  setText("#confidence", `${(confidence * 100).toFixed(1)}%`);
  setText("#response", route.response_text || route.message || "No response returned.");
  document.querySelector("#confidence-fill").style.width = `${confidence * 100}%`;
  const badge = document.querySelector("#confidence-badge");
  badge.textContent = route.status === "accepted" ? "Understood" : "Please repeat";
  badge.className = `confidence-badge ${route.status === "accepted" ? "is-good" : "is-warn"}`;
  playGlosses(payload.avatar.glosses || [], payload.avatar.unmatched || []);
  loadFailureEvents();
}

async function loadSamples() {
  const response = await fetch("/api/samples");
  if (!response.ok) throw new Error("Recorded samples are unavailable.");
  const data = await response.json();
  sampleRows = data.samples;
  sampleSelect.replaceChildren();
  sampleRows.forEach((sample, index) => {
    const option = document.createElement("option");
    option.value = sample.sample_id;
    option.textContent = `Sample ${index + 1}: ${friendlyWord(sample.label)}`;
    sampleSelect.append(option);
  });
}

async function renderSupportedSigns() {
  const response = await fetch("/avatar/clips/index.json");
  if (!response.ok) throw new Error("Supported sign list is unavailable.");
  avatarClipIndex = (await response.json()).gloss_to_clip;
  const container = document.querySelector("#supported-signs");
  Object.keys(avatarClipIndex).forEach((gloss) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sign-chip";
    button.textContent = friendlyWord(gloss);
    button.setAttribute("aria-label", `Play ${friendlyWord(gloss)} skeleton`);
    button.addEventListener("click", () => playGlosses([gloss]));
    container.append(button);
  });
}

async function loadFailureEvents() {
  const body = document.querySelector("#failure-events");
  const note = document.querySelector("#failure-note");
  try {
    const response = await fetch("/api/failures");
    if (!response.ok) throw new Error("Failure log unavailable.");
    const data = await response.json();
    body.replaceChildren();
    data.events.slice(-100).reverse().forEach((event) => {
      const row = document.createElement("tr");
      [event.reason, event.gloss, `${(Number(event.confidence) * 100).toFixed(1)}%`]
        .forEach((value) => {
          const cell = document.createElement("td");
          cell.textContent = value;
          row.append(cell);
        });
      body.append(row);
    });
    note.textContent = data.events.length
      ? `Showing the latest ${data.events.length} event(s).`
      : "No clarification events logged yet.";
  } catch (error) {
    note.textContent = error.message;
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error();
    const data = await response.json();
    systemStatus.textContent = `${data.labels} signs ready`;
  } catch {
    systemStatus.textContent = "Service unavailable";
  }
}

async function captureFrame() {
  if (!stream || frameBusy || !camera.videoWidth || !camera.videoHeight) return;
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
    if (!stream) return;
    if (!frameResponse.ok) throw new Error(`Landmark extraction failed (${frameResponse.status}).`);
    const frame = await frameResponse.json();
    if (!frame.has_landmarks) {
      cameraNote.textContent = "Camera is live. Move your hands and upper body into view.";
      return;
    }
    frameBuffer.push(frame);
    cameraNote.textContent = `Live · collecting sign frames ${frameBuffer.length}/${sequenceLength}`;
    if (frameBuffer.length >= sequenceLength) {
      const sequence = frameBuffer.splice(0, sequenceLength);
      const inferenceResponse = await fetch("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          features: sequence.map((item) => item.features),
          mask: sequence.map((item) => item.mask),
          timestamp: Date.now() / 1000,
        }),
      });
      if (!stream) return;
      if (!inferenceResponse.ok) throw new Error(`Sign recognition failed (${inferenceResponse.status}).`);
      showResult(await inferenceResponse.json());
      cameraNote.textContent = "Live preview · show another sign when ready.";
    }
  } catch (error) {
    cameraNote.textContent = `${error.message} The preview is still on; you can retry or stop the camera.`;
    window.clearInterval(captureTimer);
    captureTimer = undefined;
  } finally {
    frameBusy = false;
  }
}

cameraEnable.addEventListener("click", enableCamera);
cameraStop.addEventListener("click", () => stopCamera());
runSample.addEventListener("click", async () => {
  runSample.disabled = true;
  try {
    const response = await fetch(`/api/infer?sample_id=${encodeURIComponent(sampleSelect.value)}`);
    if (!response.ok) throw new Error(`Recorded sample failed (${response.status}).`);
    showResult(await response.json());
  } catch (error) {
    setText("#response", error.message);
  } finally {
    runSample.disabled = false;
  }
});

document.querySelector("#text-to-sign").addEventListener("click", async () => {
  const button = document.querySelector("#text-to-sign");
  const text = document.querySelector("#hearing-text").value;
  button.disabled = true;
  try {
    const response = await fetch("/api/text_to_sign", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) throw new Error(`Text-to-sign failed (${response.status}).`);
    const result = await response.json();
    playGlosses(result.glosses, result.unmatched);
    setText(
      "#text-sign-unmatched",
      result.unmatched.length
        ? `Not in the supported signs: ${result.unmatched.join(", ")}`
        : result.glosses.length
          ? `Playing ${result.glosses.length} supported sign(s).`
          : "Enter a supported word to play its sign.",
    );
  } catch (error) {
    setText("#text-sign-unmatched", error.message);
  } finally {
    button.disabled = false;
  }
});

document.querySelector("#avatar-play").addEventListener("click", startAvatarPlayback);
document.querySelector("#avatar-pause").addEventListener("click", () => {
  avatarPlaying = false;
  previousAnimationTime = 0;
});
document.querySelector("#avatar-speed").addEventListener("change", (event) => {
  avatarSpeed = Number(event.target.value);
});
document.querySelector("#refresh-failures").addEventListener("click", loadFailureEvents);
window.addEventListener("resize", () => {
  if (activeAvatarFrames.length) {
    const frame = activeAvatarFrames[activeAvatarFrame];
    drawAvatarFrame(frame, calculateClipBounds(activeAvatarFrames));
  }
});

Promise.all([checkHealth(), loadSamples(), renderSupportedSigns(), loadFailureEvents()])
  .catch((error) => {
    setText("#system-status", error.message);
  });
