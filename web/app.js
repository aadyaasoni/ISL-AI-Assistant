const camera = document.querySelector("#camera");
const cameraFrame = document.querySelector("#camera-frame");
const cameraEmpty = document.querySelector("#camera-empty");
const landmarkOverlay = document.querySelector("#landmark-overlay");
const landmarkContext = landmarkOverlay.getContext("2d");
const cameraEnable = document.querySelector("#camera-enable");
const cameraStop = document.querySelector("#camera-stop");
const cameraNote = document.querySelector("#camera-note");
const systemStatus = document.querySelector("#system-status");
const sampleSelect = document.querySelector("#sample-select");
const runSample = document.querySelector("#run-sample");
const avatarCanvas = document.querySelector("#avatar-canvas");
const avatarContext = avatarCanvas.getContext("2d");
const avatarClipCache = new Map();
const sequenceLength = 56;
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
let handLandmarker;
let poseLandmarker;
let detectionFrameId;
let lastDetectionAt = 0;
let previousHandCenter;
let preRollFrames = [];
let segmentFrames = [];
let segmentStartedAt = 0;
let lastHandMotionAt = 0;
let segmentProcessing = false;
let pendingClarification;
let recognizedGlosses = [];
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
  if (detectionFrameId) window.cancelAnimationFrame(detectionFrameId);
  detectionFrameId = undefined;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = undefined;
  camera.pause();
  camera.srcObject = null;
  segmentFrames = [];
  preRollFrames = [];
  previousHandCenter = undefined;
  segmentProcessing = false;
  landmarkContext.clearRect(0, 0, landmarkOverlay.width, landmarkOverlay.height);
  updateCameraState(false);
  cameraNote.textContent = message;
  setText("#recognition-status", "Show your hands to begin.");
}

const visionVersion = "0.10.22-rc.20250304";
const visionBaseUrl = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${visionVersion}`;

async function initializeLandmarkers() {
  if (handLandmarker && poseLandmarker) return;
  const { FilesetResolver, HandLandmarker, PoseLandmarker } = await import(
    `${visionBaseUrl}/vision_bundle.mjs`
  );
  const fileset = await FilesetResolver.forVisionTasks(`${visionBaseUrl}/wasm`);
  const handTask = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: "/models/mediapipe/hand_landmarker.task",
      delegate: "CPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });
  try {
    poseLandmarker = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: "/models/mediapipe/pose_landmarker_lite.task",
        delegate: "CPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
    });
    handLandmarker = handTask;
  } catch (error) {
    handTask.close();
    throw error;
  }
}

async function enableCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    cameraNote.textContent = "Camera access is unavailable in this browser. Open the app over HTTPS in a current browser.";
    setText("#recognition-status", "Live recognition unavailable; use a recorded sample.");
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
    updateCameraState(true);
    landmarkOverlay.width = camera.videoWidth;
    landmarkOverlay.height = camera.videoHeight;
    cameraNote.textContent = `Live mirrored preview · ${camera.videoWidth} × ${camera.videoHeight}.`;
    setText("#recognition-status", "Loading live recognition…");
    try {
      await initializeLandmarkers();
      cameraNote.textContent = "Live recognition ready. Show your hands and sign one word.";
      setText("#recognition-status", "Show your hands.");
      lastDetectionAt = 0;
      detectionFrameId = window.requestAnimationFrame(detectLiveFrame);
    } catch (error) {
      cameraNote.textContent = `Live recognition could not load (${error.message}). Camera preview is available; use a recorded sample instead.`;
      setText("#recognition-status", "Live recognition unavailable; recorded samples still work.");
      handLandmarker?.close();
      poseLandmarker?.close();
      handLandmarker = undefined;
      poseLandmarker = undefined;
    }
  } catch (error) {
    stopCamera(cameraErrorMessage(error));
    setText("#recognition-status", "Live recognition unavailable; use a recorded sample.");
  } finally {
    cameraEnable.disabled = false;
  }
}

function jsonFrame(handResult, poseResult) {
  const hands = (handResult.landmarks || []).map((landmarks, index) => ({
    handedness: handResult.handedness?.[index]?.[0]?.categoryName || "",
    landmarks: landmarks.map(({ x, y, z }) => ({ x, y, z })),
  }));
  const pose = poseResult.landmarks?.[0] || [];
  return {
    hands,
    pose_landmarks: pose.map(({ x, y, z, visibility }) => ({
      x,
      y,
      z,
      visibility: visibility ?? 0,
    })),
  };
}

function drawLiveOverlay(frame) {
  const width = landmarkOverlay.width;
  const height = landmarkOverlay.height;
  landmarkContext.clearRect(0, 0, width, height);
  const point = (group, index) => {
    if (!group[index] || group[index].visibility === 0) return null;
    return [group[index].x * width, group[index].y * height];
  };
  const drawConnections = (points, connections, color, radius, pointIndices = points.map((_, index) => index)) => {
    landmarkContext.strokeStyle = color;
    landmarkContext.fillStyle = color;
    landmarkContext.lineWidth = Math.max(3, width * 0.004);
    landmarkContext.lineCap = "round";
    landmarkContext.lineJoin = "round";
    connections.forEach(([from, to]) => {
      const start = point(points, from);
      const end = point(points, to);
      if (!start || !end) return;
      landmarkContext.beginPath();
      landmarkContext.moveTo(...start);
      landmarkContext.lineTo(...end);
      landmarkContext.stroke();
    });
    pointIndices.forEach((index) => {
      const location = point(points, index);
      if (!location) return;
      landmarkContext.beginPath();
      landmarkContext.arc(location[0], location[1], radius, 0, Math.PI * 2);
      landmarkContext.fill();
    });
  };
  const pose = frame.pose_landmarks || [];
  drawConnections(
    pose,
    poseEdges.filter(([a, b]) => a <= 24 && b <= 24),
    "#eef4f7",
    Math.max(3, width * 0.004),
    poseUpperBodyIndices,
  );
  const leftEar = point(pose, 7);
  const rightEar = point(pose, 8);
  const nose = point(pose, 0);
  if (leftEar && rightEar && nose) {
    const centerX = (leftEar[0] + rightEar[0]) / 2;
    const centerY = (leftEar[1] + rightEar[1]) / 2;
    const radius = Math.max(Math.hypot(rightEar[0] - leftEar[0], rightEar[1] - leftEar[1]) * 1.15, width * 0.025);
    landmarkContext.strokeStyle = "#eef4f7";
    landmarkContext.lineWidth = Math.max(3, width * 0.004);
    landmarkContext.beginPath();
    landmarkContext.ellipse(centerX, centerY, radius, radius * 1.18, 0, 0, Math.PI * 2);
    landmarkContext.stroke();
  }
  frame.hands.forEach((hand) => {
    drawConnections(hand.landmarks, handEdges, "#50d5be", Math.max(2, width * 0.003));
  });
}

function handCenter(frame) {
  const points = frame.hands.flatMap((hand) => hand.landmarks.slice(0, 21));
  if (!points.length) return null;
  return {
    x: points.reduce((sum, item) => sum + item.x, 0) / points.length,
    y: points.reduce((sum, item) => sum + item.y, 0) / points.length,
  };
}

function resampleFrames(frames, targetLength) {
  return Array.from({ length: targetLength }, (_, index) => {
    const sourceIndex = Math.round((index * (frames.length - 1)) / (targetLength - 1));
    return frames[sourceIndex];
  });
}

function processSegment() {
  const duration = performance.now() - segmentStartedAt;
  const frames = segmentFrames;
  segmentFrames = [];
  segmentStartedAt = 0;
  if (frames.length < 8 || duration < 350) {
    setText("#recognition-status", "That sign was too brief. Try again with a clear movement.");
    return;
  }
  segmentProcessing = true;
  setText("#recognition-status", "Signing…");
  inferSegment(resampleFrames(frames, sequenceLength))
    .catch((error) => {
      cameraNote.textContent = `${error.message} Recorded samples are still available.`;
      setText("#recognition-status", "Could not recognize that sign; try again or use a sample.");
    })
    .finally(() => {
      segmentProcessing = false;
      if (stream) setText("#recognition-status", "Show your hands.");
    });
}

async function inferSegment(frames) {
  const response = await fetch("/api/infer_landmarks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frames, context_glosses: recognizedGlosses.slice(-14) }),
  });
  if (!response.ok) {
    const result = await response.json().catch(() => ({}));
    throw new Error(result.error || `Sign recognition failed (${response.status}).`);
  }
  const result = await response.json();
  if (result.decision === "clarify") {
    showClarification(result);
    return;
  }
  acceptRecognizedGloss(result.gloss);
  showLiveResult(result);
}

function detectLiveFrame(now) {
  if (!stream || !handLandmarker || !poseLandmarker) return;
  detectionFrameId = window.requestAnimationFrame(detectLiveFrame);
  if (now - lastDetectionAt < 66 || camera.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  lastDetectionAt = now;
  try {
    const hands = handLandmarker.detectForVideo(camera, now);
    const pose = poseLandmarker.detectForVideo(camera, now);
    const frame = jsonFrame(hands, pose);
    drawLiveOverlay(frame);
    const center = handCenter(frame);
    const moving = Boolean(
      center
      && previousHandCenter
      && Math.hypot(center.x - previousHandCenter.x, center.y - previousHandCenter.y) > 0.012,
    );
    if (center) previousHandCenter = center;
    else previousHandCenter = undefined;

    if (!segmentFrames.length && !segmentProcessing && center && moving) {
      segmentFrames = [...preRollFrames, frame];
      segmentStartedAt = now;
      lastHandMotionAt = now;
      setText("#recognition-status", "Signing…");
    } else if (segmentFrames.length && !segmentProcessing) {
      segmentFrames.push(frame);
      if (moving) lastHandMotionAt = now;
      if (now - lastHandMotionAt >= 600 || segmentFrames.length >= 120) processSegment();
    } else if (!segmentProcessing && !center) {
      setText("#recognition-status", "Show your hands.");
    } else if (!segmentProcessing && center && !segmentFrames.length) {
      setText("#recognition-status", "Hands visible. Move to sign.");
    }
    preRollFrames.push(frame);
    if (preRollFrames.length > 4) preRollFrames.shift();
  } catch (error) {
    window.cancelAnimationFrame(detectionFrameId);
    detectionFrameId = undefined;
    cameraNote.textContent = `Live detection stopped (${error.message}). Use a recorded sample instead.`;
    setText("#recognition-status", "Live recognition unavailable; recorded samples still work.");
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
  if (route.status === "accepted") acceptRecognizedGloss(prediction.gloss);
  playGlosses(payload.avatar.glosses || [], payload.avatar.unmatched || []);
  setText("#recognition-status", route.status === "accepted" ? "Recognised from a recorded sample." : route.message);
  loadFailureEvents();
}

function renderRecognizedWords() {
  const container = document.querySelector("#recognized-words");
  container.replaceChildren();
  recognizedGlosses.forEach((gloss) => {
    const chip = document.createElement("span");
    chip.className = "recognized-word";
    chip.textContent = friendlyWord(gloss);
    container.append(chip);
  });
}

function acceptRecognizedGloss(gloss) {
  recognizedGlosses.push(gloss);
  if (recognizedGlosses.length > 15) recognizedGlosses.shift();
  renderRecognizedWords();
}

function showLiveResult(result) {
  const confidence = Number(result.confidence);
  setText("#gloss", friendlyWord(result.gloss));
  setText("#confidence", `${(confidence * 100).toFixed(1)}%`);
  setText("#response", result.sentence || result.route.response_text);
  document.querySelector("#confidence-fill").style.width = `${confidence * 100}%`;
  const badge = document.querySelector("#confidence-badge");
  badge.textContent = "Understood";
  badge.className = "confidence-badge is-good";
  document.querySelector("#clarification-options").hidden = true;
  playGlosses(result.avatar.glosses || [], result.avatar.unmatched || []);
  setText("#recognition-status", `Recognised: ${friendlyWord(result.gloss)} ${(confidence * 100).toFixed(0)}%`);
  loadFailureEvents();
}

function showClarification(result) {
  pendingClarification = { result };
  setText("#gloss", friendlyWord(result.gloss));
  setText("#confidence", `${(Number(result.confidence) * 100).toFixed(1)}%`);
  setText("#response", result.route.message);
  const badge = document.querySelector("#confidence-badge");
  badge.textContent = "Please choose";
  badge.className = "confidence-badge is-warn";
  setText(
    "#recognition-status",
    `Uncertain: ${friendlyWord(result.candidates[0].gloss)} or ${friendlyWord(result.candidates[1].gloss)}?`,
  );
  setText(
    "#clarification-prompt",
    `Did you mean ${friendlyWord(result.candidates[0].gloss)} or ${friendlyWord(result.candidates[1].gloss)}?`,
  );
  const buttons = document.querySelector("#clarification-buttons");
  buttons.replaceChildren();
  result.candidates.slice(0, 2).forEach((candidate) => {
    const button = document.createElement("button");
    button.className = "button button-secondary";
    button.type = "button";
    button.textContent = `Use ${friendlyWord(candidate.gloss)}`;
    button.addEventListener("click", () => chooseClarification(candidate));
    buttons.append(button);
  });
  document.querySelector("#clarification-options").hidden = false;
}

async function chooseClarification(candidate) {
  if (!pendingClarification) return;
  const { result } = pendingClarification;
  const buttons = [...document.querySelectorAll("#clarification-buttons button")];
  buttons.forEach((button) => { button.disabled = true; });
  try {
    const response = await fetch("/api/clarification", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        gloss: candidate.gloss,
        confidence: result.confidence,
        context_glosses: recognizedGlosses.slice(-14),
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `Clarification failed (${response.status}).`);
    acceptRecognizedGloss(candidate.gloss);
    setText("#response", payload.sentence);
    setText("#gloss", friendlyWord(candidate.gloss));
    document.querySelector("#clarification-options").hidden = true;
    document.querySelector("#confidence-badge").textContent = "Understood";
    document.querySelector("#confidence-badge").className = "confidence-badge is-good";
    setText("#recognition-status", `Recognised: ${friendlyWord(candidate.gloss)} (you chose)`);
    playGlosses(payload.reply_glosses || [], payload.avatar?.unmatched || []);
    pendingClarification = undefined;
    loadFailureEvents();
  } catch (error) {
    setText("#recognition-status", error.message);
    buttons.forEach((button) => { button.disabled = false; });
  }
}

async function updateSentence() {
  renderRecognizedWords();
  if (!recognizedGlosses.length) {
    setText("#response", "Say a sign to build a sentence.");
    setText("#gloss", "—");
    setText("#confidence", "—");
    document.querySelector("#confidence-fill").style.width = "0%";
    document.querySelector("#clarification-options").hidden = true;
    playGlosses([]);
    return;
  }
  const response = await fetch("/api/compose_sentence", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ glosses: recognizedGlosses }),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Sentence update failed (${response.status}).`);
  setText("#response", result.sentence);
  playGlosses(result.reply_glosses || recognizedGlosses);
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
document.querySelector("#undo-word").addEventListener("click", async () => {
  recognizedGlosses.pop();
  try {
    await updateSentence();
  } catch (error) {
    setText("#recognition-status", error.message);
  }
});
document.querySelector("#clear-words").addEventListener("click", async () => {
  recognizedGlosses = [];
  pendingClarification = undefined;
  try {
    await updateSentence();
    setText("#recognition-status", "Show your hands.");
  } catch (error) {
    setText("#recognition-status", error.message);
  }
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
