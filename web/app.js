import { FilesetResolver, HandLandmarker, PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/+esm";

const camera = document.querySelector("#camera");
const cameraEmpty = document.querySelector("#camera-empty");
const cameraToggle = document.querySelector("#camera-toggle");
const systemStatus = document.querySelector("#system-status");
const sampleSelect = document.querySelector("#sample-select");
const runSample = document.querySelector("#run-sample");
let stream;
let handLandmarker;
let poseLandmarker;
let animationFrame;
let lastInference = 0;
const frameBuffer = [];
const SEQUENCE_LENGTH = 36;

function setText(selector, value) {
  document.querySelector(selector).textContent = value;
}

function showResult(payload) {
  const prediction = payload.prediction;
  const route = payload.route;
  const confidence = Number(prediction.confidence);
  setText("#gloss", prediction.gloss);
  setText("#confidence", `${(confidence * 100).toFixed(1)}%`);
  setText("#timestamp", new Date(prediction.timestamp * 1000).toLocaleTimeString());
  setText("#response", route.response_text || route.message || "No response returned.");
  setText("#avatar-status", `${payload.avatar.status} · ${payload.avatar.clip}`);
  document.querySelector("#confidence-fill").style.width = `${confidence * 100}%`;
  const badge = document.querySelector("#confidence-badge");
  badge.textContent = route.status === "accepted" ? "Accepted" : "Clarification";
  badge.className = `confidence-badge ${route.status === "accepted" ? "is-good" : "is-warn"}`;
}

async function createLandmarkers() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });
}

function normalizeFrame(handResult, poseResult) {
  const handValues = { left: Array(63).fill(0), right: Array(63).fill(0) };
  const handMask = { left: Array(21).fill(0), right: Array(21).fill(0) };
  const hands = handResult.landmarks || [];
  hands.forEach((landmarks, index) => {
    const label = (handResult.handedness?.[index]?.[0]?.categoryName || "").toLowerCase();
    const side = label === "left" ? "left" : label === "right" ? "right" : null;
    if (!side) return;
    landmarks.forEach((point, pointIndex) => {
      handValues[side].splice(pointIndex * 3, 3, point.x, point.y, point.z);
      handMask[side][pointIndex] = 1;
    });
  });

  const pose = poseResult.landmarks?.[0] || [];
  const leftShoulder = pose[11];
  const rightShoulder = pose[12];
  let center;
  let scale;
  if (leftShoulder && rightShoulder && leftShoulder.visibility > 0 && rightShoulder.visibility > 0) {
    center = {
      x: (leftShoulder.x + rightShoulder.x) / 2,
      y: (leftShoulder.y + rightShoulder.y) / 2,
      z: (leftShoulder.z + rightShoulder.z) / 2,
    };
    scale = Math.hypot(leftShoulder.x - rightShoulder.x, leftShoulder.y - rightShoulder.y, leftShoulder.z - rightShoulder.z);
  } else {
    const detected = hands.flat();
    center = detected.length
      ? detected.reduce((sum, point) => ({ x: sum.x + point.x, y: sum.y + point.y, z: sum.z + point.z }), { x: 0, y: 0, z: 0 })
      : { x: 0, y: 0, z: 0 };
    if (detected.length) {
      center.x /= detected.length;
      center.y /= detected.length;
      center.z /= detected.length;
    }
    scale = detected.length > 1 ? Math.hypot(detected[0].x - detected[1].x, detected[0].y - detected[1].y, detected[0].z - detected[1].z) : 1;
  }
  scale = Math.max(scale || 0, 1e-6);
  const poseValues = [];
  const poseMask = [];
  for (let index = 0; index < 33; index += 1) {
    const point = pose[index];
    if (!point) {
      poseValues.push(0, 0, 0, 0);
      poseMask.push(0);
    } else {
      poseValues.push((point.x - center.x) / scale, (point.y - center.y) / scale, (point.z - center.z) / scale, point.visibility || 0);
      poseMask.push(1);
    }
  }
  return {
    features: [...handValues.left, ...handValues.right, ...poseValues],
    mask: [...handMask.left, ...handMask.right, ...poseMask],
  };
}

async function inferCameraFrame(now) {
  if (!stream || !handLandmarker || !poseLandmarker) return;
  const timestamp = performance.now();
  const handResult = handLandmarker.detectForVideo(camera, timestamp);
  const poseResult = poseLandmarker.detectForVideo(camera, timestamp);
  const frame = normalizeFrame(handResult, poseResult);
  if (frame.mask.some((value) => value > 0)) frameBuffer.push(frame);
  if (frameBuffer.length > SEQUENCE_LENGTH) frameBuffer.shift();
  if (frameBuffer.length === SEQUENCE_LENGTH && now - lastInference > 1000) {
    lastInference = now;
    const response = await fetch("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        features: frameBuffer.map((frame) => frame.features),
        mask: frameBuffer.map((frame) => frame.mask),
        timestamp: timestamp / 1000,
      }),
    });
    if (response.ok) showResult(await response.json());
  }
  animationFrame = requestAnimationFrame(inferCameraFrame);
}

async function loadSamples() {
  const response = await fetch("/api/samples");
  const data = await response.json();
  data.samples.slice(0, 12).forEach((sample) => {
    const option = document.createElement("option");
    option.value = sample.sample_id;
    option.textContent = `${sample.label} · ${sample.sample_id}`;
    sampleSelect.append(option);
  });
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

cameraToggle.addEventListener("click", async () => {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    cancelAnimationFrame(animationFrame);
    stream = undefined;
    camera.srcObject = null;
    cameraEmpty.hidden = false;
    cameraToggle.textContent = "Enable camera";
    return;
  }
  try {
    cameraToggle.disabled = true;
    setText("#camera-note", "Loading browser landmark models…");
    await createLandmarkers();
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    camera.srcObject = stream;
    cameraEmpty.hidden = true;
    cameraToggle.textContent = "Stop camera";
    setText("#camera-note", "Browser landmarks are sent as 36-frame sequences.");
    animationFrame = requestAnimationFrame(inferCameraFrame);
  } catch {
    setText("#camera-note", "Camera permission was not granted.");
  } finally {
    cameraToggle.disabled = false;
  }
});

Promise.all([checkHealth(), loadSamples()]);
