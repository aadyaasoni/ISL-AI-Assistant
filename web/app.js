const camera = document.querySelector("#camera");
const cameraEmpty = document.querySelector("#camera-empty");
const cameraToggle = document.querySelector("#camera-toggle");
const systemStatus = document.querySelector("#system-status");
const sampleSelect = document.querySelector("#sample-select");
const runSample = document.querySelector("#run-sample");
let stream;
let captureTimer;
let frameCanvas;
let frameBuffer = [];
let frameBusy = false;
const sequenceLength = 24;

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

Promise.all([checkHealth(), loadSamples()]);
