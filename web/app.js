const camera = document.querySelector("#camera");
const cameraEmpty = document.querySelector("#camera-empty");
const cameraToggle = document.querySelector("#camera-toggle");
const systemStatus = document.querySelector("#system-status");
const sampleSelect = document.querySelector("#sample-select");
const runSample = document.querySelector("#run-sample");
let stream;

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
    stream = undefined;
    camera.srcObject = null;
    cameraEmpty.hidden = false;
    cameraToggle.textContent = "Enable camera";
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    camera.srcObject = stream;
    cameraEmpty.hidden = true;
    cameraToggle.textContent = "Stop camera";
  } catch {
    setText("#camera-note", "Camera permission was not granted.");
  }
});

Promise.all([checkHealth(), loadSamples()]);
