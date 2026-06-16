const apiStatus = document.getElementById("apiStatus");
const healthStatus = document.getElementById("healthStatus");
const gpuStatus = document.getElementById("gpuStatus");
const modelStatus = document.getElementById("modelStatus");
const checkpointStatus = document.getElementById("checkpointStatus");
const imageInput = document.getElementById("imageInput");
const predictButton = document.getElementById("predictButton");
const explainButton = document.getElementById("explainButton");
const imagePreview = document.getElementById("imagePreview");
const emptyPreview = document.getElementById("emptyPreview");
const resultFilename = document.getElementById("resultFilename");
const resultClass = document.getElementById("resultClass");
const resultConfidence = document.getElementById("resultConfidence");
const predictionBadge = document.getElementById("predictionBadge");
const probabilities = document.getElementById("probabilities");
const downloadJson = document.getElementById("downloadJson");
const message = document.getElementById("message");
const probabilityPlot = document.getElementById("probabilityPlot");
const emptyPlot = document.getElementById("emptyPlot");
const attentionOverlay = document.getElementById("attentionOverlay");
const emptyOverlay = document.getElementById("emptyOverlay");
const resetButton = document.getElementById("resetButton");

let selectedFile = null;
let predictionDownloadUrl = "";

function setMessage(text, isError = false) {
  message.textContent = text;
  message.classList.toggle("error", isError);
}

function setApiStatus(text, state) {
  apiStatus.textContent = text;
  apiStatus.classList.remove("ok", "error");
  if (state) {
    apiStatus.classList.add(state);
  }
}

function formatPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function artifactPathToUrl(path) {
  if (!path) {
    return "";
  }

  // The API returns repository-relative paths such as outputs/attention_maps/x.png.
  // Prefixing with "/" lets the browser request the FastAPI static route.
  return path.startsWith("/") ? path : `/${path}`;
}

function updatePredictionColor(predictedClass) {
  const normalizedClass = (predictedClass || "").toLowerCase();
  predictionBadge.classList.remove("cancer", "benign");

  if (normalizedClass.includes("cancer")) {
    predictionBadge.textContent = "Cancer";
    predictionBadge.classList.add("cancer");
  } else if (normalizedClass.includes("benign")) {
    predictionBadge.textContent = "Benign";
    predictionBadge.classList.add("benign");
  } else {
    predictionBadge.textContent = "Waiting";
  }
}

async function loadStatus() {
  try {
    const [healthResponse, modelResponse] = await Promise.all([
      fetch("/health"),
      fetch("/model/status"),
    ]);

    if (!healthResponse.ok || !modelResponse.ok) {
      throw new Error("Status request failed.");
    }

    const health = await healthResponse.json();
    const model = await modelResponse.json();

    healthStatus.textContent = health.status;
    gpuStatus.textContent = health.gpu ? "Available" : "CPU";
    modelStatus.textContent = model.model_loaded ? "Loaded" : "Not loaded";
    checkpointStatus.textContent = model.checkpoint_path || "Unknown";
    setApiStatus("API online", "ok");
  } catch (error) {
    healthStatus.textContent = "Offline";
    gpuStatus.textContent = "Unknown";
    modelStatus.textContent = "Unknown";
    checkpointStatus.textContent = "Unknown";
    setApiStatus("API offline", "error");
  }
}

function showPreview(file) {
  const previewUrl = URL.createObjectURL(file);
  imagePreview.src = previewUrl;
  imagePreview.style.display = "block";
  emptyPreview.style.display = "none";
}

function clearResult() {
  resultFilename.textContent = "-";
  resultClass.textContent = "-";
  resultConfidence.textContent = "-";
  predictionBadge.textContent = "Waiting";
  predictionBadge.classList.remove("cancer", "benign");
  probabilities.innerHTML = "Run prediction to view probability bars.";
  probabilities.classList.add("empty-probabilities");
  if (predictionDownloadUrl) {
    URL.revokeObjectURL(predictionDownloadUrl);
    predictionDownloadUrl = "";
  }

  downloadJson.removeAttribute("href");
  downloadJson.style.display = "none";
  probabilityPlot.removeAttribute("src");
  probabilityPlot.style.display = "none";
  emptyPlot.style.display = "block";
  attentionOverlay.removeAttribute("src");
  attentionOverlay.style.display = "none";
  emptyOverlay.style.display = "block";
}

function renderProbabilities(probabilityMap) {
  probabilities.innerHTML = "";
  probabilities.classList.remove("empty-probabilities");

  Object.entries(probabilityMap).forEach(([className, value]) => {
    const row = document.createElement("div");
    row.className = "probability";

    const header = document.createElement("div");
    header.className = "probability-header";
    header.innerHTML = `<span>${className}</span><strong>${formatPercent(value)}</strong>`;

    const bar = document.createElement("div");
    bar.className = "bar";

    const fill = document.createElement("span");
    fill.style.width = formatPercent(value);
    fill.className = className.toLowerCase();
    bar.appendChild(fill);

    row.appendChild(header);
    row.appendChild(bar);
    probabilities.appendChild(row);
  });
}

function renderArtifacts(artifacts, payload) {
  const probabilityPlotUrl = artifactPathToUrl(artifacts?.probability_plot);
  const attentionOverlayUrl = artifactPathToUrl(artifacts?.attention_overlay);

  if (artifacts?.prediction_json) {
    if (predictionDownloadUrl) {
      URL.revokeObjectURL(predictionDownloadUrl);
    }

    const jsonBlob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    predictionDownloadUrl = URL.createObjectURL(jsonBlob);
    downloadJson.href = predictionDownloadUrl;
    downloadJson.download = artifacts.prediction_json.split("/").pop();
    downloadJson.style.display = "flex";
  }

  if (probabilityPlotUrl) {
    probabilityPlot.src = probabilityPlotUrl;
    probabilityPlot.style.display = "block";
    emptyPlot.style.display = "none";
  }

  if (attentionOverlayUrl) {
    attentionOverlay.src = attentionOverlayUrl;
    attentionOverlay.style.display = "block";
    emptyOverlay.style.display = "none";
  }
}

function renderPrediction(payload) {
  resultFilename.textContent = payload.filename;
  resultClass.textContent = payload.predicted_class;
  resultConfidence.textContent = formatPercent(payload.confidence);
  updatePredictionColor(payload.predicted_class);
  renderProbabilities(payload.probabilities);
  renderArtifacts(payload.artifacts || {}, payload);
}

async function predictSelectedImage() {
  if (!selectedFile) {
    return;
  }

  const formData = new FormData();
  formData.append("file", selectedFile);

  predictButton.disabled = true;
  explainButton.disabled = true;
  setMessage("Running prediction...");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Prediction failed.");
    }

    renderPrediction(payload);
    setMessage("Prediction complete.");
  } catch (error) {
    clearResult();
    setMessage(error.message || "Prediction failed.", true);
  } finally {
    predictButton.disabled = !selectedFile;
    explainButton.disabled = !selectedFile;
  }
}

async function explainSelectedImage() {
  if (!selectedFile) {
    return;
  }

  const formData = new FormData();
  formData.append("file", selectedFile);

  predictButton.disabled = true;
  explainButton.disabled = true;
  setMessage("Generating explanation...");

  try {
    const response = await fetch("/explain", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Explanation failed.");
    }

    renderPrediction(payload);
    setMessage(payload.warning || "Explanation complete.");
  } catch (error) {
    setMessage(error.message || "Explanation failed.", true);
  } finally {
    predictButton.disabled = !selectedFile;
    explainButton.disabled = !selectedFile;
  }
}

imageInput.addEventListener("change", () => {
  selectedFile = imageInput.files[0] || null;
  predictButton.disabled = !selectedFile;
  explainButton.disabled = !selectedFile;
  clearResult();
  setMessage("");

  if (selectedFile) {
    showPreview(selectedFile);
  }
});

resetButton.addEventListener("click", () => {
  selectedFile = null;
  imageInput.value = "";
  imagePreview.removeAttribute("src");
  imagePreview.style.display = "none";
  emptyPreview.style.display = "block";
  predictButton.disabled = true;
  explainButton.disabled = true;
  clearResult();
  setMessage("");
});

predictButton.addEventListener("click", predictSelectedImage);
explainButton.addEventListener("click", explainSelectedImage);

loadStatus();
