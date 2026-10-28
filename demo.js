// Interactive demo logic: dataset -> model -> canvas rendering.
// Imports the exact same modules the Node tests cover, so the page and the
// test suite can never drift apart.
import { twoBlobs, moons, spirals, rings, xorQuadrants, noisyLine } from "./src/datasets.js";
import { fitStandardScaler } from "./src/scaling.js";
import { KnnClassifier, KnnRegressor } from "./src/knn.js";
import { LogisticRegression } from "./src/logistic.js";
import { fitLinear } from "./src/linear.js";
import { accuracy, mse, r2 } from "./src/metrics.js";
import {
  serializeLinear,
  serializeLogistic,
  serializeKnn,
  modelFromString,
  modelToString,
} from "./src/serialization.js";

const canvas = document.getElementById("chart");
const ctx = canvas.getContext("2d");
const datasetEl = document.getElementById("dataset");
const modelEl = document.getElementById("model");
const kEl = document.getElementById("k");
const kWrap = document.getElementById("kWrap");
const clickLabelEl = document.getElementById("clickLabel");
const metricsEl = document.getElementById("metrics");

let points = [];
let seed = 42;
let model = null; // { kind, predict(point) -> label | number, isClassifier }
let rawModel = null; // the underlying trained model, for export
let scaler = null;

const COLORS = ["#2563eb", "#ea580c"];
const BOUNDS = { minX: -4.5, maxX: 4.5, minY: -3.5, maxY: 3.5 };

function generate() {
  seed += 1;
  if (datasetEl.value === "blobs") points = twoBlobs({ n: 100, seed });
  else if (datasetEl.value === "moons") points = moons({ n: 100, seed });
  else if (datasetEl.value === "spirals") points = spirals({ n: 100, seed });
  else if (datasetEl.value === "rings") points = rings({ n: 100, seed });
  else if (datasetEl.value === "xor") points = xorQuadrants({ n: 100, seed });
  else points = noisyLine({ n: 60, seed });
  model = null;
  rawModel = null;
  draw();
  metricsEl.textContent = "Press Train to fit a model.";
}

function syncModelOptions() {
  const isRegression = datasetEl.value === "line";
  modelEl.innerHTML = isRegression
    ? `<option value="linear">Linear regression</option><option value="knn">k-NN regressor</option>`
    : `<option value="knn">k-NN</option><option value="logistic">Logistic regression</option>`;
  kWrap.style.display = modelEl.value === "logistic" || modelEl.value === "linear" ? "none" : "flex";
}

function train() {
  const k = Math.max(1, parseInt(kEl.value, 10) || 5);
  if (datasetEl.value === "line") {
    if (modelEl.value === "linear") {
      const lin = fitLinear(points);
      rawModel = lin;
      model = { kind: "linear", isClassifier: false, predict: (p) => lin.predict(p.x) };
      metricsEl.textContent =
        `y = ${lin.a.toFixed(2)}x + ${lin.b.toFixed(2)} — ` +
        `MSE ${mse(points, model.predict).toFixed(3)}, R² ${r2(points, model.predict).toFixed(3)}`;
    } else {
      const knn = new KnnRegressor({ k }).fit(points);
      rawModel = knn;
      model = { kind: "knn", isClassifier: false, predict: (p) => knn.predict(p) };
      metricsEl.textContent =
        `k-NN regressor (k=${k}) — MSE ${mse(points, model.predict).toFixed(3)}, ` +
        `R² ${r2(points, model.predict).toFixed(3)}`;
    }
  } else {
    scaler = fitStandardScaler(points);
    const scaled = points.map((p) => ({ ...scaler.transform(p), label: p.label }));
    if (modelEl.value === "knn") {
      const knn = new KnnClassifier({ k }).fit(scaled);
      rawModel = knn;
      model = {
        kind: "knn", isClassifier: true,
        predict: (p) => knn.predict(scaler.transform(p)),
      };
    } else {
      const logreg = new LogisticRegression({ epochs: 800 }).fit(scaled);
      rawModel = logreg;
      model = {
        kind: "logistic", isClassifier: true,
        predict: (p) => logreg.predict(scaler.transform(p)),
      };
    }
    metricsEl.textContent =
      `${model.kind === "knn" ? `k-NN (k=${k})` : "Logistic regression"} — ` +
      `train accuracy ${(accuracy(points, model.predict) * 100).toFixed(1)}%`;
  }
  draw();
}

function toPx(p) {
  const { minX, maxX, minY, maxY } = BOUNDS;
  return {
    px: ((p.x - minX) / (maxX - minX)) * canvas.width,
    py: canvas.height - ((p.y - minY) / (maxY - minY)) * canvas.height,
  };
}

function toData(px, py) {
  const { minX, maxX, minY, maxY } = BOUNDS;
  return {
    x: minX + (px / canvas.width) * (maxX - minX),
    y: minY + ((canvas.height - py) / canvas.height) * (maxY - minY),
  };
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (model?.isClassifier) {
    // Decision-boundary background, sampled on a coarse grid.
    const cols = 64;
    const rows = 48;
    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < rows; j++) {
        const p = toData(
          ((i + 0.5) / cols) * canvas.width,
          ((j + 0.5) / rows) * canvas.height,
        );
        ctx.fillStyle = model.predict(p) === 0 ? "#dbeafe" : "#ffedd5";
        ctx.fillRect(
          (i / cols) * canvas.width,
          (j / rows) * canvas.height,
          canvas.width / cols + 1,
          canvas.height / rows + 1,
        );
      }
    }
  } else if (model) {
    ctx.strokeStyle = "#16a34a";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i <= 100; i++) {
      const x = BOUNDS.minX + ((BOUNDS.maxX - BOUNDS.minX) * i) / 100;
      const { px, py } = toPx({ x, y: model.predict({ x, y: 0 }) });
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
  for (const p of points) {
    const { px, py } = toPx(p);
    ctx.beginPath();
    ctx.arc(px, py, 5, 0, 2 * Math.PI);
    ctx.fillStyle = p.label === undefined ? "#111827" : COLORS[p.label];
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

canvas.addEventListener("click", (event) => {
  const rect = canvas.getBoundingClientRect();
  const px = ((event.clientX - rect.left) / rect.width) * canvas.width;
  const py = ((event.clientY - rect.top) / rect.height) * canvas.height;
  const p = toData(px, py);
  if (datasetEl.value === "line") {
    points.push(p);
  } else {
    points.push({ ...p, label: parseInt(clickLabelEl.value, 10) });
  }
  if (model) train(); // keep the boundary live as the user draws
  else draw();
});

function exportModel() {
  if (!model || !rawModel) {
    metricsEl.textContent = "Train a model first, then export it.";
    return;
  }
  let doc;
  if (model.kind === "linear") doc = serializeLinear(rawModel);
  else if (model.kind === "logistic") doc = serializeLogistic(rawModel, scaler);
  else doc = serializeKnn(rawModel, model.isClassifier ? scaler : null);
  const blob = new Blob([modelToString(doc)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `ml-playground-${model.kind}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
  metricsEl.textContent = `Exported ${model.kind} model as JSON.`;
}

async function importModel(file) {
  try {
    const { type, model: loaded, scaler: loadedScaler } = modelFromString(await file.text());
    scaler = loadedScaler;
    rawModel = loaded;
    const scaled = (p) => (scaler ? scaler.transform(p) : p);
    if (type === "linear") {
      model = { kind: "linear", isClassifier: false, predict: (p) => loaded.predict(p.x) };
    } else if (type === "logistic") {
      model = { kind: "logistic", isClassifier: true, predict: (p) => loaded.predict(scaled(p)) };
    } else if (type === "knn-classifier") {
      model = { kind: "knn", isClassifier: true, predict: (p) => loaded.predict(scaled(p)) };
    } else {
      model = { kind: "knn", isClassifier: false, predict: (p) => loaded.predict(p) };
    }
    metricsEl.textContent =
      `Imported ${type} model — its boundary/curve now renders over the current points.`;
    draw();
  } catch (err) {
    metricsEl.textContent = `Import failed: ${err.message}`;
  }
}

datasetEl.addEventListener("change", () => {
  syncModelOptions();
  generate();
});
modelEl.addEventListener("change", () => {
  kWrap.style.display = modelEl.value === "logistic" || modelEl.value === "linear" ? "none" : "flex";
});
document.getElementById("train").addEventListener("click", train);
document.getElementById("reshuffle").addEventListener("click", generate);
document.getElementById("export").addEventListener("click", exportModel);
document.getElementById("import").addEventListener("click", () => {
  document.getElementById("importFile").click();
});
document.getElementById("importFile").addEventListener("change", (event) => {
  const file = event.target.files[0];
  event.target.value = "";
  if (file) importModel(file);
});

syncModelOptions();
generate();
