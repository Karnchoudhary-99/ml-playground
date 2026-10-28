// Versioned model import/export, dependency-free like the rest of src/.
// A model file is plain JSON so a user can save a trained model from the
// demo page, share it, and load it back later — in the browser or in Node.
// Loading is deliberately strict: unknown formats, unsupported versions,
// missing fields and non-finite numbers are all rejected, because a model
// file that silently loads wrong is worse than one that refuses to load.
//
// Format: { format: "ml-playground-model", version: 1,
//           type: "linear" | "logistic" | "knn-classifier" | "knn-regressor",
//           params: { ...model-specific... }, scaler: { mean, std } | null }
// Classifiers trained on scaled features carry their scaler, so an imported
// classifier scales new points exactly like the original did.
import { LogisticRegression } from "./logistic.js";
import { KnnClassifier, KnnRegressor } from "./knn.js";
import { scalerFromParams } from "./scaling.js";

export const MODEL_FORMAT = "ml-playground-model";
export const MODEL_VERSION = 1;

function fail(message) {
  throw new Error(`model JSON: ${message}`);
}

function finiteNumber(value, name) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    fail(`${name} must be a finite number`);
  }
  return value;
}

function checkScaler(scaler) {
  if (scaler === null || scaler === undefined) return null;
  if (typeof scaler !== "object" || Array.isArray(scaler)) {
    fail("scaler must be an object or null");
  }
  const mean = {};
  const std = {};
  for (const dim of ["x", "y"]) {
    mean[dim] = finiteNumber(scaler.mean?.[dim], `scaler.mean.${dim}`);
    std[dim] = finiteNumber(scaler.std?.[dim], `scaler.std.${dim}`);
    if (std[dim] <= 0) fail(`scaler.std.${dim} must be positive`);
  }
  return scalerFromParams(mean, std);
}

function checkPoint(point, labelled) {
  if (typeof point !== "object" || point === null || Array.isArray(point)) {
    fail("training point must be an object");
  }
  const clean = {
    x: finiteNumber(point.x, "training point x"),
    y: finiteNumber(point.y, "training point y"),
  };
  if (labelled) {
    if (point.label !== 0 && point.label !== 1) {
      fail("classifier training point label must be 0 or 1");
    }
    clean.label = point.label;
  }
  return clean;
}

function checkKnnConfig(params) {
  if (!Number.isInteger(params.k) || params.k < 1) {
    fail("k-NN k must be an integer >= 1");
  }
  if (params.metric !== "euclidean" && params.metric !== "manhattan") {
    fail('k-NN metric must be "euclidean" or "manhattan"');
  }
  if (params.weights !== "distance" && params.weights !== "uniform") {
    fail('k-NN weights must be "distance" or "uniform"');
  }
}

export function serializeLinear(model) {
  return {
    format: MODEL_FORMAT,
    version: MODEL_VERSION,
    type: "linear",
    params: {
      a: finiteNumber(model.a, "linear slope a"),
      b: finiteNumber(model.b, "linear intercept b"),
    },
    scaler: null,
  };
}

export function serializeLogistic(model, scaler = null) {
  if (!Array.isArray(model.weights) || model.weights.length !== 2) {
    fail("logistic model needs a 2-element weights array");
  }
  return {
    format: MODEL_FORMAT,
    version: MODEL_VERSION,
    type: "logistic",
    params: {
      weights: model.weights.map((w, i) => finiteNumber(w, `weight ${i}`)),
      bias: finiteNumber(model.bias, "logistic bias"),
      learningRate: finiteNumber(model.learningRate, "learningRate"),
      epochs: finiteNumber(model.epochs, "epochs"),
      l2: finiteNumber(model.l2, "l2"),
    },
    scaler: scaler ? { mean: scaler.mean, std: scaler.std } : null,
  };
}

export function serializeKnn(model, scaler = null) {
  const isClassifier = model instanceof KnnClassifier;
  if (!isClassifier && !(model instanceof KnnRegressor)) {
    fail("k-NN serialization needs a KnnClassifier or KnnRegressor");
  }
  if (!Array.isArray(model.train) || model.train.length === 0) {
    fail("k-NN model has no training points to save");
  }
  const params = {
    k: model.k,
    metric: model.metric,
    weights: model.weights,
    train: model.train.map((p) => checkPoint(p, isClassifier)),
  };
  checkKnnConfig(params);
  return {
    format: MODEL_FORMAT,
    version: MODEL_VERSION,
    type: isClassifier ? "knn-classifier" : "knn-regressor",
    params,
    scaler: scaler ? { mean: scaler.mean, std: scaler.std } : null,
  };
}

// Rebuild a model from a parsed model document. Returns
// { type, model, scaler } where model exposes predict() like the original:
// linear models predict(x), k-NN regressors predict(point), and classifiers
// predict(point) — callers must pass points through scaler.transform first
// when scaler is non-null, exactly as the demo page does when training.
export function deserializeModel(doc) {
  if (typeof doc !== "object" || doc === null || Array.isArray(doc)) {
    fail("expected a model object");
  }
  if (doc.format !== MODEL_FORMAT) fail(`unknown format ${JSON.stringify(doc.format)}`);
  if (doc.version !== MODEL_VERSION) {
    fail(`unsupported version ${JSON.stringify(doc.version)} (expected ${MODEL_VERSION})`);
  }
  if (typeof doc.params !== "object" || doc.params === null) {
    fail("params must be an object");
  }
  const scaler = checkScaler(doc.scaler ?? null);
  switch (doc.type) {
    case "linear": {
      const a = finiteNumber(doc.params.a, "linear slope a");
      const b = finiteNumber(doc.params.b, "linear intercept b");
      return { type: doc.type, model: { a, b, predict: (x) => a * x + b }, scaler };
    }
    case "logistic": {
      if (!Array.isArray(doc.params.weights) || doc.params.weights.length !== 2) {
        fail("logistic weights must be a 2-element array");
      }
      const model = new LogisticRegression({
        learningRate: finiteNumber(doc.params.learningRate, "learningRate"),
        epochs: finiteNumber(doc.params.epochs, "epochs"),
        l2: finiteNumber(doc.params.l2, "l2"),
      });
      model.weights = doc.params.weights.map((w, i) => finiteNumber(w, `weight ${i}`));
      model.bias = finiteNumber(doc.params.bias, "logistic bias");
      return { type: doc.type, model, scaler };
    }
    case "knn-classifier":
    case "knn-regressor": {
      checkKnnConfig(doc.params);
      if (!Array.isArray(doc.params.train) || doc.params.train.length === 0) {
        fail("k-NN model needs a non-empty train array");
      }
      const labelled = doc.type === "knn-classifier";
      const train = doc.params.train.map((p) => checkPoint(p, labelled));
      const Model = labelled ? KnnClassifier : KnnRegressor;
      const model = new Model({
        k: doc.params.k,
        metric: doc.params.metric,
        weights: doc.params.weights,
      }).fit(train);
      return { type: doc.type, model, scaler };
    }
    default:
      fail(`unknown type ${JSON.stringify(doc.type)}`);
  }
}

export function modelToString(doc) {
  return JSON.stringify(doc, null, 2);
}

export function modelFromString(text) {
  let doc;
  try {
    doc = JSON.parse(text);
  } catch {
    fail("invalid JSON");
  }
  return deserializeModel(doc);
}
