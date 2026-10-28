import test from "node:test";
import assert from "node:assert/strict";
import { fitLinear } from "../src/linear.js";
import { LogisticRegression } from "../src/logistic.js";
import { KnnClassifier, KnnRegressor } from "../src/knn.js";
import { fitStandardScaler } from "../src/scaling.js";
import {
  serializeLinear,
  serializeLogistic,
  serializeKnn,
  deserializeModel,
  modelToString,
  modelFromString,
} from "../src/serialization.js";

test("linear model round-trips through JSON with identical predictions", () => {
  const original = fitLinear([
    { x: 0, y: 1 },
    { x: 2, y: 5 },
    { x: 4, y: 9 },
  ]);
  const { model } = deserializeModel(serializeLinear(original));
  assert.equal(model.a, original.a);
  assert.equal(model.b, original.b);
  assert.equal(model.predict(3), original.predict(3));
});

test("logistic model round-trips weights, bias and scaler", () => {
  const raw = [
    { x: -2, y: -1, label: 0 },
    { x: -1, y: -1, label: 0 },
    { x: 2, y: 1, label: 1 },
    { x: 1, y: 2, label: 1 },
  ];
  const scaler = fitStandardScaler(raw);
  const scaled = raw.map((p) => ({ ...scaler.transform(p), label: p.label }));
  const original = new LogisticRegression({ epochs: 200 }).fit(scaled);
  const doc = serializeLogistic(original, scaler);
  const { model, scaler: loadedScaler } = deserializeModel(doc);
  assert.deepEqual(model.weights, original.weights);
  assert.equal(model.bias, original.bias);
  assert.deepEqual(loadedScaler.mean, scaler.mean);
  // Predictions through the loaded scaler match the original pipeline.
  for (const p of raw) {
    assert.equal(
      model.predict(loadedScaler.transform(p)),
      original.predict(scaler.transform(p)),
    );
  }
});

test("k-NN classifier round-trips its training data", () => {
  const train = [
    { x: 0, y: 0, label: 0 },
    { x: 0, y: 1, label: 0 },
    { x: 5, y: 5, label: 1 },
  ];
  const original = new KnnClassifier({ k: 1 }).fit(train);
  const { model } = deserializeModel(serializeKnn(original));
  assert.equal(model.predict({ x: 0.1, y: 0.1 }), 0);
  assert.equal(model.predict({ x: 4.9, y: 4.9 }), 1);
  assert.equal(model.predict({ x: 0.1, y: 0.1 }), original.predict({ x: 0.1, y: 0.1 }));
});

test("k-NN regressor round-trips and modelFromString parses text", () => {
  const train = [
    { x: 0, y: 0 },
    { x: 2, y: 4 },
    { x: 4, y: 8 },
  ];
  const original = new KnnRegressor({ k: 1 }).fit(train);
  const { model } = modelFromString(modelToString(serializeKnn(original)));
  assert.equal(model.predict({ x: 2.1, y: 4 }), 4);
});

test("deserialize rejects unknown format, version, type and bad JSON", () => {
  const good = serializeLinear(fitLinear([{ x: 0, y: 0 }, { x: 1, y: 1 }]));
  assert.throws(() => deserializeModel({ ...good, format: "other" }), /unknown format/);
  assert.throws(() => deserializeModel({ ...good, version: 99 }), /unsupported version/);
  assert.throws(() => deserializeModel({ ...good, type: "neural-net" }), /unknown type/);
  assert.throws(() => deserializeModel({ ...good, params: { a: NaN, b: 0 } }), /finite/);
  assert.throws(() => deserializeModel(null), /expected a model object/);
  assert.throws(() => modelFromString("{not json"), /invalid JSON/);
});

test("deserialize rejects malformed k-NN payloads", () => {
  const good = serializeKnn(new KnnClassifier({ k: 3 }).fit([{ x: 0, y: 0, label: 0 }]));
  assert.throws(() => deserializeModel({ ...good, params: { ...good.params, k: 0 } }), /k must be/);
  assert.throws(
    () => deserializeModel({ ...good, params: { ...good.params, metric: "cosine" } }),
    /metric must be/,
  );
  assert.throws(
    () => deserializeModel({ ...good, params: { ...good.params, train: [] } }),
    /non-empty train/,
  );
  assert.throws(
    () =>
      deserializeModel({
        ...good,
        params: { ...good.params, train: [{ x: 0, y: 0, label: 2 }] },
      }),
    /label must be 0 or 1/,
  );
  assert.throws(
    () => deserializeModel({ ...good, scaler: { mean: { x: 0, y: 0 }, std: { x: 1, y: 0 } } }),
    /must be positive/,
  );
});

test("serializing an unfitted k-NN model fails instead of saving an empty model", () => {
  assert.throws(() => serializeKnn(new KnnClassifier({ k: 3 })), /no training points/);
});
