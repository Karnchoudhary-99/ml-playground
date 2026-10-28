import test from "node:test";
import assert from "node:assert/strict";
import { KnnClassifier, KnnRegressor } from "../src/knn.js";
import { twoBlobs, noisyLine, trainTestSplit } from "../src/datasets.js";

test("k-NN classifier separates two blobs", () => {
  const pts = twoBlobs({ n: 100, seed: 11 });
  const { train, test } = trainTestSplit(pts, { seed: 4 });
  const knn = new KnnClassifier({ k: 5 }).fit(train);
  const correct = test.filter((p) => knn.predict(p) === p.label).length;
  assert.ok(correct / test.length >= 0.9, `accuracy ${correct}/${test.length}`);
});

test("k-NN classifier with k=1 memorises its own training point", () => {
  const train = [
    { x: 0, y: 0, label: 0 },
    { x: 10, y: 10, label: 1 },
  ];
  const knn = new KnnClassifier({ k: 1 }).fit(train);
  assert.equal(knn.predict({ x: 0, y: 0 }), 0);
  assert.equal(knn.predict({ x: 10, y: 10 }), 1);
});

test("k-NN rejects empty training data and k < 1", () => {
  assert.throws(() => new KnnClassifier({ k: 5 }).fit([]).predict({ x: 0, y: 0 }), /no training/);
  assert.throws(
    () => new KnnClassifier({ k: 0 }).fit([{ x: 0, y: 0, label: 0 }]).predict({ x: 0, y: 0 }),
    /k must be/,
  );
});

test("k-NN regressor tracks a noisy line", () => {
  const pts = noisyLine({ n: 60, seed: 2, noise: 0.3 });
  const { train, test } = trainTestSplit(pts, { seed: 6 });
  const knn = new KnnRegressor({ k: 3 }).fit(train);
  const mse =
    test.reduce((s, p) => s + (knn.predict(p) - p.y) ** 2, 0) / test.length;
  assert.ok(mse < 1.0, `mse ${mse}`);
});

test("manhattan metric works", () => {
  const knn = new KnnClassifier({ k: 1, metric: "manhattan" }).fit([
    { x: 0, y: 0, label: 0 },
    { x: 3, y: 4, label: 1 },
  ]);
  assert.equal(knn.predict({ x: 1, y: 0 }), 0);
});
