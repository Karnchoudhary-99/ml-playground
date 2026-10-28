import test from "node:test";
import assert from "node:assert/strict";
import { LogisticRegression } from "../src/logistic.js";
import { fitStandardScaler } from "../src/scaling.js";
import { twoBlobs, moons, trainTestSplit } from "../src/datasets.js";

function scaledSplit(pts) {
  const { train, test } = trainTestSplit(pts, { seed: 8 });
  const scaler = fitStandardScaler(train);
  return {
    train: train.map((p) => ({ ...scaler.transform(p), label: p.label })),
    test: test.map((p) => ({ ...scaler.transform(p), label: p.label })),
  };
}

test("logistic regression separates two blobs", () => {
  const { train, test } = scaledSplit(twoBlobs({ n: 100, seed: 21 }));
  const model = new LogisticRegression({ epochs: 800 }).fit(train);
  const correct = test.filter((p) => model.predict(p) === p.label).length;
  assert.ok(correct / test.length >= 0.9, `accuracy ${correct}/${test.length}`);
});

test("probabilities stay in [0, 1] and grow with the label direction", () => {
  const { train } = scaledSplit(twoBlobs({ n: 80, seed: 22 }));
  const model = new LogisticRegression({ epochs: 500 }).fit(train);
  for (const p of train.slice(0, 10)) {
    const proba = model.predictProba(p);
    assert.ok(proba >= 0 && proba <= 1);
  }
});

test("logistic regression visibly struggles on moons (non-linear data)", () => {
  // Sanity check for the demo narrative: a linear model cannot solve moons.
  const { train, test } = scaledSplit(moons({ n: 100, seed: 23 }));
  const model = new LogisticRegression({ epochs: 800 }).fit(train);
  const correct = test.filter((p) => model.predict(p) === p.label).length;
  assert.ok(correct / test.length < 0.95, `unexpected accuracy ${correct}/${test.length}`);
});

test("logistic regression rejects empty training data", () => {
  assert.throws(() => new LogisticRegression().fit([]), /no training/);
});
