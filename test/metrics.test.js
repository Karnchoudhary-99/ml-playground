import test from "node:test";
import assert from "node:assert/strict";
import { accuracy, mse, r2, confusionMatrix } from "../src/metrics.js";

test("accuracy and confusion matrix agree", () => {
  const pts = [
    { x: 0, y: 0, label: 1 },
    { x: 1, y: 0, label: 1 },
    { x: 2, y: 0, label: 0 },
    { x: 3, y: 0, label: 0 },
  ];
  const predict = (p) => (p.x < 2 ? 1 : 0);
  assert.equal(accuracy(pts, predict), 1);
  assert.deepEqual(confusionMatrix(pts, predict), [
    [2, 0],
    [0, 2],
  ]);
});

test("mse and r2 on a perfect predictor", () => {
  const pts = [
    { x: 0, y: 1 },
    { x: 1, y: 3 },
  ];
  assert.equal(mse(pts, (p) => 2 * p.x + 1), 0);
  assert.equal(r2(pts, (p) => 2 * p.x + 1), 1);
});

test("metrics reject empty input", () => {
  assert.throws(() => accuracy([], () => 0), /needs points/);
  assert.throws(() => mse([], () => 0), /needs points/);
  assert.throws(() => r2([], () => 0), /needs points/);
});
