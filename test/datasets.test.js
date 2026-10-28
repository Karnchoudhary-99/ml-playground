import test from "node:test";
import assert from "node:assert/strict";
import {
  twoBlobs,
  moons,
  noisyLine,
  trainTestSplit,
} from "../src/datasets.js";

test("twoBlobs is deterministic and balanced", () => {
  const a = twoBlobs({ n: 40, seed: 3 });
  const b = twoBlobs({ n: 40, seed: 3 });
  assert.deepEqual(a, b);
  assert.equal(a.filter((p) => p.label === 0).length, 20);
  assert.equal(a.filter((p) => p.label === 1).length, 20);
});

test("moons has both labels and bounded range", () => {
  const pts = moons({ n: 60, seed: 5 });
  assert.equal(pts.length, 60);
  assert.ok(pts.some((p) => p.label === 0) && pts.some((p) => p.label === 1));
  assert.ok(pts.every((p) => Math.abs(p.x) < 4 && Math.abs(p.y) < 4));
});

test("noisyLine follows y = 2x + 1 on average", () => {
  const pts = noisyLine({ n: 100, seed: 1, noise: 0.1 });
  const meanResidual =
    pts.reduce((s, p) => s + (p.y - (2 * p.x + 1)), 0) / pts.length;
  assert.ok(Math.abs(meanResidual) < 0.1, `mean residual ${meanResidual}`);
});

test("trainTestSplit partitions without loss or duplication", () => {
  const pts = twoBlobs({ n: 40, seed: 9 });
  const { train, test } = trainTestSplit(pts, { testFraction: 0.25 });
  assert.equal(train.length + test.length, 40);
  assert.equal(test.length, 10);
  const keys = new Set([...train, ...test].map((p) => `${p.x},${p.y}`));
  assert.equal(keys.size, 40);
});
