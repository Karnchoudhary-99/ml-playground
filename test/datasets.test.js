import test from "node:test";
import assert from "node:assert/strict";
import {
  twoBlobs,
  moons,
  noisyLine,
  spirals,
  rings,
  xorQuadrants,
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

test("spirals are deterministic and both arms grow outward", () => {
  const a = spirals({ n: 60, seed: 7 });
  const b = spirals({ n: 60, seed: 7 });
  assert.deepEqual(a, b);
  assert.equal(a.filter((p) => p.label === 0).length, 30);
  // Later points of each arm sit farther from the origin than early ones.
  const arm0 = a.filter((p) => p.label === 0);
  const r = (p) => Math.hypot(p.x, p.y);
  assert.ok(r(arm0[arm0.length - 1]) > r(arm0[0]) + 1);
});

test("rings keep inner and outer radii apart", () => {
  const pts = rings({ n: 60, seed: 7, noise: 0.05 });
  const r = (p) => Math.hypot(p.x, p.y);
  const inner = pts.filter((p) => p.label === 0).map(r);
  const outer = pts.filter((p) => p.label === 1).map(r);
  assert.ok(Math.max(...inner) < 1.5);
  assert.ok(Math.min(...outer) > 2.0);
});

test("xorQuadrants labels follow the opposite-signs rule", () => {
  const pts = xorQuadrants({ n: 80, seed: 7 });
  assert.ok(
    pts.every((p) => p.label === ((p.x > 0) !== (p.y > 0) ? 1 : 0)),
  );
  assert.ok(pts.some((p) => p.label === 0) && pts.some((p) => p.label === 1));
});

test("trainTestSplit partitions without loss or duplication", () => {
  const pts = twoBlobs({ n: 40, seed: 9 });
  const { train, test } = trainTestSplit(pts, { testFraction: 0.25 });
  assert.equal(train.length + test.length, 40);
  assert.equal(test.length, 10);
  const keys = new Set([...train, ...test].map((p) => `${p.x},${p.y}`));
  assert.equal(keys.size, 40);
});
