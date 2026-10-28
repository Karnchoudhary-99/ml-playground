import test from "node:test";
import assert from "node:assert/strict";
import { fitStandardScaler } from "../src/scaling.js";

test("standard scaler centres data and round-trips", () => {
  const pts = [
    { x: 0, y: 10 },
    { x: 2, y: 20 },
    { x: 4, y: 30 },
  ];
  const scaler = fitStandardScaler(pts);
  const scaled = pts.map(scaler.transform);
  const meanX = scaled.reduce((s, p) => s + p.x, 0) / 3;
  assert.ok(Math.abs(meanX) < 1e-9);
  for (let i = 0; i < pts.length; i++) {
    const back = scaler.inverse(scaled[i]);
    assert.ok(Math.abs(back.x - pts[i].x) < 1e-9);
    assert.ok(Math.abs(back.y - pts[i].y) < 1e-9);
  }
});

test("constant feature does not produce NaN", () => {
  const scaler = fitStandardScaler([
    { x: 5, y: 1 },
    { x: 5, y: 2 },
  ]);
  const out = scaler.transform({ x: 5, y: 1.5 });
  assert.ok(Number.isFinite(out.x) && Number.isFinite(out.y));
  assert.equal(out.x, 0);
});
