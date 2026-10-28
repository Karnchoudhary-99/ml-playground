import test from "node:test";
import assert from "node:assert/strict";
import { mulberry32, gaussian } from "../src/rng.js";

test("mulberry32 is deterministic for a seed", () => {
  const a = mulberry32(42);
  const b = mulberry32(42);
  const seqA = [a(), a(), a()];
  const seqB = [b(), b(), b()];
  assert.deepEqual(seqA, seqB);
  assert.ok(seqA.every((v) => v >= 0 && v < 1));
});

test("mulberry32 differs across seeds", () => {
  assert.notEqual(mulberry32(1)(), mulberry32(2)());
});

test("gaussian samples roughly standard-normal", () => {
  const rng = mulberry32(7);
  const samples = Array.from({ length: 2000 }, () => gaussian(rng));
  const mean = samples.reduce((s, v) => s + v, 0) / samples.length;
  const variance =
    samples.reduce((s, v) => s + (v - mean) ** 2, 0) / samples.length;
  assert.ok(Math.abs(mean) < 0.1, `mean ${mean}`);
  assert.ok(Math.abs(variance - 1) < 0.15, `variance ${variance}`);
});
