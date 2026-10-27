import test from "node:test";
import assert from "node:assert/strict";
import { fitLinear } from "../src/linear.js";

test("fitLinear recovers y = 2x + 1", () => {
  const model = fitLinear([{ x: 0, y: 1 }, { x: 1, y: 3 }, { x: 2, y: 5 }]);
  assert.ok(Math.abs(model.a - 2) < 1e-9);
  assert.ok(Math.abs(model.b - 1) < 1e-9);
  assert.ok(Math.abs(model.predict(3) - 7) < 1e-9);
});

test("fitLinear rejects too few points", () => {
  assert.throws(() => fitLinear([{ x: 0, y: 1 }]), /at least two/);
});
