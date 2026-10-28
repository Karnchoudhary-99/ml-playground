// Train a logistic model on moons, export it to JSON, load it back and
// show the reloaded model predicts identically. Usage: node examples/export.mjs
import { moons } from "../src/datasets.js";
import { fitStandardScaler } from "../src/scaling.js";
import { LogisticRegression } from "../src/logistic.js";
import {
  serializeLogistic,
  modelToString,
  modelFromString,
} from "../src/serialization.js";

const raw = moons({ n: 100, seed: 42 });
const scaler = fitStandardScaler(raw);
const scaled = raw.map((p) => ({ ...scaler.transform(p), label: p.label }));
const original = new LogisticRegression({ epochs: 500 }).fit(scaled);

const json = modelToString(serializeLogistic(original, scaler));
console.log("Exported JSON (first 300 chars):");
console.log(json.slice(0, 300) + "...");

const { model, scaler: loadedScaler } = modelFromString(json);
const mismatches = raw.filter(
  (p) => model.predict(loadedScaler.transform(p)) !== original.predict(scaler.transform(p)),
).length;
console.log(`Reloaded model disagrees on ${mismatches}/${raw.length} training points.`);
