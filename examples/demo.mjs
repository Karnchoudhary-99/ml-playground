// Runnable tour of every model: node examples/demo.mjs
// Prints train/test metrics so the numbers in the README stay honest.
import { twoBlobs, moons, noisyLine, trainTestSplit } from "../src/datasets.js";
import { fitStandardScaler } from "../src/scaling.js";
import { KnnClassifier, KnnRegressor } from "../src/knn.js";
import { LogisticRegression } from "../src/logistic.js";
import { fitLinear } from "../src/linear.js";
import { accuracy, mse, r2 } from "../src/metrics.js";

function reportClassification(name, train, test, predict) {
  console.log(
    `${name}: train acc ${(accuracy(train, predict) * 100).toFixed(1)}%` +
      `, test acc ${(accuracy(test, predict) * 100).toFixed(1)}%`,
  );
}

for (const [datasetName, make] of [
  ["two blobs", () => twoBlobs({ n: 120, seed: 31 })],
  ["moons", () => moons({ n: 120, seed: 31 })],
]) {
  console.log(`\n== ${datasetName} (classification) ==`);
  const { train, test } = trainTestSplit(make(), { seed: 5 });
  const scaler = fitStandardScaler(train);
  const sTrain = train.map((p) => ({ ...scaler.transform(p), label: p.label }));
  const sTest = test.map((p) => ({ ...scaler.transform(p), label: p.label }));
  reportClassification(
    "k-NN (k=5)",
    sTrain,
    sTest,
    (p) => new KnnClassifier({ k: 5 }).fit(sTrain).predict(p),
  );
  const logreg = new LogisticRegression({ epochs: 800 }).fit(sTrain);
  reportClassification("logistic regression", sTrain, sTest, (p) =>
    logreg.predict(p),
  );
}

console.log("\n== noisy line (regression) ==");
{
  const { train, test } = trainTestSplit(noisyLine({ n: 80, seed: 3 }), {
    seed: 5,
  });
  const lin = fitLinear(train);
  const linPredict = (p) => lin.predict(p.x); // fitLinear predicts from x, metrics pass a point
  console.log(
    `linear regression: y = ${lin.a.toFixed(2)}x + ${lin.b.toFixed(2)}` +
      `, test MSE ${mse(test, linPredict).toFixed(3)}` +
      `, test R2 ${r2(test, linPredict).toFixed(3)}`,
  );
  const knn = new KnnRegressor({ k: 3 }).fit(train);
  console.log(
    `k-NN regressor (k=3): test MSE ${mse(test, (p) => knn.predict(p)).toFixed(3)}` +
      `, test R2 ${r2(test, (p) => knn.predict(p)).toFixed(3)}`,
  );
}
