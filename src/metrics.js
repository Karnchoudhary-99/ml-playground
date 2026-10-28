// Evaluation metrics shared by the Node demo and the browser page.
export function accuracy(points, predict) {
  if (points.length === 0) throw new Error("accuracy needs points");
  const correct = points.filter((p) => predict(p) === p.label).length;
  return correct / points.length;
}

export function mse(points, predict) {
  if (points.length === 0) throw new Error("mse needs points");
  return (
    points.reduce((s, p) => s + (predict(p) - p.y) ** 2, 0) / points.length
  );
}

export function r2(points, predict) {
  if (points.length === 0) throw new Error("r2 needs points");
  const meanY = points.reduce((s, p) => s + p.y, 0) / points.length;
  const ssTot = points.reduce((s, p) => s + (p.y - meanY) ** 2, 0);
  if (ssTot === 0) return 1;
  const ssRes = points.reduce((s, p) => s + (p.y - predict(p)) ** 2, 0);
  return 1 - ssRes / ssTot;
}

// [[true-neg, false-pos], [false-neg, true-pos]] for binary labels 0/1.
export function confusionMatrix(points, predict) {
  let tn = 0;
  let fp = 0;
  let fn = 0;
  let tp = 0;
  for (const p of points) {
    const pred = predict(p);
    if (p.label === 1 && pred === 1) tp++;
    else if (p.label === 0 && pred === 0) tn++;
    else if (p.label === 0 && pred === 1) fp++;
    else fn++;
  }
  return [
    [tn, fp],
    [fn, tp],
  ];
}
