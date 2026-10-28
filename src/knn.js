// k-nearest neighbours, classifier and regressor. Distance-weighted voting
// is the default because it degrades gracefully as k grows; ties break
// towards the single nearest neighbour, never towards an arbitrary label.
function distance(a, b, metric) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  if (metric === "manhattan") return Math.abs(dx) + Math.abs(dy);
  return Math.hypot(dx, dy); // euclidean
}

function neighbours(train, point, k, metric) {
  if (train.length === 0) throw new Error("k-NN: no training points");
  if (k < 1) throw new Error("k-NN: k must be >= 1");
  const ranked = train
    .map((p) => ({ point: p, d: distance(p, point, metric) }))
    .sort((a, b) => a.d - b.d);
  return ranked.slice(0, Math.min(k, train.length));
}

export class KnnClassifier {
  constructor({ k = 5, metric = "euclidean", weights = "distance" } = {}) {
    this.k = k;
    this.metric = metric;
    this.weights = weights;
  }
  fit(train) {
    this.train = train;
    return this;
  }
  predict(point) {
    const near = neighbours(this.train, point, this.k, this.metric);
    const votes = new Map();
    for (const { point: p, d } of near) {
      const w = this.weights === "distance" ? 1 / (d + 1e-9) : 1;
      votes.set(p.label, (votes.get(p.label) ?? 0) + w);
    }
    // Highest vote wins; an exact tie falls back to the nearest neighbour.
    let best = near[0].point.label;
    let bestScore = -Infinity;
    for (const [label, score] of votes) {
      if (score > bestScore) {
        best = label;
        bestScore = score;
      }
    }
    return best;
  }
}

export class KnnRegressor {
  constructor({ k = 5, metric = "euclidean", weights = "distance" } = {}) {
    this.k = k;
    this.metric = metric;
    this.weights = weights;
  }
  fit(train) {
    this.train = train;
    return this;
  }
  predict(point) {
    const near = neighbours(this.train, point, this.k, this.metric);
    let num = 0;
    let den = 0;
    for (const { point: p, d } of near) {
      const w = this.weights === "distance" ? 1 / (d + 1e-9) : 1;
      num += w * p.y;
      den += w;
    }
    return num / den;
  }
}
