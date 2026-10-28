// Binary logistic regression trained by batch gradient descent. Callers are
// expected to standardise features first (see scaling.js) — unscaled inputs
// with very different ranges make a single learning rate misbehave.
const sigmoid = (z) => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));

export class LogisticRegression {
  constructor({ learningRate = 0.1, epochs = 500, l2 = 0.01 } = {}) {
    this.learningRate = learningRate;
    this.epochs = epochs;
    this.l2 = l2;
    this.weights = [0, 0];
    this.bias = 0;
  }
  fit(train) {
    if (train.length === 0) throw new Error("logistic: no training points");
    const n = train.length;
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      let gwX = 0;
      let gwY = 0;
      let gb = 0;
      for (const p of train) {
        const err = this.predictProba(p) - p.label;
        gwX += err * p.x;
        gwY += err * p.y;
        gb += err;
      }
      this.weights[0] -=
        this.learningRate * (gwX / n + this.l2 * this.weights[0]);
      this.weights[1] -=
        this.learningRate * (gwY / n + this.l2 * this.weights[1]);
      this.bias -= this.learningRate * (gb / n);
    }
    return this;
  }
  predictProba(p) {
    return sigmoid(this.weights[0] * p.x + this.weights[1] * p.y + this.bias);
  }
  predict(p, threshold = 0.5) {
    return this.predictProba(p) >= threshold ? 1 : 0;
  }
}
