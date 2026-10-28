// Deterministic synthetic datasets for the playground. Every generator takes
// a seed so the same preset always renders the same points, in Node and in
// the browser. Points are { x, y, label? } with label 0 or 1 for classifiers.
import { mulberry32, gaussian } from "./rng.js";

// Two Gaussian blobs, linearly separable-ish: the classic first demo.
export function twoBlobs({ n = 100, seed = 42, spread = 0.7 } = {}) {
  const rng = mulberry32(seed);
  const points = [];
  const centres = [
    { x: -2, y: -1, label: 0 },
    { x: 2, y: 1, label: 1 },
  ];
  for (let i = 0; i < n; i++) {
    const c = centres[i % 2];
    points.push({
      x: c.x + gaussian(rng) * spread,
      y: c.y + gaussian(rng) * spread,
      label: c.label,
    });
  }
  return points;
}

// Two interleaving half-moons: not linearly separable, so k-NN shines and
// plain logistic regression visibly struggles. Great teaching dataset.
export function moons({ n = 100, seed = 42, noise = 0.15 } = {}) {
  const rng = mulberry32(seed);
  const points = [];
  for (let i = 0; i < n; i++) {
    const label = i % 2;
    const angle = rng() * Math.PI;
    const x = Math.cos(angle) + (label === 1 ? 1 : 0);
    const y = Math.sin(angle) + (label === 1 ? -0.5 : 0);
    points.push({
      x: x + gaussian(rng) * noise,
      y: y + gaussian(rng) * noise,
      label,
    });
  }
  return points;
}

// Noisy straight line y = 2x + 1, for the regression models.
export function noisyLine({ n = 50, seed = 42, noise = 0.6 } = {}) {
  const rng = mulberry32(seed);
  const points = [];
  for (let i = 0; i < n; i++) {
    const x = -4 + (8 * i) / (n - 1);
    points.push({ x, y: 2 * x + 1 + gaussian(rng) * noise });
  }
  return points;
}

// Deterministic train/test split that keeps class balance for labelled data.
export function trainTestSplit(points, { testFraction = 0.25, seed = 99 } = {}) {
  const rng = mulberry32(seed);
  const shuffled = [...points];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  const testSize = Math.floor(points.length * testFraction);
  return {
    train: shuffled.slice(testSize),
    test: shuffled.slice(0, testSize),
  };
}
