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

// Two interleaving spirals: the hardest teaching case here — a linear model
// scores near chance while k-NN follows the arms. Angle grows with radius.
export function spirals({ n = 100, seed = 42, noise = 0.15 } = {}) {
  const rng = mulberry32(seed);
  const points = [];
  for (let i = 0; i < n; i++) {
    const label = i % 2;
    const t = (i / n) * 2 * Math.PI * 2; // two full turns, interleaved
    const r = 0.3 + t * 0.35;
    const angle = t + label * Math.PI;
    points.push({
      x: r * Math.cos(angle) + gaussian(rng) * noise,
      y: r * Math.sin(angle) + gaussian(rng) * noise,
      label,
    });
  }
  return points;
}

// Concentric rings: inner ring label 0, outer ring label 1. No single line
// separates them, so it is a clean non-linearity demo next to moons.
export function rings({ n = 100, seed = 42, noise = 0.15 } = {}) {
  const rng = mulberry32(seed);
  const points = [];
  for (let i = 0; i < n; i++) {
    const label = i % 2;
    const radius = label === 0 ? 1 : 2.5;
    const angle = rng() * 2 * Math.PI;
    points.push({
      x: radius * Math.cos(angle) + gaussian(rng) * noise,
      y: radius * Math.sin(angle) + gaussian(rng) * noise,
      label,
    });
  }
  return points;
}

// XOR quadrants: label is 1 exactly when x and y have opposite signs.
// The classic case a single linear boundary cannot solve at all.
export function xorQuadrants({ n = 100, seed = 42 } = {}) {
  const rng = mulberry32(seed);
  const points = [];
  for (let i = 0; i < n; i++) {
    const x = (rng() * 2 - 1) * 3;
    const y = (rng() * 2 - 1) * 3;
    points.push({ x, y, label: (x > 0) !== (y > 0) ? 1 : 0 });
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
