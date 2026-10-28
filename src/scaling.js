// Feature scalers. Fit on training data only, then reuse the same scaler for
// test points and for the demo page's decision-boundary grid — refitting on
// test data is a classic leakage bug this module makes hard to commit.
export function fitStandardScaler(points) {
  if (points.length === 0) throw new Error("fitStandardScaler needs points");
  const dims = ["x", "y"];
  const mean = {};
  const std = {};
  for (const d of dims) {
    mean[d] = points.reduce((s, p) => s + p[d], 0) / points.length;
    const variance =
      points.reduce((s, p) => s + (p[d] - mean[d]) ** 2, 0) / points.length;
    std[d] = Math.sqrt(variance) || 1; // constant feature -> scale 1, no NaN
  }
  return {
    transform: (p) => ({ x: (p.x - mean.x) / std.x, y: (p.y - mean.y) / std.y }),
    inverse: (p) => ({ x: p.x * std.x + mean.x, y: p.y * std.y + mean.y }),
    mean,
    std,
  };
}
