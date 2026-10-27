// Tiny linear regression (y = a*x + b) via closed-form least squares.
// Kept dependency-free so it runs identically in Node and in the browser.
export function fitLinear(points) {
  if (!Array.isArray(points) || points.length < 2) {
    throw new Error("fitLinear needs at least two points");
  }
  const n = points.length;
  const meanX = points.reduce((s, p) => s + p.x, 0) / n;
  const meanY = points.reduce((s, p) => s + p.y, 0) / n;
  let num = 0;
  let den = 0;
  for (const p of points) {
    num += (p.x - meanX) * (p.y - meanY);
    den += (p.x - meanX) ** 2;
  }
  if (den === 0) throw new Error("fitLinear: all x values are identical");
  const a = num / den;
  const b = meanY - a * meanX;
  return { a, b, predict: (x) => a * x + b };
}
