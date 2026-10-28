// Small deterministic PRNG (mulberry32) so dataset presets and demos are
// reproducible in Node and in the browser. No dependencies.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Standard-normal sample via Box-Muller, using the supplied uniform RNG.
export function gaussian(rng) {
  let u = 0;
  while (u === 0) u = rng(); // avoid log(0)
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
