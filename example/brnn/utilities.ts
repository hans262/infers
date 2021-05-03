/**
 * Returns an array of zeros
 */
export function zeros(size: number): Float32Array {
  return new Float32Array(size);
}

export function ones(size: number): Float32Array {
  return new Float32Array(size).fill(1);
}

/**
 * Returns a random float between given min and max bounds (inclusive)
 * @param min Minimum value of the ranfom float
 * @param max Maximum value of the random float
 */
export function randomFloat(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

/**
 * If you know what this is: https://en.wikipedia.org/wiki/Normal_distribution
 * @param mu
 * @param std
 */
export function randomN(mu: number, std: number): number {
  return mu + gaussRandom() * std;
}


/**
 * Complicated math. All you need to know is that it returns a random number.
 * More info: https://en.wikipedia.org/wiki/Normal_distribution
 */
export function gaussRandom(): number {
  if (gaussRandom.returnV) {
    gaussRandom.returnV = false;
    return gaussRandom.vVal;
  }
  const u = 2 * Math.random() - 1;
  const v = 2 * Math.random() - 1;
  const r = u * u + v * v;
  if (r === 0 || r > 1) {
    return gaussRandom();
  }
  const c = Math.sqrt((-2 * Math.log(r)) / r);
  gaussRandom.vVal = v * c; // cache this
  gaussRandom.returnV = true;
  return u * c;
}

gaussRandom.returnV = false;
gaussRandom.vVal = 0;