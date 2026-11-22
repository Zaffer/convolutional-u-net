// @ts-check
// A tiny U-net as pure functions. Every intermediate map and every gradient is kept, so the page
// can show how each neuron got its value and how each weight is about to move.
//
//   x (1×8×8) ─enc conv 3×3, ReLU─▶ e (2×8×8) ──────────────── skip (copy) ─────────────────┐
//                                     │ max-pool 2×2                                          ▼
//                                   p (2×4×4) ─mid conv 3×3, ReLU─▶ b (4×4×4) ─up-conv 2×2─▶ u (2×8×8)
//                                                                              [u, e] ─dec conv 3×3, ReLU─▶ d (2×8×8)
//                                                                                       d ─out conv 1×1, sigmoid─▶ y (1×8×8)
//
// As in real U-nets, the channels double where the resolution halves (the bottleneck has 4), and a
// learned up-convolution halves them again on the way back up.
//
// A map is a Float64Array of n×n values, row-major. A tensor is an array of maps (channels).
// A kernel is a Float64Array of ks×ks weights, indexed (dr + h) * ks + (dc + h) for offsets -h..h:
// cross-correlation with zero padding, as every deep learning library does it.

export const N = 8; // full resolution
export const M = N / 2; // after pooling
export const C = 2; // channels at full resolution
export const CB = 2 * C; // channels in the bottleneck
/** channels in each map */
export const CH = /** @type {const} */ ({ x: 1, t: 1, e: C, p: C, b: CB, u: C, d: C, y: 1 });

/** @typedef {Float64Array} Map */
/** @typedef {{ w: Float64Array[][], b: number[] }} Layer  w[out][in] is one kernel */
/** @typedef {{ enc: Layer, mid: Layer, up: Layer, dec: Layer, out: Layer }} Params */
/** @typedef {'enc' | 'mid' | 'up' | 'dec' | 'out'} LayerId */

/**
 * The five layers with weights. `from` names the maps each one reads, in channel order; `to` the
 * map it writes; `n` the size of the map it writes. The decoder reads the up-convolved map u and,
 * through the skip connection, e. The up-conv is a transposed convolution: each bottleneck neuron
 * writes its value, times a 2×2 kernel, into a 2×2 block of u.
 */
export const LAYERS = /** @type {const} */ ({
  enc: { name: 'encoder conv', from: ['x'], to: 'e', ks: 3, n: N, act: 'ReLU', ins: 1, outs: C },
  mid: { name: 'bottleneck conv', from: ['p'], to: 'b', ks: 3, n: M, act: 'ReLU', ins: C, outs: CB },
  up: { name: 'up-conv', from: ['b'], to: 'u', ks: 2, n: N, act: 'none', ins: CB, outs: C },
  dec: { name: 'decoder conv', from: ['u', 'e'], to: 'd', ks: 3, n: N, act: 'ReLU', ins: 2 * C, outs: C },
  out: { name: 'output conv', from: ['d'], to: 'y', ks: 1, n: N, act: 'sigmoid', ins: C, outs: 1 },
});
/** @type {LayerId[]} */
export const LAYER_IDS = ['enc', 'mid', 'up', 'dec', 'out'];

// ── a seeded random source, so a test set is the same every time ──

/** @param {number} seed */
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** He initialisation: weights ~ N(0, 2 / fan-in), small positive biases so no ReLU starts dead.
 *  (An up-conv output neuron hears from one weight per input channel, so its fan-in is just ins.) */
export function init(seed = (Math.random() * 2 ** 32) >>> 0) {
  const r = rng(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
  /** @returns {Layer} */
  const layer = (/** @type {LayerId} */ id) => {
    const L = LAYERS[id];
    const sd = Math.sqrt(2 / (id === 'up' ? L.ins : L.ins * L.ks * L.ks));
    return {
      w: Array.from({ length: L.outs }, () =>
        Array.from({ length: L.ins }, () => Float64Array.from({ length: L.ks * L.ks }, () => gauss() * sd)),
      ),
      b: Array.from({ length: L.outs }, () => (id === 'out' || id === 'up' ? 0 : 0.1)),
    };
  };
  return /** @type {Params} */ (Object.fromEntries(LAYER_IDS.map(id => [id, layer(id)])));
}

// ── the operations, each with its backward pass ──

/**
 * @param {Map[]} input @param {number} n @param {Layer} L @param {number} ks
 * @returns {Map[]} pre-activations, one map per output channel
 */
export function conv(input, n, L, ks) {
  const h = ks >> 1;
  return L.w.map((row, o) => {
    const out = new Float64Array(n * n).fill(L.b[o]);
    row.forEach((k, i) => {
      const src = input[i];
      for (let r = 0; r < n; r++)
        for (let c = 0; c < n; c++) {
          let s = 0;
          for (let dr = -h; dr <= h; dr++) {
            const rr = r + dr;
            if (rr < 0 || rr >= n) continue;
            for (let dc = -h; dc <= h; dc++) {
              const cc = c + dc;
              if (cc < 0 || cc >= n) continue;
              s += src[rr * n + cc] * k[(dr + h) * ks + (dc + h)];
            }
          }
          out[r * n + c] += s;
        }
    });
    return out;
  });
}

/**
 * Given ∂L/∂(pre-activation) for every output neuron, the gradient of every weight and bias and of
 * every input neuron. A weight's gradient sums over every position it was used at.
 * @param {Map[]} input @param {number} n @param {Layer} L @param {number} ks @param {Map[]} dPre
 * @returns {{ grad: Layer, dIn: Map[] }}
 */
export function convBack(input, n, L, ks, dPre) {
  const h = ks >> 1;
  const dIn = input.map(() => new Float64Array(n * n));
  const w = L.w.map((row, o) =>
    row.map((k, i) => {
      const g = new Float64Array(ks * ks);
      const src = input[i];
      const back = dIn[i];
      for (let r = 0; r < n; r++)
        for (let c = 0; c < n; c++) {
          const d = dPre[o][r * n + c];
          if (d === 0) continue;
          for (let dr = -h; dr <= h; dr++) {
            const rr = r + dr;
            if (rr < 0 || rr >= n) continue;
            for (let dc = -h; dc <= h; dc++) {
              const cc = c + dc;
              if (cc < 0 || cc >= n) continue;
              const j = (dr + h) * ks + (dc + h);
              g[j] += d * src[rr * n + cc];
              back[rr * n + cc] += d * k[j];
            }
          }
        }
      return g;
    }),
  );
  const b = dPre.map(m => m.reduce((a, v) => a + v, 0));
  return { grad: { w, b }, dIn };
}

/** @param {Map[]} t */
export const relu = t => t.map(m => m.map(v => (v > 0 ? v : 0)));
/** @param {Map[]} t */
export const sigmoid = t => t.map(m => m.map(v => 1 / (1 + Math.exp(-v))));
/** ReLU passes a gradient only where the neuron was active. @param {Map[]} dOut @param {Map[]} pre */
export const reluBack = (dOut, pre) => dOut.map((m, c) => m.map((v, j) => (pre[c][j] > 0 ? v : 0)));

/**
 * 2×2 max-pool: each output neuron keeps the largest of its four inputs, and remembers which one.
 * @param {Map[]} t @param {number} n  input size
 */
export function pool(t, n) {
  const m = n / 2;
  const out = t.map(() => new Float64Array(m * m));
  const arg = t.map(() => new Int32Array(m * m));
  t.forEach((src, c) => {
    for (let r = 0; r < m; r++)
      for (let q = 0; q < m; q++) {
        let best = -Infinity;
        let at = 0;
        for (const j of [2 * r * n + 2 * q, 2 * r * n + 2 * q + 1, (2 * r + 1) * n + 2 * q, (2 * r + 1) * n + 2 * q + 1])
          if (src[j] > best) {
            best = src[j];
            at = j;
          }
        out[c][r * m + q] = best;
        arg[c][r * m + q] = at;
      }
  });
  return { out, arg };
}
/** The gradient goes back only to the input that won the max. @param {Map[]} dOut @param {Int32Array[]} arg @param {number} n */
export const poolBack = (dOut, arg, n) =>
  dOut.map((m, c) => {
    const back = new Float64Array(n * n);
    m.forEach((v, j) => (back[arg[c][j]] += v));
    return back;
  });

/**
 * 2×2 up-convolution (a transposed convolution, stride 2): bottleneck neuron i at (R, Q) adds
 * value × w[o][i][a·2 + c] to u_o at (2R + a, 2Q + c). Each u neuron hears from exactly one
 * position of every bottleneck map, through one weight of each kernel.
 * @param {Map[]} input  CB maps of m×m @param {number} m @param {Layer} L
 */
export function upconv(input, m, L) {
  const n = 2 * m;
  return L.w.map((row, o) => {
    const out = new Float64Array(n * n).fill(L.b[o]);
    row.forEach((k, i) => {
      for (let R = 0; R < m; R++)
        for (let Q = 0; Q < m; Q++) {
          const v = input[i][R * m + Q];
          for (let a = 0; a < 2; a++) for (let c = 0; c < 2; c++) out[(2 * R + a) * n + 2 * Q + c] += v * k[a * 2 + c];
        }
    });
    return out;
  });
}
/** @param {Map[]} input @param {number} m @param {Layer} L @param {Map[]} dOut  ∂L/∂u @returns {{ grad: Layer, dIn: Map[] }} */
export function upconvBack(input, m, L, dOut) {
  const n = 2 * m;
  const dIn = input.map(() => new Float64Array(m * m));
  const w = L.w.map((row, o) =>
    row.map((k, i) => {
      const g = new Float64Array(4);
      for (let R = 0; R < m; R++)
        for (let Q = 0; Q < m; Q++)
          for (let a = 0; a < 2; a++)
            for (let c = 0; c < 2; c++) {
              const d = dOut[o][(2 * R + a) * n + 2 * Q + c];
              g[a * 2 + c] += d * input[i][R * m + Q];
              dIn[i][R * m + Q] += d * k[a * 2 + c];
            }
      return g;
    }),
  );
  return { grad: { w, b: dOut.map(g => g.reduce((s, v) => s + v, 0)) }, dIn };
}

// ── the whole network ──

/**
 * @typedef {object} Trace  every map of one forward pass
 * @property {Map[]} x  @property {Map[]} t  target
 * @property {Map[]} ePre @property {Map[]} e @property {Map[]} p @property {Int32Array[]} pArg
 * @property {Map[]} bPre @property {Map[]} b @property {Map[]} u @property {Map[]} skip  what the decoder got from e
 * @property {Map[]} dPre @property {Map[]} d @property {Map[]} yPre @property {Map[]} y
 * @property {number} loss  binary cross-entropy, mean per pixel
 * @property {{ skip: boolean, deep: boolean }} cut  which path into the decoder was cut
 */

/**
 * @param {Params} P @param {Map} x @param {Map} t
 * @param {{ skip?: boolean, deep?: boolean }} [opts]  false cuts that path: the decoder gets zeros from the
 *   skip connection (skip) or from the bottleneck (deep) instead
 * @returns {Trace}
 */
export function forward(P, x, t, { skip = true, deep = true } = {}) {
  const X = [x];
  const ePre = conv(X, N, P.enc, 3);
  const e = relu(ePre);
  const { out: p, arg: pArg } = pool(e, N);
  const bPre = conv(p, M, P.mid, 3);
  const b = relu(bPre);
  const zeros = (/** @type {Map[]} */ t) => t.map(m => new Float64Array(m.length));
  const u = deep ? upconv(b, M, P.up) : zeros(e);
  const sk = skip ? e : zeros(e);
  const dPre = conv([...u, ...sk], N, P.dec, 3);
  const d = relu(dPre);
  const yPre = conv(d, N, P.out, 1);
  const y = sigmoid(yPre);
  let loss = 0;
  y[0].forEach((v, j) => {
    const q = Math.min(Math.max(v, 1e-7), 1 - 1e-7);
    loss -= t[j] * Math.log(q) + (1 - t[j]) * Math.log(1 - q);
  });
  return { x: X, t: [t], ePre, e, p, pArg, bPre, b, u, skip: sk, dPre, d, yPre, y, loss: loss / (N * N), cut: { skip: !skip, deep: !deep } };
}

/**
 * @typedef {object} Grads  ∂L/∂ of every map and weight, L summed over the 64 pixels
 * @property {Map[]} y  ∂L/∂(pre-sigmoid) = y − t, the output error
 * @property {Map[]} d @property {Map[]} dPre @property {Map[]} u @property {Map[]} skip  through the skip connection
 * @property {Map[]} b @property {Map[]} bPre @property {Map[]} p
 * @property {Map[]} eFromPool @property {Map[]} e @property {Map[]} ePre
 * @property {Params} w  the weight gradients
 */

/** @param {Params} P @param {Trace} T @returns {Grads} */
export function backward(P, T) {
  const y = [T.y[0].map((v, j) => v - T.t[0][j])];
  const o = convBack(T.d, N, P.out, 1, y);
  const d = o.dIn;
  const dPre = reluBack(d, T.dPre);
  const dc = convBack([...T.u, ...T.skip], N, P.dec, 3, dPre);
  // a cut path carries no gradient back
  const u = dc.dIn.slice(0, C).map(m => (T.cut.deep ? m.map(() => 0) : m));
  const skip = dc.dIn.slice(C).map(m => (T.cut.skip ? m.map(() => 0) : m));
  const up = upconvBack(T.b, M, P.up, u);
  const b = up.dIn;
  const bPre = reluBack(b, T.bPre);
  const m = convBack(T.p, M, P.mid, 3, bPre);
  const p = m.dIn;
  const eFromPool = poolBack(p, T.pArg, N);
  // e feeds two places, the pool and the skip connection, so its gradient is the sum of both
  const e = eFromPool.map((g, c) => g.map((v, j) => v + skip[c][j]));
  const ePre = reluBack(e, T.ePre);
  const en = convBack(T.x, N, P.enc, 3, ePre);
  return { y, d, dPre, u, skip, b, bPre, p, eFromPool, e, ePre, w: { enc: en.grad, mid: m.grad, up: up.grad, dec: dc.grad, out: o.grad } };
}

/** One gradient-descent step: every weight w ← w − α·∂L/∂w. Returns new params. @param {Params} P @param {Params} G @param {number} alpha */
export function apply(P, G, alpha) {
  /** @returns {Layer} */
  const step = (/** @type {LayerId} */ id) => ({
    w: P[id].w.map((row, o) => row.map((k, i) => k.map((v, j) => v - alpha * G[id].w[o][i][j]))),
    b: P[id].b.map((v, o) => v - alpha * G[id].b[o]),
  });
  return /** @type {Params} */ (Object.fromEntries(LAYER_IDS.map(id => [id, step(id)])));
}

/** Fraction of pixels the network gets right (output rounded to 0 or 1). @param {Trace} T */
export const accuracy = T => T.y[0].reduce((a, v, j) => a + (Math.round(v) === T.t[0][j] ? 1 : 0), 0) / (N * N);
