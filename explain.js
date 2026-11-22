// @ts-check
// What the inspector says: for any neuron or weight, the exact arithmetic that produced its value,
// its gradient, or its update, written out as plain text. Pure functions of the store's data.
import { N, M, C, CH, LAYERS } from './net.js';
import { mapName, SUB } from './view.js';

/** @typedef {import('./net.js').Params} Params */
/** @typedef {import('./net.js').Trace} Trace */
/** @typedef {import('./net.js').Grads} Grads */
/** @typedef {import('./net.js').LayerId} LayerId */
/** @typedef {import('./view.js').MapName} MapName */
/** @typedef {import('./view.js').Selection} Selection */
/** @typedef {import('./view.js').Highlight} Highlight */
/** @typedef {import('./view.js').Mark} Mark */

/**
 * One pass over one sample, in the order the step button walks it. `shows` is the map whose values
 * (forward) or gradients (backward) the stage computes; `weights` the layer whose weight gradients it
 * computes.
 */
export const STAGES = /** @type {const} */ ([
  { label: '⓪ load a sample: input x, target t', dir: 'fwd', shows: 'x' },
  { label: '① encoder conv 3×3 + ReLU → e', dir: 'fwd', shows: 'e' },
  { label: '② max-pool 2×2 → p', dir: 'fwd', shows: 'p' },
  { label: '③ bottleneck conv 3×3 + ReLU → b', dir: 'fwd', shows: 'b' },
  { label: '④ up-conv 2×2: 4 channels back to 2, at double resolution → u', dir: 'fwd', shows: 'u' },
  { label: '⑤ decoder conv 3×3 + ReLU, reading u and the skip copy of e → d', dir: 'fwd', shows: 'd' },
  { label: '⑥ output conv 1×1 + sigmoid → y, and the loss', dir: 'fwd', shows: 'y' },
  { label: '⑦ output error y − t, and the output weights’ gradients', dir: 'back', shows: 'y', weights: 'out' },
  { label: '⑧ back through the output conv → ∂L/∂d, and the decoder weights’ gradients', dir: 'back', shows: 'd', weights: 'dec' },
  { label: '⑨ back through the decoder conv → ∂L/∂u, along the skip connection, and the up-conv weights’ gradients', dir: 'back', shows: 'u', weights: 'up' },
  { label: '⑩ back through the up-conv → ∂L/∂b, and the bottleneck weights’ gradients', dir: 'back', shows: 'b', weights: 'mid' },
  { label: '⑪ back through the bottleneck conv → ∂L/∂p', dir: 'back', shows: 'p' },
  { label: '⑫ back through the pool, plus the skip → ∂L/∂e, and the encoder weights’ gradients', dir: 'back', shows: 'e', weights: 'enc' },
  { label: '⑬ update every weight: w ← w − α·∂L/∂w', dir: 'update' },
]);
export const LAST = STAGES.length - 1;

/** the stage that computes each map's value, and each map's gradient */
export const FWD = /** @type {Record<MapName, number>} */ ({ x: 0, t: 0, e: 1, p: 2, b: 3, u: 4, d: 5, y: 6 });
export const BACK = /** @type {Record<MapName, number>} */ ({ x: Infinity, t: Infinity, y: 7, d: 8, u: 9, b: 10, p: 11, e: 12 });
/** the stage that computes each layer's weight gradients */
export const WGRAD = /** @type {Record<LayerId, number>} */ ({ out: 7, dec: 8, up: 9, mid: 10, enc: 12 });

/**
 * @typedef {object} Ctx
 * @property {Params} P  the weights now
 * @property {Params | null} before  the weights before the last update, once stage ⑬ has run
 * @property {Trace | null} T @property {Grads | null} G
 * @property {number} stage  the last stage done, -1 for none
 * @property {'value' | 'grad'} mode
 * @property {number} alpha
 */

const size = (/** @type {MapName} */ m) => (m === 'p' || m === 'b' ? M : N);
const v3 = (/** @type {number} */ v) => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(3);
const f4 = (/** @type {number} */ v) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(4);
const u3 = (/** @type {number} */ v) => (v < 0 ? '−' : ' ') + Math.abs(v).toFixed(3);
const at = (/** @type {number} */ r, /** @type {number} */ c) => `(${r},${c})`;
const plural = (/** @type {number} */ n, /** @type {string} */ w) => `${n} ${w}${n === 1 ? '' : 's'}`;

/** the maps a layer reads, in its input-channel order: [map, channel] */
const inputsOf = (/** @type {LayerId} */ id) => LAYERS[id].from.flatMap(m => Array.from({ length: CH[m] }, (_, ch) => /** @type {[MapName, number]} */ ([m, ch])));
const layerWriting = (/** @type {MapName} */ m) => /** @type {LayerId | undefined} */ (Object.entries(LAYERS).find(([, L]) => L.to === m)?.[0]);
/** what the dec layer's input channel i actually received (the cut paths give zeros) */
const received = (/** @type {Trace} */ T, /** @type {LayerId} */ id) => (id === 'dec' ? [...T.u, ...T.skip] : id === 'enc' ? T.x : id === 'mid' ? T.p : id === 'up' ? T.b : T.d);
/** ∂L/∂(pre-activation) of a layer's outputs (the up-conv has no activation: its δ is ∂L/∂u itself) */
const dPreOf = (/** @type {Grads} */ G, /** @type {LayerId} */ id) =>
  id === 'out' ? G.y : id === 'dec' ? G.dPre : id === 'up' ? G.u : id === 'mid' ? G.bPre : G.ePre;
const corner = (/** @type {number} */ a, /** @type {number} */ c) => `${a ? 'bottom' : 'top'}-${c ? 'right' : 'left'}`;
const wName = (/** @type {LayerId} */ id, /** @type {number} */ o, /** @type {number} */ i) => {
  const [m, ch] = inputsOf(id)[i];
  return `w[${mapName(LAYERS[id].to, o)}←${mapName(m, ch)}]`;
};
/** A weight's or bias's short name, for the editor. @param {Exclude<Selection, { kind: 'neuron' }>} sel */
export const paramLabel = sel => {
  const L = LAYERS[sel.layer];
  return sel.kind === 'bias'
    ? `bias b[${mapName(L.to, sel.o)}]`
    : `${wName(sel.layer, sel.o, sel.i)}, kernel row ${Math.floor(sel.k / L.ks)}, column ${sel.k % L.ks}`;
};
/** @param {MapName} m @param {Trace} T */
const fwdMap = (m, T) => /** @type {Float64Array[]} */ (T[m]);

// ── the text ──

/** @param {Selection | null} sel @param {Ctx} ctx @returns {string} */
export function explain(sel, ctx) {
  if (!sel) return 'Tap any neuron in the network, or any weight, to see how it got its value.';
  if (sel.kind !== 'neuron') return explainWeight(sel, ctx);
  const { T, G, stage, mode } = ctx;
  const { map: m, ch, r, c } = sel;
  const head = `${mapName(m, ch)} at row ${r}, column ${c}  (${NAMES[m]})`;
  if (!T || stage < FWD[m]) return `${head}\n\nnot computed yet: stage ${STAGES[FWD[m]].label.slice(0, 1)} computes it.`;
  if (mode === 'grad') {
    if (m === 'x' || m === 't') return `${head}\n\nThis is data, not a neuron: nothing upstream to train, so no gradient.`;
    if (!G || stage < BACK[m]) return `${head}\n\nits gradient is not computed yet: the backward pass reaches it at stage ${STAGES[BACK[m]].label.slice(0, 1)}.`;
    return `${head}\n\n${gradText(m, ch, r, c, ctx)}`;
  }
  return `${head}\n\n${valueText(m, ch, r, c, ctx)}`;
}

const NAMES = /** @type {Record<MapName, string>} */ ({
  x: 'input pixel', t: 'target pixel', e: 'encoder conv 3×3, ReLU', p: 'max-pool 2×2', b: 'bottleneck conv 3×3, ReLU',
  u: 'up-conv 2×2', d: 'decoder conv 3×3, ReLU', y: 'output conv 1×1, sigmoid',
});

/** How the forward pass computed this neuron. */
function valueText(/** @type {MapName} */ m, /** @type {number} */ ch, /** @type {number} */ r, /** @type {number} */ c, /** @type {Ctx} */ ctx) {
  const T = /** @type {Trace} */ (ctx.T);
  const n = size(m);
  const j = r * n + c;
  if (m === 'x') return `input pixel = ${T.x[0][j]}\nthe picture the network is given.`;
  if (m === 't') return `target pixel = ${T.t[0][j]}\nwhat the output y should be here.`;
  if (m === 'p') {
    const src = T.e[ch];
    const win = [[2 * r, 2 * c], [2 * r, 2 * c + 1], [2 * r + 1, 2 * c], [2 * r + 1, 2 * c + 1]];
    const best = T.pArg[ch][j];
    const cell = (/** @type {number[]} */ [rr, cc]) => {
      const k = rr * N + cc;
      return k === best ? `[${src[k].toFixed(3)}]` : ` ${src[k].toFixed(3)} `;
    };
    return [
      `the largest of these 4 neurons of e${SUB[ch]}`,
      `(rows ${2 * r}–${2 * r + 1}, columns ${2 * c}–${2 * c + 1}):`,
      '',
      `  ${cell(win[0])} ${cell(win[1])}`,
      `  ${cell(win[2])} ${cell(win[3])}`,
      '',
      `= ${T.p[ch][j].toFixed(3)}`,
      '',
      'No weights here. Half the resolution, and the pool',
      'remembers which of the four won: only that one gets',
      'a gradient on the way back.',
    ].join('\n');
  }
  if (m === 'u') {
    const R = r >> 1;
    const Q = c >> 1;
    const a = r & 1;
    const cc = c & 1;
    if (T.cut.deep) return '✂️ the bottleneck path is cut: the decoder gets 0 here\ninstead of the up-conv’s output.';
    const lines = [
      `The up-conv works backwards from a conv: each bottleneck`,
      `position (${R},${Q}) writes into a 2×2 block of u, rows ${2 * R}–${2 * R + 1},`,
      `columns ${2 * Q}–${2 * Q + 1}. This neuron is the block’s ${corner(a, cc)},`,
      `so it gets one weight from each of the ${CH.b} bottleneck maps:`,
      '',
    ];
    let sum = 0;
    for (let i = 0; i < CH.b; i++) {
      const v = T.b[i][R * M + Q];
      const w = ctx.P.up.w[ch][i][a * 2 + cc];
      sum += v * w;
      lines.push(`  b${SUB[i]}${at(R, Q)} ${u3(v)} × ${wName('up', ch, i)} ${v3(w)} = ${v3(v * w)}`);
    }
    const b = ctx.P.up.b[ch];
    lines.push(
      '',
      `${`sum of the ${CH.b}`.padEnd(19)}${v3(sum)}`,
      `${`+ bias b[u${SUB[ch]}]`.padEnd(19)}${v3(b)}`,
      `${'= u'.padEnd(19)}${v3(sum + b)}   (no activation)`,
      '',
      `${CH.b} coarse channels in, ${C} fine ones out: the learned`,
      'weights decide what each corner of the block gets, so',
      'the bottleneck can say more than “something here”.',
    );
    if (ctx.stage === LAST) lines.push('', '(the weights have just been updated; these are the new ones.\nThe values on screen came from the old ones.)');
    return lines.join('\n');
  }
  return convText(/** @type {LayerId} */ (layerWriting(m)), ch, r, c, ctx);
}

/** A conv neuron: every window × kernel, the sum, the bias, the activation. */
function convText(/** @type {LayerId} */ id, /** @type {number} */ o, /** @type {number} */ r, /** @type {number} */ c, /** @type {Ctx} */ ctx) {
  const T = /** @type {Trace} */ (ctx.T);
  const L = LAYERS[id];
  const n = L.n;
  const h = L.ks >> 1;
  const inputs = inputsOf(id);
  const recv = received(T, id);
  const lines = [];
  /** @type {number[]} */
  const parts = [];
  if (L.ks === 1) {
    lines.push(`one weight per input map, at this same pixel:`, '');
    inputs.forEach(([m, ch], i) => {
      const x = recv[i][r * n + c];
      const w = ctx.P[id].w[o][i][0];
      parts.push(x * w);
      lines.push(`  ${mapName(m, ch)} ${u3(x)} × ${wName(id, o, i)} ${v3(w)} = ${v3(x * w)}`);
    });
  } else {
    lines.push(`a 3×3 window of each of its ${plural(inputs.length, 'input map')},`, 'times that map’s 3×3 kernel. · is padding (0).');
    inputs.forEach(([m, ch], i) => {
      const k = ctx.P[id].w[o][i];
      const src = recv[i];
      let s = 0;
      const note =
        id === 'dec' ? (m === 'u' ? (T.cut.deep ? '  ✂️ cut: zeros' : '  bottleneck path') : T.cut.skip ? '  ✂️ cut: zeros' : '  skip connection') : '';
      lines.push('', `from ${mapName(m, ch)}${note}`, `   window                   kernel ${wName(id, o, i)}`);
      for (let dr = -h; dr <= h; dr++) {
        const win = [];
        const ker = [];
        for (let dc = -h; dc <= h; dc++) {
          const rr = r + dr;
          const cc = c + dc;
          const kv = k[(dr + h) * L.ks + dc + h];
          ker.push(v3(kv));
          if (rr < 0 || rr >= n || cc < 0 || cc >= n) win.push('     ·');
          else {
            win.push(u3(src[rr * n + cc]));
            s += src[rr * n + cc] * kv;
          }
        }
        lines.push(`  ${win.join(' ')}   ${ker.join(' ')}`);
      }
      lines.push(`  Σ window × kernel = ${v3(s)}`);
      parts.push(s);
    });
  }
  const sum = parts.reduce((a, b) => a + b, 0);
  const b = ctx.P[id].b[o];
  const pre = sum + b;
  const out = /** @type {Float64Array[]} */ (T[L.to])[o][r * n + c];
  // after an update the forward pass on screen used the old weights; say so rather than show a mismatch
  const stale = ctx.stage === LAST ? '\n(the weights have just been updated; these are the new ones.\nThe values on screen came from the old ones.)' : '';
  lines.push(
    '',
    `${`sum of the ${parts.length}`.padEnd(19)}${v3(sum)}`,
    `${`+ bias b[${mapName(L.to, o)}]`.padEnd(19)}${v3(b)}`,
    `${'= pre-activation'.padEnd(19)}${v3(pre)}`,
    L.act === 'ReLU' ? `ReLU: max(0, ${v3(pre)}) = ${Math.max(0, pre).toFixed(3)}` : `sigmoid: 1 / (1 + e^${pre < 0 ? '+' : '−'}${Math.abs(pre).toFixed(3)}) = ${(1 / (1 + Math.exp(-pre))).toFixed(3)}`,
  );
  if (stale) lines.push(stale);
  else if (Math.abs(out - (L.act === 'ReLU' ? Math.max(0, pre) : 1 / (1 + Math.exp(-pre)))) > 1e-9) lines.push('', '(mismatch: the weights changed since this pass)');
  if (L.to === 'y') {
    const t = T.t[0][r * n + c];
    lines.push('', `target t = ${t}: ${Math.round(out) === t ? '✅ right' : '❌ wrong'} (rounds to ${Math.round(out)})`);
  }
  return lines.join('\n');
}

/**
 * The terms of ∂L/∂(an input neuron of layer id): every output neuron whose window covers it,
 * δ of that neuron × the weight it reads this one with.
 * @param {LayerId} id @param {number} i  input channel @param {number} r @param {number} c @param {Ctx} ctx
 */
function backTerms(id, i, r, c, ctx) {
  const L = LAYERS[id];
  const n = L.n;
  const h = L.ks >> 1;
  const dPre = dPreOf(/** @type {Grads} */ (ctx.G), id);
  const terms = [];
  for (let o = 0; o < L.outs; o++)
    for (let dr = -h; dr <= h; dr++)
      for (let dc = -h; dc <= h; dc++) {
        const R = r - dr;
        const Q = c - dc;
        if (R < 0 || R >= n || Q < 0 || Q >= n) continue;
        const d = dPre[o][R * n + Q];
        const w = ctx.P[id].w[o][i][(dr + h) * L.ks + dc + h];
        terms.push({ who: `${mapName(L.to, o)}${at(R, Q)}`, d, w, v: d * w });
      }
  return terms;
}
/** @param {{ who: string, d: number, w: number, v: number }[]} terms @param {number} [show] */
function termLines(terms, show = 8) {
  const live = terms.filter(t => t.v !== 0).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
  const lines = live.slice(0, show).map(t => `  ${t.who.padEnd(8)} ${v3(t.d)} × ${v3(t.w)} = ${v3(t.v)}`);
  const zero = terms.length - live.length;
  if (live.length > show) lines.push(`  … ${plural(live.length - show, 'smaller term')}`);
  if (zero) lines.push(`  (${plural(zero, 'term')} are 0: those neurons’ ReLU was off)`);
  return lines;
}
/** @param {number} g @param {number} pre @param {string} name */
const reluGate = (g, pre, name) =>
  pre > 0
    ? `ReLU was on (pre-activation ${v3(pre)} > 0), so it passes\nthe gradient through: ∂L/∂pre = ${v3(g)}`
    : `ReLU was off (pre-activation ${v3(pre)} ≤ 0), so the gradient\nstops here: ∂L/∂pre = 0. Nothing upstream of ${name}\nlearns from this pixel through this neuron.`;

/** How the backward pass computed this neuron's gradient. */
function gradText(/** @type {MapName} */ m, /** @type {number} */ ch, /** @type {number} */ r, /** @type {number} */ c, /** @type {Ctx} */ ctx) {
  const T = /** @type {Trace} */ (ctx.T);
  const G = /** @type {Grads} */ (ctx.G);
  const n = size(m);
  const j = r * n + c;
  const intro = '∂L/∂ of a neuron: how much the loss L would change\nif this neuron’s value went up a little. The update\npushes every weight the way that lowers L.\n';
  if (m === 'y') {
    const y = T.y[0][j];
    const t = T.t[0][j];
    const q = Math.min(Math.max(y, 1e-7), 1 - 1e-7);
    const loss = -(t * Math.log(q) + (1 - t) * Math.log(1 - q));
    return [
      intro,
      `output y = ${y.toFixed(3)}, target t = ${t}`,
      `this pixel’s loss = −[t·ln y + (1−t)·ln(1−y)] = ${loss.toFixed(3)}`,
      '',
      'With a sigmoid and this loss, the gradient at the',
      'pre-activation is simply the error:',
      `  δ = y − t = ${y.toFixed(3)} − ${t} = ${v3(G.y[0][j])}`,
      '',
      G.y[0][j] < 0 ? 'Negative: the output is too low; the update raises it.' : 'Positive: the output is too high; the update lowers it.',
      'Every gradient upstream starts from these 64 errors.',
    ].join('\n');
  }
  if (m === 'd') {
    const terms = backTerms('out', ch, r, c, ctx);
    return [intro, `d${SUB[ch]} feeds only y at this pixel, through ${wName('out', 0, ch)}:`, ...termLines(terms), `∂L/∂d${SUB[ch]} = ${v3(G.d[ch][j])}`, '', reluGate(G.dPre[ch][j], T.dPre[ch][j], `d${SUB[ch]}`)].join('\n');
  }
  if (m === 'u') {
    if (T.cut.deep) return `${intro}\n✂️ the bottleneck path is cut: u feeds nothing, so ∂L/∂u = 0.`;
    const terms = backTerms('dec', ch, r, c, ctx);
    return [
      intro,
      'u feeds every decoder neuron whose 3×3 window covers it.',
      'each term: δ of that d neuron × the weight it reads u with',
      ...termLines(terms),
      `∂L/∂u${SUB[ch]} = ${v3(G.u[ch][j])}`,
      '(the up-conv has no activation, so this is also its δ:',
      'the up-conv weights’ gradients are built from it)',
    ].join('\n');
  }
  if (m === 'b') {
    const terms = [];
    for (let o = 0; o < C; o++)
      for (let a = 0; a < 2; a++)
        for (let cc = 0; cc < 2; cc++) {
          const d = G.u[o][(2 * r + a) * N + 2 * c + cc];
          const w = ctx.P.up.w[o][ch][a * 2 + cc];
          terms.push({ who: `u${SUB[o]}${at(2 * r + a, 2 * c + cc)}`, d, w, v: d * w });
        }
    return [
      intro,
      `b${SUB[ch]} wrote into a 2×2 block of each u map, through`,
      `one up-conv weight per neuron: ${terms.length} terms.`,
      'each term: ∂L/∂ of that u neuron × the weight between them',
      ...termLines(terms),
      `∂L/∂b${SUB[ch]} = ${v3(G.b[ch][j])}`,
      '',
      reluGate(G.bPre[ch][j], T.bPre[ch][j], `b${SUB[ch]}`),
    ].join('\n');
  }
  if (m === 'p') {
    const terms = backTerms('mid', ch, r, c, ctx);
    return [
      intro,
      'p feeds every bottleneck neuron whose 3×3 window covers it.',
      'each term: δ of that b neuron × the weight it reads p with',
      ...termLines(terms),
      `∂L/∂p${SUB[ch]} = ${v3(G.p[ch][j])}`,
      '(a pool has no activation, so nothing gates it)',
    ].join('\n');
  }
  // e: two roads back, through the pool and through the skip connection
  const R = r >> 1;
  const Q = c >> 1;
  const won = T.pArg[ch][R * M + Q] === j;
  const skipTerms = T.cut.skip ? [] : backTerms('dec', C + ch, r, c, ctx);
  return [
    intro,
    `e${SUB[ch]} feeds two places, and its gradient adds both:`,
    '',
    `1. the pool: p${SUB[ch]}${at(R, Q)} took the max of its 2×2 block,`,
    won ? `   and this neuron won it, so it gets all of ∂L/∂p${SUB[ch]}${at(R, Q)} = ${v3(G.eFromPool[ch][j])}` : `   and this neuron did not win it, so it gets 0`,
    '',
    T.cut.skip ? '2. the skip connection: ✂️ cut, so 0' : '2. the skip connection: decoder neurons read it directly.\n   each term: δ of that d neuron × its weight',
    ...(T.cut.skip ? [] : termLines(skipTerms, 6)),
    T.cut.skip ? '' : `   from the skip: ${v3(G.skip[ch][j])}`,
    '',
    `∂L/∂e${SUB[ch]} = ${v3(G.eFromPool[ch][j])} ${G.skip[ch][j] < 0 ? '−' : '+'} ${Math.abs(G.skip[ch][j]).toFixed(3)} = ${v3(G.e[ch][j])}`,
    '',
    reluGate(G.ePre[ch][j], T.ePre[ch][j], `e${SUB[ch]}`),
  ].join('\n');
}

/** A weight or bias: what it does, its gradient summed over every position it is used at, its update. */
function explainWeight(/** @type {Exclude<Selection, { kind: 'neuron' }>} */ sel, /** @type {Ctx} */ ctx) {
  const { T, G, stage, alpha } = ctx;
  const id = sel.layer;
  const L = LAYERS[id];
  const n = L.n;
  const h = L.ks >> 1;
  const o = sel.o;
  const out = mapName(L.to, o);
  const isBias = sel.kind === 'bias';
  const i = isBias ? 0 : sel.i;
  const k = isBias ? 0 : sel.k;
  const dr = Math.floor(k / L.ks) - h;
  const dc = (k % L.ks) - h;
  const isUp = id === 'up';
  const name = isBias ? `b[${out}]` : `${wName(id, o, i)}, kernel row ${Math.floor(k / L.ks)}, column ${k % L.ks}`;
  const now = isBias ? ctx.P[id].b[o] : ctx.P[id].w[o][i][k];
  const [im, ich] = inputsOf(id)[i];
  const lines = [`${name}  (${id}: ${L.name})`, ''];
  const where = isBias
    ? `added to every one of the ${n * n} ${out} neurons${L.act === 'none' ? '' : ` before the ${L.act}`}.`
    : isUp
      ? `carries b${SUB[i]} into the ${corner(Math.floor(k / 2), k % 2)} neuron of each 2×2 block\nof ${out}: one weight, shared by all ${M * M} blocks.`
      : L.ks === 1
      ? `multiplies ${mapName(im, ich)} at each pixel, into ${out} at that pixel.`
      : `used by every one of the ${n * n} ${out} neurons: each multiplies\nthe ${mapName(im, ich)} neuron ${offset(dr, dc)} of itself by it.\nOne weight, shared across the whole image.`;
  lines.push(where, '');
  const updated = stage === LAST && ctx.before;
  const was = updated ? (isBias ? /** @type {Params} */ (ctx.before)[id].b[o] : /** @type {Params} */ (ctx.before)[id].w[o][i][k]) : now;
  if (!T || !G || stage < WGRAD[id]) {
    lines.push(`value ${v3(now)}`, '', `its gradient is not computed yet: stage ${STAGES[WGRAD[id]].label.slice(0, 1)} computes it.`);
    return lines.join('\n');
  }
  const dPre = dPreOf(G, id);
  const src = received(T, id)[i];
  const terms = [];
  for (let r = 0; r < n; r++)
    for (let c = 0; c < n; c++) {
      const d = dPre[o][r * n + c];
      if (isBias) {
        terms.push({ who: `${out}${at(r, c)}`, d, w: 1, v: d });
        continue;
      }
      if (isUp) {
        // only the block corner this weight writes to; the input is the block's bottleneck neuron
        if (r % 2 !== Math.floor(k / 2) || c % 2 !== k % 2) continue;
        const x = src[(r >> 1) * M + (c >> 1)];
        terms.push({ who: `${out}${at(r, c)}`, d, w: x, v: d * x });
        continue;
      }
      const rr = r + dr;
      const cc = c + dc;
      if (rr < 0 || rr >= n || cc < 0 || cc >= n) continue;
      terms.push({ who: `${out}${at(r, c)}`, d, w: src[rr * n + cc], v: d * src[rr * n + cc] });
    }
  const g = isBias ? G.w[id].b[o] : G.w[id].w[o][i][k];
  lines.push(
    isBias
      ? `∂L/∂b = Σ over every ${out} neuron of its δ (∂L/∂pre):`
      : isUp
        ? `∂L/∂w = Σ over the ${M * M} blocks of\n  ∂L/∂ of the ${out} neuron it wrote × the ${mapName(im, ich)} value it carried:`
        : `∂L/∂w = Σ over every position of\n  δ of the ${out} neuron there × the ${mapName(im, ich)} value it read:`,
    isBias ? '' : `  ${'neuron'.padEnd(8)} ${'δ'.padStart(6)}   ${'input'.padStart(6)}`,
    ...termLines(terms, 6).map(l => l.replace('those neurons’ ReLU was off', 'ReLU off or input 0')),
    `total ${isBias ? '∂L/∂b' : '∂L/∂w'} = ${v3(g)}`,
    '',
    `update: w − α·∂L/∂w = ${f4(was)} − ${alpha} × ${g < 0 ? `(${v3(g)})` : g.toFixed(3)} = ${f4(was - alpha * g)}`,
    updated ? `✅ applied: it was ${f4(was)}, it is now ${f4(now)}` : 'applied at stage ⑬',
  );
  return lines.join('\n');
}
/** @param {number} dr @param {number} dc */
function offset(dr, dc) {
  if (!dr && !dc) return 'at the same position as';
  const v = dr < 0 ? 'above' : dr > 0 ? 'below' : '';
  const hz = dc < 0 ? 'left' : dc > 0 ? 'right' : '';
  return `${[v, hz].filter(Boolean).join('-')}`;
}

// ── the highlight: what the selected thing is connected to ──

/**
 * @param {Selection | null} sel @param {Ctx} ctx
 * @param {boolean} field  also mark every input pixel that can reach the selected neuron, coloured by path
 * @returns {Highlight}
 */
export function highlight(sel, ctx, field) {
  /** @type {Highlight} */
  const h = { cells: [], maps: [], kernels: [] };
  if (!sel) return h;
  if (sel.kind !== 'neuron') {
    const L = LAYERS[sel.layer];
    h.maps.push(`${L.to}${sel.o}`);
    if (sel.kind === 'weight') {
      const [m, ch] = inputsOf(sel.layer)[sel.i];
      h.maps.push(`${m}${ch}`);
      h.kernels.push({ key: `${sel.layer} ${sel.o} ${sel.i}`, mark: 'in' });
      h.cells.push({ key: `${sel.layer} ${sel.o} ${sel.i}`, j: sel.k, mark: 'sel' });
    } else h.kernels.push({ key: `${sel.layer} ${sel.o} b`, mark: 'sel' });
    return h;
  }
  const { map: m, ch, r, c } = sel;
  const n = size(m);
  if (field) h.cells.push(...receptiveField(m, ch, r, c, ctx.T?.cut ?? { skip: false, deep: false }));
  h.cells.push({ key: `${m}${ch}`, j: r * n + c, mark: 'sel' });
  /** @param {MapName} mm @param {number} cc @param {number} rr0 @param {number} cc0 @param {number} nn @param {number} hh @param {Mark} mark */
  const window = (mm, cc, rr0, cc0, nn, hh, mark) => {
    for (let dr = -hh; dr <= hh; dr++)
      for (let dc = -hh; dc <= hh; dc++) {
        const rr = rr0 + dr;
        const q = cc0 + dc;
        if (rr >= 0 && rr < nn && q >= 0 && q < nn) h.cells.push({ key: `${mm}${cc}`, j: rr * nn + q, mark });
      }
  };
  if (ctx.mode === 'value') {
    const id = layerWriting(m);
    if (m === 'u') {
      for (let i = 0; i < CH.b; i++) {
        h.cells.push({ key: `b${i}`, j: (r >> 1) * M + (c >> 1), mark: 'in' });
        h.kernels.push({ key: `up ${ch} ${i}`, mark: 'in' });
        h.cells.push({ key: `up ${ch} ${i}`, j: (r & 1) * 2 + (c & 1), mark: 'sel' });
      }
      h.kernels.push({ key: `up ${ch} b`, mark: 'in' });
    } else if (id) {
      const L = LAYERS[id];
      inputsOf(id).forEach(([im, ich], i) => {
        window(im, ich, r, c, L.n, L.ks >> 1, 'in');
        h.kernels.push({ key: `${id} ${ch} ${i}`, mark: 'in' });
      });
      h.kernels.push({ key: `${id} ${ch} b`, mark: 'in' });
    } else if (m === 'p') {
      for (const [rr, q] of [[0, 0], [0, 1], [1, 0], [1, 1]]) {
        const j = (2 * r + rr) * N + 2 * c + q;
        h.cells.push({ key: `e${ch}`, j, mark: ctx.T && ctx.T.pArg[ch][r * M + c] === j ? 'max' : 'in' });
      }
    }
  } else {
    // backward: the neurons this one feeds, whose gradients flow back into it
    if (m === 'y') h.cells.push({ key: 't0', j: r * N + c, mark: 'in' });
    if (m === 'd') {
      h.cells.push({ key: 'y0', j: r * N + c, mark: 'in' });
      h.kernels.push({ key: `out 0 ${ch}`, mark: 'in' });
    }
    if (m === 'u' || m === 'e') {
      for (let o = 0; o < C; o++) window('d', o, r, c, N, 1, 'in');
      for (let o = 0; o < C; o++) h.kernels.push({ key: `dec ${o} ${m === 'u' ? ch : C + ch}`, mark: 'in' });
    }
    if (m === 'e') h.cells.push({ key: `p${ch}`, j: (r >> 1) * M + (c >> 1), mark: 'in' });
    if (m === 'b')
      for (let o = 0; o < C; o++) {
        for (const [rr, q] of [[0, 0], [0, 1], [1, 0], [1, 1]]) h.cells.push({ key: `u${o}`, j: (2 * r + rr) * N + 2 * c + q, mark: 'in' });
        h.kernels.push({ key: `up ${o} ${ch}`, mark: 'in' });
      }
    if (m === 'p') {
      for (let o = 0; o < CH.b; o++) window('b', o, r, c, M, 1, 'in');
      for (let o = 0; o < CH.b; o++) h.kernels.push({ key: `mid ${o} ${ch}`, mark: 'in' });
    }
  }
  return h;
}

/**
 * Every neuron upstream that can affect the selected one, whatever the weights. From the decoder
 * on, each is tagged by the road it takes: through the bottleneck (deep), through the skip, or both.
 * @param {MapName} m @param {number} ch @param {number} r @param {number} c @param {{ skip: boolean, deep: boolean }} cut
 * @returns {{ key: string, j: number, mark: Mark }[]}
 */
export function receptiveField(m, ch, r, c, cut) {
  const DEEPB = 1;
  const SKIPB = 2;
  const PLAIN = 4;
  /** @type {Record<string, Map<number, number>>} */
  const f = {};
  const add = (/** @type {string} */ key, /** @type {number} */ j, /** @type {number} */ bits) => {
    const s = (f[key] ??= new Map());
    s.set(j, (s.get(j) ?? 0) | bits);
  };
  const chans = (/** @type {MapName} */ mm) => Array.from({ length: CH[mm] }, (_, k) => k);
  add(`${m}${ch}`, r * size(m) + c, PLAIN);
  /** conv: every channel of `from` in the 3×3 (or 1×1) window of every marked neuron of `to` */
  const conv = (/** @type {MapName} */ to, /** @type {MapName} */ from, /** @type {number} */ hh, /** @type {(b: number) => number} */ tag) => {
    const n = size(to);
    for (const o of chans(to))
      for (const [j, bits] of f[`${to}${o}`] ?? []) {
        const r0 = Math.floor(j / n);
        const c0 = j % n;
        for (let dr = -hh; dr <= hh; dr++)
          for (let dc = -hh; dc <= hh; dc++) {
            const rr = r0 + dr;
            const q = c0 + dc;
            if (rr < 0 || rr >= n || q < 0 || q >= n) continue;
            for (const i of chans(from)) add(`${from}${i}`, rr * n + q, tag(bits));
          }
      }
  };
  const same = (/** @type {number} */ b) => b;
  conv('y', 'd', 0, same);
  if (!cut.deep) conv('d', 'u', 1, b => (b & PLAIN ? DEEPB : b));
  if (!cut.skip) conv('d', 'e', 1, b => (b & PLAIN ? SKIPB : b));
  // the up-conv: each u neuron hears from one position of every bottleneck map
  for (const o of chans('u')) for (const [j, bits] of f[`u${o}`] ?? []) for (const i of chans('b')) add(`b${i}`, (Math.floor(j / N) >> 1) * M + ((j % N) >> 1), bits);
  conv('b', 'p', 1, same);
  for (const i of chans('p'))
    for (const [j, bits] of f[`p${i}`] ?? []) {
      const r0 = Math.floor(j / M) * 2;
      const c0 = (j % M) * 2;
      for (const k of [0, 1, N, N + 1]) add(`e${i}`, r0 * N + c0 + k, bits);
    }
  conv('e', 'x', 1, same);
  /** @type {{ key: string, j: number, mark: Mark }[]} */
  const out = [];
  for (const [key, s] of Object.entries(f))
    for (const [j, bits] of s) {
      if (key === `${m}${ch}` && j === r * size(m) + c) continue;
      out.push({ key, j, mark: bits & PLAIN ? 'plain' : bits === (DEEPB | SKIPB) ? 'both' : bits & DEEPB ? 'deep' : 'skip' });
    }
  return out;
}
