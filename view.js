// @ts-check
// The drawings: plain imperative SVG behind a narrow API, no signals in here. The store pushes
// values and highlights in; taps come out through one callback. Every SVG has a viewBox and a
// natural width, so CSS (or, with no CSS, nothing at all) decides its size.
import { N, M, C, CH, LAYERS, LAYER_IDS } from './net.js';

/** @typedef {import('./net.js').LayerId} LayerId */
/** @typedef {'x' | 't' | 'e' | 'p' | 'b' | 'u' | 'd' | 'y'} MapName */
/** @typedef {{ kind: 'neuron', map: MapName, ch: number, r: number, c: number }
 *   | { kind: 'weight', layer: LayerId, o: number, i: number, k: number }
 *   | { kind: 'bias', layer: LayerId, o: number }} Selection */
/** @typedef {{ values: Float64Array | null, scale: number }} Shown  null: not computed yet. A cell's colour is value / scale. */
/** @typedef {'sel' | 'in' | 'max' | 'deep' | 'skip' | 'both' | 'plain'} Mark */
/** @typedef {{ cells: { key: string, j: number, mark: Mark }[], maps: string[], kernels: { key: string, mark: Mark }[] }} Highlight
 *   key: a map key like 'e0', or a kernel key like 'dec 0 3' / 'dec 0 b' */

const NS = 'http://www.w3.org/2000/svg';
export const SUB = ['₀', '₁', '₂', '₃'];
export const POSITIVE = '#2a9df4';
export const NEGATIVE = '#f0605a';
export const DEEP = '#a970ff';
export const SKIP = '#e89a1c';
const GRID = 'rgba(128, 128, 128, 0.45)';

/** @param {string} tag @param {Record<string, string | number>} attrs @param {Element} [parent] */
function svg(tag, attrs, parent) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  parent?.append(node);
  return node;
}
/** @param {Element} parent @param {number} x @param {number} y @param {string} text @param {Record<string, string | number>} [attrs] */
const label = (parent, x, y, text, attrs = {}) => {
  const t = svg('text', { x, y, 'font-size': 8, fill: 'currentColor', ...attrs }, parent);
  t.textContent = text;
  return t;
};

/** One cell's colour: blue for positive, red for negative, opacity for size. */
function paint(/** @type {SVGRectElement} */ rect, /** @type {number | null} */ v, /** @type {number} */ scale) {
  const fill = v === null || v === 0 ? 'none' : v > 0 ? POSITIVE : NEGATIVE;
  const op = v === null ? '0' : Math.min(1, Math.abs(v) / (scale || 1)).toFixed(2);
  if (rect.getAttribute('fill') !== fill) rect.setAttribute('fill', fill);
  if (rect.getAttribute('fill-opacity') !== op) rect.setAttribute('fill-opacity', op);
}

/** A grid of n×n cells at (x, y). Returns the cell rects, row-major. */
function grid(/** @type {Element} */ parent, /** @type {number} */ x, /** @type {number} */ y, /** @type {number} */ n, /** @type {number} */ s, /** @type {Record<string, string>} */ data) {
  const g = svg('g', {}, parent);
  const cells = [];
  for (let r = 0; r < n; r++)
    for (let c = 0; c < n; c++) {
      const rect = /** @type {SVGRectElement} */ (svg('rect', { x: x + c * s, y: y + r * s, width: s, height: s, fill: 'none', stroke: GRID, 'stroke-width': 0.5, 'pointer-events': 'all' }, g)); // tappable even when blank
      for (const [k, v] of Object.entries(data)) rect.dataset[k] = v;
      rect.dataset.r = String(r);
      rect.dataset.c = String(c);
      cells.push(rect);
    }
  return cells;
}

/** @param {Element} layer @param {{ x: number, y: number, s: number }} box @param {Mark} mark */
function outline(layer, box, mark) {
  const inset = mark === 'skip' ? 1.6 : mark === 'both' ? 0 : 0.5;
  const colour = mark === 'deep' || mark === 'both' ? DEEP : mark === 'skip' ? SKIP : 'currentColor';
  const width = mark === 'sel' || mark === 'max' ? 2 : mark === 'plain' || mark === 'in' ? 0.8 : 1.2;
  const r = svg('rect', {
    x: box.x + inset, y: box.y + inset, width: box.s - 2 * inset, height: box.s - 2 * inset,
    fill: 'none', stroke: colour, 'stroke-width': width, 'pointer-events': 'none',
  }, layer);
  if (mark === 'in') r.setAttribute('stroke-dasharray', '2 1.5');
  if (mark === 'both') outline(layer, box, 'skip');
}

/** Turns taps on cells into selections. @param {SVGSVGElement} root @param {(s: Selection) => void} onPick */
function listen(root, onPick) {
  root.addEventListener('click', ev => {
    const d = /** @type {SVGElement} */ (ev.target).dataset;
    if (d.map) onPick({ kind: 'neuron', map: /** @type {MapName} */ (d.map), ch: +(d.ch ?? 0), r: +(d.r ?? 0), c: +(d.c ?? 0) });
    else if (d.layer && d.k === 'b') onPick({ kind: 'bias', layer: /** @type {LayerId} */ (d.layer), o: +(d.o ?? 0) });
    else if (d.layer) onPick({ kind: 'weight', layer: /** @type {LayerId} */ (d.layer), o: +(d.o ?? 0), i: +(d.i ?? 0), k: +(d.r ?? 0) * LAYERS[/** @type {LayerId} */ (d.layer)].ks + +(d.c ?? 0) });
  });
}

// ── the network, drawn as a U: encoder down the left, bottleneck at the bottom, decoder up the right ──

const S = 10; // one neuron
const BIG = N * S;
const SMALL = M * S;
/** where each map sits, keyed like 'e0' */
const AT = /** @type {Record<string, { x: number, y: number, n: number }>} */ ({
  x0: { x: 43, y: 16, n: N },
  e0: { x: 0, y: 128, n: N }, e1: { x: 86, y: 128, n: N },
  p0: { x: 80, y: 240, n: M }, p1: { x: 126, y: 240, n: M },
  b0: { x: 86, y: 340, n: M }, b1: { x: 132, y: 340, n: M }, b2: { x: 178, y: 340, n: M }, b3: { x: 224, y: 340, n: M },
  u0: { x: 200, y: 240, n: N }, u1: { x: 286, y: 240, n: N },
  d0: { x: 200, y: 128, n: N }, d1: { x: 286, y: 128, n: N },
  y0: { x: 200, y: 16, n: N },
  t0: { x: 286, y: 16, n: N },
});
const NAMES = /** @type {Record<MapName, string>} */ ({
  x: 'input', t: 'target', e: 'encoder', p: 'pooled', b: 'bottleneck', u: 'up-conv', d: 'decoder', y: 'output',
});
/** @param {MapName} m @param {number} ch */
export const mapName = (m, ch) => (m === 'x' || m === 't' || m === 'y' ? m : m + SUB[ch]);

/**
 * @param {SVGSVGElement} root  an empty <svg>
 * @param {(s: Selection) => void} onPick
 */
export function createNetView(root, onPick) {
  root.setAttribute('viewBox', '0 0 366 400');
  root.setAttribute('width', '366');
  // arrows first, so the maps sit on top of them
  const defs = svg('defs', {}, root);
  const head = svg('marker', { id: 'head', viewBox: '0 0 6 6', refX: 5, refY: 3, markerWidth: 5, markerHeight: 5, orient: 'auto-start-reverse' }, defs);
  svg('path', { d: 'M0,0 L6,3 L0,6 z', fill: 'currentColor' }, head);
  const arrows = svg('g', { stroke: 'currentColor', 'stroke-width': 1, fill: 'none', 'marker-end': 'url(#head)', opacity: 0.6 }, root);
  /** @param {number[]} pts @param {string} [dash] */
  const arrow = (pts, dash) => {
    const p = svg('polyline', { points: pts.join(' ') }, arrows);
    if (dash) p.setAttribute('stroke-dasharray', dash);
  };
  const ops = svg('g', { opacity: 0.85 }, root);
  arrow([83, 97, 83, 117]);
  label(ops, 128, 110, 'conv 3×3 · ReLU');
  arrow([83, 209, 110, 229]);
  label(ops, 4, 222, 'max-pool 2×2');
  arrow([123, 281, 145, 329]);
  label(ops, 30, 308, 'conv 3×3 · ReLU');
  label(ops, 30, 317, '2 channels → 4', { 'font-size': 7 });
  arrow([268, 338, 290, 326]);
  label(ops, 300, 345, 'up-conv 2×2');
  label(ops, 300, 354, '4 channels → 2', { 'font-size': 7 });
  arrow([283, 239, 283, 219]);
  label(ops, 290, 224, 'conv 3×3 · reads u and e', { 'font-size': 7 });
  arrow([240, 127, 240, 106]);
  label(ops, 246, 111, 'conv 1×1 · sigmoid', { 'font-size': 7 });
  // the skip connection: e is copied straight across to the decoder
  const skip = svg('g', { stroke: SKIP, 'stroke-width': 1.5, fill: 'none' }, root);
  svg('polyline', { points: '167,168 197,168', 'stroke-dasharray': '3 2', 'marker-end': 'url(#head)' }, skip);
  label(root, 170, 162, 'skip', { fill: SKIP, 'font-size': 7 });
  label(root, 170, 178, 'copy', { fill: SKIP, 'font-size': 7 });

  /** @type {Record<string, SVGRectElement[]>} */
  const cells = {};
  for (const [key, at] of Object.entries(AT)) {
    const m = /** @type {MapName} */ (key[0]);
    const ch = +key.slice(1);
    // wide maps carry their description beside the first name; the small bottleneck maps carry it below
    label(root, at.x, at.y - 4, mapName(m, ch) + (ch === 0 && m !== 'b' ? '  ' + NAMES[m] : ''), { 'font-size': 9 });
    if (m === 'b' && ch === 0) label(root, at.x, at.y + SMALL + 11, NAMES.b, { 'font-size': 9 });
    cells[key] = grid(root, at.x, at.y, at.n, S, { map: m, ch: String(ch) });
  }
  const marks = svg('g', {}, root);
  listen(root, onPick);

  return {
    /** @param {Record<string, Shown>} shown  keyed like 'e0' */
    setMaps(shown) {
      for (const [key, rects] of Object.entries(cells)) {
        const s = shown[key];
        rects.forEach((rect, j) => paint(rect, s?.values ? s.values[j] : null, s?.scale ?? 1));
      }
    },
    /** @param {Highlight} h */
    setHighlight(h) {
      marks.replaceChildren();
      for (const key of h.maps) {
        const at = AT[key];
        if (at) svg('rect', { x: at.x - 1.5, y: at.y - 1.5, width: at.n * S + 3, height: at.n * S + 3, fill: 'none', stroke: 'currentColor', 'stroke-width': 1, 'stroke-dasharray': '3 2', 'pointer-events': 'none' }, marks);
      }
      for (const { key, j, mark } of h.cells) {
        const at = AT[key];
        if (at) outline(marks, { x: at.x + (j % at.n) * S, y: at.y + Math.floor(j / at.n) * S, s: S }, mark);
      }
    },
  };
}

// ── the weights: one 3×3 grid per kernel, rows are output maps, columns input maps ──

const K = 9; // one weight
const names = (/** @type {MapName} */ m) => Array.from({ length: CH[m] }, (_, ch) => mapName(m, ch));
const BLOCKS = /** @type {Record<LayerId, { x: number, y: number, cols: string[], rows: string[] }>} */ ({
  enc: { x: 0, y: 0, cols: names('x'), rows: names('e') },
  dec: { x: 0, y: 96, cols: [...names('u'), ...names('e')], rows: names('d') },
  out: { x: 0, y: 192, cols: names('d'), rows: names('y') },
  mid: { x: 190, y: 0, cols: names('p'), rows: names('b') },
  up: { x: 190, y: 162, cols: names('b'), rows: names('u') },
});

/**
 * @param {SVGSVGElement} root  an empty <svg>
 * @param {(s: Selection) => void} onPick
 */
export function createWeightView(root, onPick) {
  root.setAttribute('viewBox', '0 -2 316 236');
  root.setAttribute('width', '316');
  /** @type {Record<string, SVGRectElement[]>} key 'dec 0 3' (layer, out, in) or 'dec 0 b' */
  const cells = {};
  /** @type {Record<string, { x: number, y: number, s: number }>} */
  const boxes = {};
  for (const id of LAYER_IDS) {
    const L = LAYERS[id];
    const B = BLOCKS[id];
    const pitch = L.ks * K + 5; // one kernel and a gap, across and down
    label(root, B.x, B.y + 8, `${id}: ${L.name}`, { 'font-weight': 'bold' });
    B.cols.forEach((name, i) => label(root, B.x + 18 + i * pitch, B.y + 20, name, { 'font-size': 7 }));
    const bx = B.x + 18 + L.ins * pitch + 4;
    label(root, bx, B.y + 20, 'bias', { 'font-size': 7 });
    B.rows.forEach((name, o) => {
      const y = B.y + 24 + o * pitch;
      label(root, B.x, y + 9, name, { 'font-size': 7 });
      for (let i = 0; i < L.ins; i++) {
        const key = `${id} ${o} ${i}`;
        const x = B.x + 18 + i * pitch;
        cells[key] = grid(root, x, y, L.ks, K, { layer: id, o: String(o), i: String(i) });
        boxes[key] = { x, y, s: L.ks * K };
      }
      const key = `${id} ${o} b`;
      cells[key] = grid(root, bx, y, 1, K, { layer: id, o: String(o), k: 'b' });
      boxes[key] = { x: bx, y, s: K };
    });
  }
  const marks = svg('g', {}, root);
  listen(root, onPick);

  return {
    /** @param {import('./net.js').Params} P  the numbers to colour by, @param {Record<LayerId, number>} scale */
    setWeights(P, scale) {
      for (const id of LAYER_IDS)
        P[id].w.forEach((row, o) => {
          row.forEach((k, i) => cells[`${id} ${o} ${i}`].forEach((rect, j) => paint(rect, k[j], scale[id])));
          paint(cells[`${id} ${o} b`][0], P[id].b[o], scale[id]);
        });
    },
    /** @param {Highlight} h  uses kernels (a whole kernel or bias) and cells keyed by kernel (one weight, j its index) */
    setHighlight(h) {
      marks.replaceChildren();
      for (const { key, mark } of h.kernels) {
        const b = boxes[key];
        if (b) outline(marks, { x: b.x - 1.5, y: b.y - 1.5, s: b.s + 3 }, mark);
      }
      for (const { key, j, mark } of h.cells) {
        const b = boxes[key];
        if (!b) continue; // a network cell, not a weight
        const ks = LAYERS[/** @type {LayerId} */ (key.split(' ')[0])].ks;
        outline(marks, { x: b.x + (j % ks) * K, y: b.y + Math.floor(j / ks) * K, s: K }, mark);
      }
    },
  };
}

// ── a few test images: input, output, target ──

/** @param {SVGSVGElement} root  an empty <svg> @param {number} count */
export function createTestView(root, count) {
  const s = 5;
  const w = N * s;
  const pitch = 3 * w + 2 * 6 + 18;
  const perRow = 2;
  const rows = Math.ceil(count / perRow);
  root.setAttribute('viewBox', `0 0 ${perRow * pitch - 18} ${rows * (w + 22)}`);
  root.setAttribute('width', String(perRow * pitch - 18));
  const sets = Array.from({ length: count }, (_, k) => {
    const x = (k % perRow) * pitch;
    const y = Math.floor(k / perRow) * (w + 22) + 10;
    const maps = ['x', 'y', 't'].map((name, m) => {
      label(root, x + m * (w + 6), y - 3, name, { 'font-size': 7 });
      return grid(root, x + m * (w + 6), y, N, s, {});
    });
    const verdict = label(root, x, y + w + 9, '', { 'font-size': 7 });
    return { maps, verdict };
  });
  return {
    /** @param {{ x: Float64Array, y: Float64Array, t: Float64Array, wrong: number }[]} tests */
    setTests(tests) {
      tests.forEach((tst, k) => {
        const set = sets[k];
        if (!set) return;
        [tst.x, tst.y, tst.t].forEach((values, m) => set.maps[m].forEach((rect, j) => paint(rect, values[j], 1)));
        const text = tst.wrong === 0 ? '✓ every pixel right' : `✗ ${tst.wrong} pixel${tst.wrong === 1 ? '' : 's'} wrong`;
        if (set.verdict.textContent !== text) set.verdict.textContent = text;
      });
    },
  };
}

// ── the loss chart: one polyline, stretched to the box ──

/** @param {SVGSVGElement} root  an empty <svg> */
export function createLossView(root) {
  const W = 300;
  const H = 80;
  root.setAttribute('viewBox', `0 0 ${W} ${H}`);
  root.setAttribute('preserveAspectRatio', 'none');
  root.setAttribute('width', String(W));
  root.setAttribute('height', String(H));
  for (let i = 1; i < 4; i++) svg('line', { x1: 0, x2: W, y1: (i * H) / 4, y2: (i * H) / 4, stroke: GRID, 'stroke-width': 0.5, 'vector-effect': 'non-scaling-stroke' }, root);
  const line = svg('polyline', { fill: 'none', stroke: POSITIVE, 'stroke-width': 1.5, 'vector-effect': 'non-scaling-stroke' }, root);
  return {
    /** @param {number[]} values  already averaged, oldest first @param {number} top  the loss at the top edge */
    setLoss(values, top) {
      const pts = values.map((v, i) => `${((i / Math.max(1, values.length - 1)) * W).toFixed(1)},${(H - (Math.min(v, top) / top) * (H - 1)).toFixed(1)}`);
      line.setAttribute('points', pts.join(' '));
    },
  };
}

export { C };
