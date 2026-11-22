// @ts-check
// The store, its actions, and the effects that push it into the page and the drawings.
import { signal, computed, effect, batch, untracked, bind, series } from 'plainpanel';
import * as net from './net.js';
import { TASKS, sample } from './data.js';
import { STAGES, LAST, FWD, BACK, WGRAD, explain, highlight } from './explain.js';
import { createNetView, createWeightView, createTestView, createLossView } from './view.js';

/** @typedef {import('./view.js').Selection} Selection */
/** @typedef {import('./view.js').MapName} MapName */
/** @typedef {import('./data.js').Task} Task */

// ── store ──

const state = {
  params: signal(net.init()),
  before: signal(/** @type {net.Params | null} */ (null)), // the weights before the last update
  trace: signal(/** @type {net.Trace | null} */ (null)), // the current sample's forward pass
  grads: signal(/** @type {net.Grads | null} */ (null)), // and its backward pass
  stage: signal(-1), // the last stage done
  samples: signal(0), // samples trained on
  playing: signal(false),
};
const train = {
  task: signal(/** @type {Task} */ ('outline')),
  skip: signal(true), // the skip connection is on
  deep: signal(true), // the bottleneck path is on
  alpha: signal(0.003), // learning rate
  speed: signal(0), // while playing: 2^speed stages per frame (negative: one stage every 2^-speed frames)
};
const ui = {
  mode: signal(/** @type {'value' | 'grad'} */ ('value')),
  sel: signal(/** @type {Selection | null} */ ({ kind: 'neuron', map: 'y', ch: 0, r: 2, c: 2 })),
  field: signal(true), // show the receptive field
  follow: signal(true), // each step selects a neuron in the map it computed
  noCss: signal(false),
};
const losses = series(20000);
const TEST_COUNT = 32;
const TEST_SHOWN = 4;

const ctx = computed(() => ({
  P: state.params(),
  before: state.before(),
  T: state.trace(),
  G: state.grads(),
  stage: state.stage(),
  mode: ui.mode(),
  alpha: train.alpha(),
}));

const tests = computed(() => {
  const task = train.task();
  const cut = { skip: train.skip(), deep: train.deep() };
  const P = state.params();
  return Array.from({ length: TEST_COUNT }, (_, k) => {
    const s = sample(task, net.rng(1000 + k));
    const T = net.forward(P, s.x, s.t, cut);
    return { x: s.x, y: T.y[0], t: s.t, wrong: Math.round((1 - net.accuracy(T)) * net.N * net.N) };
  });
});

const recent = (/** @type {number} */ k) => {
  const v = /** @type {number[]} */ (losses.read().slice(-k).filter(x => x !== null));
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
};

// loss: the samples so far, averaged into at most 300 points
const chartData = computed(() => {
  const all = /** @type {number[]} */ (losses.read().filter(v => v !== null));
  const buckets = Math.min(300, all.length);
  const per = all.length / Math.max(1, buckets);
  const values = Array.from({ length: buckets }, (_, b) => {
    const part = all.slice(Math.floor(b * per), Math.max(Math.floor(b * per) + 1, Math.floor((b + 1) * per)));
    return part.reduce((a, v) => a + v, 0) / part.length;
  });
  return { values, top: Math.max(0.05, ...values) };
});
const views = {
  samples: computed(() => String(state.samples())),
  next: computed(() => STAGES[(state.stage() + 1) % STAGES.length].label),
  done: computed(() => (state.stage() < 0 ? 'none yet' : STAGES[state.stage()].label)),
  taskAbout: computed(() => TASKS[train.task()].about),
  alpha: computed(() => train.alpha().toFixed(3)),
  speed: computed(() => {
    const k = train.speed();
    const perSec = 60 * 2 ** k;
    return perSec < STAGES.length * 2 ? `≈ ${Math.round(perSec)} stages/s` : `≈ ${Math.round(perSec / STAGES.length)} samples/s`;
  }),
  loss: computed(() => {
    const T = state.trace();
    return T && state.stage() >= FWD.y ? T.loss.toFixed(3) : '–';
  }),
  avgLoss: computed(() => {
    const v = recent(200);
    return v === null ? '–' : v.toFixed(3);
  }),
  test: computed(() => {
    const t = tests();
    const perfect = t.filter(x => x.wrong === 0).length;
    const wrong = t.reduce((a, x) => a + x.wrong, 0);
    return `${perfect} of ${TEST_COUNT} perfect · ${(100 - (100 * wrong) / (TEST_COUNT * net.N * net.N)).toFixed(1)}% of pixels right`;
  }),
  chartTop: computed(() => chartData().top.toFixed(3)),
  inspect: computed(() => explain(ui.sel(), ctx())),
  stepLocked: computed(() => state.playing()),
  pauseLocked: computed(() => !state.playing()),
};

// ── training ──

function load() {
  const s = sample(train.task(), Math.random);
  const P = state.params();
  const T = net.forward(P, s.x, s.t, { skip: train.skip(), deep: train.deep() });
  state.trace(T);
  state.grads(net.backward(P, T));
  state.before(null);
  state.stage(0);
}

/** Do the next stage. The forward and backward passes are computed when the sample loads; the stages reveal them. */
function stepStage() {
  const next = (state.stage() + 1) % STAGES.length;
  if (next === 0) return load();
  if (next === LAST) {
    const P = state.params();
    state.before(P);
    state.params(net.apply(P, /** @type {net.Grads} */ (state.grads()).w, train.alpha()));
    state.samples(state.samples() + 1);
    losses.push(/** @type {net.Trace} */ (state.trace()).loss);
  }
  state.stage(next);
}

const finishSample = () => {
  do stepStage();
  while (state.stage() !== LAST);
};

/** After a single step: show what it computed. */
function followStep() {
  const st = STAGES[state.stage()];
  if (st.dir === 'update') return;
  ui.mode(st.dir === 'back' ? 'grad' : 'value');
  if (!ui.follow()) return;
  const m = /** @type {MapName} */ (st.shows);
  const old = ui.sel();
  const from = old?.kind === 'neuron' ? old : { map: 'x', ch: 0, r: 2, c: 2 };
  const n0 = from.map === 'p' || from.map === 'b' ? net.M : net.N;
  const n1 = m === 'p' || m === 'b' ? net.M : net.N;
  const scale = (/** @type {number} */ v) => Math.min(n1 - 1, Math.floor((v * n1) / n0));
  ui.sel({ kind: 'neuron', map: m, ch: m === 'x' || m === 'y' ? 0 : Math.min(from.ch, net.C - 1), r: scale(from.r), c: scale(from.c) });
}

const actions = {
  step: () =>
    batch(() => {
      stepStage();
      followStep();
    }),
  sample: () => batch(finishSample),
  hundred: () =>
    batch(() => {
      for (let k = 0; k < 100; k++) finishSample();
    }),
  play: () => state.playing(true),
  pause: () => state.playing(false),
  reset: () =>
    batch(() => {
      state.playing(false);
      state.params(net.init());
      state.samples(0);
      losses.clear();
      load();
    }),
};

// play: the speed slider sets how many stages each animation frame does
effect(() => {
  if (!state.playing()) return;
  let id = 0;
  let frame = 0;
  const tick = () => {
    const k = untracked(() => train.speed());
    frame++;
    if (k >= 0 || frame % 2 ** -k === 0)
      batch(() => {
        for (let s = k >= 0 ? 2 ** k : 1; s > 0; s--) stepStage();
      });
    id = requestAnimationFrame(tick);
  };
  id = requestAnimationFrame(tick);
  return () => cancelAnimationFrame(id);
});

// a new task, or a path cut or restored, starts a new sample so what is on screen matches the network
let first = true;
effect(() => {
  train.task();
  train.skip();
  train.deep();
  if (first) first = false;
  else untracked(() => batch(load));
});

// ── page ──

bind(document.body, { state, train, ui, views, actions });

const pick = (/** @type {Selection} */ s) => ui.sel(s);
const netView = createNetView(/** @type {SVGSVGElement} */ (/** @type {unknown} */ (document.getElementById('net'))), pick);
const weightView = createWeightView(/** @type {SVGSVGElement} */ (/** @type {unknown} */ (document.getElementById('weights'))), pick);
const testView = createTestView(/** @type {SVGSVGElement} */ (/** @type {unknown} */ (document.getElementById('tests'))), TEST_SHOWN);
const lossView = createLossView(/** @type {SVGSVGElement} */ (/** @type {unknown} */ (document.getElementById('loss'))));

const maxAbs = (/** @type {Float64Array[]} */ maps) => Math.max(1e-9, ...maps.map(m => m.reduce((a, v) => Math.max(a, Math.abs(v)), 0)));

// the maps: values on the way forward, gradients on the way back, blank until their stage
effect(() => {
  const { T, G, stage, mode } = ctx();
  /** @type {Record<string, import('./view.js').Shown>} */
  const shown = {};
  for (const m of /** @type {MapName[]} */ (['x', 't', 'e', 'p', 'b', 'u', 'd', 'y'])) {
    const data = m === 'x' || m === 't';
    const grad = mode === 'grad' && !data;
    const ready = T && stage >= (grad ? BACK[m] : FWD[m]) && (!grad || G);
    const maps = ready ? /** @type {Float64Array[]} */ (grad ? /** @type {net.Grads} */ (G)[/** @type {keyof net.Grads} */ (m)] : T[m]) : null;
    const scale = !maps ? 1 : data || (m === 'y' && !grad) ? 1 : maxAbs(maps);
    (maps ?? [null, null]).slice(0, m === 'x' || m === 't' || m === 'y' ? 1 : net.C).forEach((values, ch) => (shown[`${m}${ch}`] = { values, scale }));
  }
  netView.setMaps(shown);
});

// the weights: their values, or their gradients once the backward pass has computed them
const ZERO = net.apply(net.init(1), net.init(1), 1); // every weight 0: "not computed yet"
effect(() => {
  const { P, G, stage, mode } = ctx();
  const show = /** @type {net.Params} */ ({});
  const scale = /** @type {Record<net.LayerId, number>} */ ({});
  for (const id of net.LAYER_IDS) {
    const layer = mode === 'grad' ? (G && stage >= WGRAD[id] ? G.w[id] : ZERO[id]) : P[id];
    show[id] = layer;
    scale[id] = maxAbs([...layer.w.flat(), Float64Array.from(layer.b)]);
  }
  weightView.setWeights(show, scale);
});

effect(() => {
  const h = highlight(ui.sel(), ctx(), ui.field());
  netView.setHighlight(h);
  weightView.setHighlight(h);
});

effect(() => testView.setTests(tests().slice(0, TEST_SHOWN)));

effect(() => {
  const { values, top } = chartData();
  lossView.setLoss(values, top);
});

effect(() => {
  const off = ui.noCss(); // "no CSS": the same HTML, zero styling
  for (const el of document.querySelectorAll('link[rel=stylesheet], style')) {
    /** @type {HTMLLinkElement | HTMLStyleElement} */ (el).disabled = off;
  }
});

load();
