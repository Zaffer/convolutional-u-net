// @ts-check
// The store, its actions, and the effects that push it into the page and the drawings.
import { signal, computed, effect, batch, untracked, bind, series } from 'plainpanel';
import * as net from './net.js';
import { TASKS, sample } from './data.js';
import { STAGES, LAST, FWD, BACK, WGRAD, explain, highlight, paramLabel } from './explain.js';
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
const edit = {
  value: signal(0), // the selected weight's value, as the slider shows it
  orig: signal(/** @type {Map<string, number>} */ (new Map())), // what each hand-edited weight was before you moved it
};
// panel sizes in px, set by dragging the splitters (wide screens only)
const SIZES = { left: 270, right: 400, bottom: 250 };
const saved = (() => {
  try {
    return { ...SIZES, ...JSON.parse(localStorage.getItem('unet-layout') ?? '{}') };
  } catch {
    return SIZES;
  }
})();
const layout = {
  left: signal(/** @type {number} */ (saved.left)),
  right: signal(/** @type {number} */ (saved.right)),
  bottom: signal(/** @type {number} */ (saved.bottom)),
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
  editName: computed(() => {
    const s = ui.sel();
    return s && s.kind !== 'neuron' ? paramLabel(s) : 'nothing yet: tap a weight or bias';
  }),
  editValue: computed(() => edit.value().toFixed(2)),
  editLocked: computed(() => state.playing() || ui.sel()?.kind === 'neuron' || !ui.sel()),
  restoreLocked: computed(() => {
    const s = ui.sel();
    return state.playing() || !s || s.kind === 'neuron' || !edit.orig().has(keyOf(s));
  }),
  undoLocked: computed(() => state.playing() || edit.orig().size === 0),
  edited: computed(() => {
    const n = edit.orig().size;
    return n === 0 ? 'none' : `${n} weight${n === 1 ? '' : 's'} (the next update clears this)`;
  }),
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
    if (edit.orig().size) edit.orig(new Map()); // training has moved every weight: "put it back" no longer means much
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
  ui.sel({ kind: 'neuron', map: m, ch: Math.min(from.ch, net.CH[m] - 1), r: scale(from.r), c: scale(from.c) });
}

// ── editing a weight by hand ──

const keyOf = (/** @type {net.ParamAddress} */ a) => `${a.layer} ${a.o} ${a.i ?? 'b'} ${a.k ?? ''}`;

/** Use hand-edited weights, then redo this sample's forward pass with them, so every map, the output and the tests follow. */
function useParams(/** @type {net.Params} */ P) {
  state.params(P);
  state.before(null);
  const T0 = state.trace();
  if (!T0) return;
  const T = net.forward(P, T0.x[0], T0.t[0], { skip: train.skip(), deep: train.deep() });
  state.trace(T);
  state.grads(net.backward(P, T));
  state.stage(FWD.y); // forward pass done; the next step starts the backward pass
  ui.mode('value');
}
const setWeight = (/** @type {net.ParamAddress} */ a, /** @type {number} */ v) => useParams(net.setParam(state.params(), a, v));

const selectedParam = () => {
  const s = ui.sel();
  return s && s.kind !== 'neuron' ? /** @type {net.ParamAddress} */ (s) : null;
};

const actions = {
  /** the slider moved */
  nudge: (/** @type {Event} */ e) => {
    const a = selectedParam();
    if (!a) return;
    const v = /** @type {HTMLInputElement} */ (e.target).valueAsNumber;
    batch(() => {
      const key = keyOf(a);
      if (!edit.orig().has(key)) edit.orig(new Map(edit.orig()).set(key, net.getParam(state.params(), a)));
      setWeight(a, v);
    });
  },
  restore: () => {
    const a = selectedParam();
    if (!a) return;
    const was = edit.orig().get(keyOf(a));
    if (was === undefined) return;
    batch(() => {
      const left = new Map(edit.orig());
      left.delete(keyOf(a));
      edit.orig(left);
      setWeight(a, was);
    });
  },
  undoAll: () =>
    batch(() => {
      let P = state.params();
      for (const [key, was] of edit.orig()) {
        const [layer, o, i, k] = key.split(' ');
        P = net.setParam(P, /** @type {net.ParamAddress} */ (i === 'b' ? { layer, o: +o } : { layer, o: +o, i: +i, k: +k }), was);
      }
      edit.orig(new Map());
      useParams(P);
    }),
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
      edit.orig(new Map());
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

// the slider shows the selected weight's current value: on selecting it, and as training moves it
effect(() => {
  const a = selectedParam();
  if (a) edit.value(net.getParam(state.params(), a));
});

// the panel sizes, into the CSS grid (and remembered in this browser)
const main = /** @type {HTMLElement} */ (document.querySelector('main'));
effect(() => {
  const sizes = { left: layout.left(), right: layout.right(), bottom: layout.bottom() };
  for (const [k, v] of Object.entries(sizes)) main.style.setProperty(`--${k}`, `${v}px`);
  try {
    localStorage.setItem('unet-layout', JSON.stringify(sizes));
  } catch {}
});

// the splitters: a drag (or the arrow keys) is one write to that panel's size
for (const bar of /** @type {NodeListOf<HTMLElement>} */ (document.querySelectorAll('[data-split]'))) {
  const side = /** @type {keyof typeof layout} */ (bar.dataset.split);
  const panel = /** @type {HTMLElement} */ (document.getElementById(side));
  const across = side === 'bottom';
  const grow = side === 'left' ? 1 : -1; // the left panel grows rightwards; the right one leftwards, the bottom one upwards
  const size = () => (across ? panel.offsetHeight : panel.offsetWidth);
  const limit = (/** @type {number} */ v) => Math.round(Math.max(across ? 100 : 160, Math.min(v, (across ? 0.7 * innerHeight : 0.45 * innerWidth))));
  /** @type {{ at: number, from: number } | null} */
  let drag = null;
  const pos = (/** @type {PointerEvent} */ e) => (across ? e.clientY : e.clientX);
  bar.addEventListener('pointerdown', e => {
    bar.setPointerCapture(e.pointerId);
    drag = { at: pos(e), from: size() };
  });
  bar.addEventListener('pointermove', e => {
    if (drag) layout[side](limit(drag.from + grow * (pos(e) - drag.at)));
  });
  bar.addEventListener('lostpointercapture', () => (drag = null));
  bar.addEventListener('keydown', e => {
    const step = { ArrowLeft: -16, ArrowUp: -16, ArrowRight: 16, ArrowDown: 16 }[e.key];
    if (!step) return;
    e.preventDefault();
    layout[side](limit(size() + grow * step));
  });
}

// ── page ──

bind(document.body, { state, train, ui, edit, views, actions });

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
    for (let ch = 0; ch < net.CH[m]; ch++) shown[`${m}${ch}`] = { values: maps ? maps[ch] : null, scale };
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
