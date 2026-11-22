// @ts-check
// The tasks: an 8×8 input image and the 8×8 image the network should turn it into.
// Each draws one random square; what to do with it is the task.
import { N } from './net.js';

/** @typedef {'outline' | 'big' | 'denoise'} Task */
/** @typedef {{ x: Float64Array, t: Float64Array }} Sample */

/** @type {Record<Task, { label: string, about: string }>} */
export const TASKS = {
  outline: {
    label: '🔲 Outline',
    about:
      'Keep only the edge of the square. Each output pixel can decide from its own small neighbourhood, but it needs that neighbourhood at full resolution: the skip connection carries it. Cut the skip and it gets close, about 97% of pixels, but rarely every pixel: the learned up-conv carries some detail up from the bottleneck, the skip carries all of it. Cut the bottleneck and it usually still learns.',
  },
  big: {
    label: '🔍 Big only',
    about:
      'Keep the square if it is 5 or more wide, erase it if it is smaller. A pixel cannot tell how big its square is from a 5×5 window, which is all the skip path sees. The bottleneck path sees the whole image. Cut the bottleneck and it stalls around 95%; cut the skip and it still learns.',
  },
  denoise: {
    label: '🧹 Denoise',
    about: 'Remove the speckles: one pixel in ten is flipped. Detail and context both help.',
  },
};

/** A random axis-aligned square, side lo to hi. @param {() => number} r @param {number} lo @param {number} hi */
function square(r, lo, hi) {
  const s = lo + Math.floor(r() * (hi - lo + 1));
  const top = Math.floor(r() * (N - s + 1));
  const left = Math.floor(r() * (N - s + 1));
  const filled = new Float64Array(N * N);
  const border = new Float64Array(N * N);
  for (let i = top; i < top + s; i++)
    for (let j = left; j < left + s; j++) {
      filled[i * N + j] = 1;
      if (i === top || i === top + s - 1 || j === left || j === left + s - 1) border[i * N + j] = 1;
    }
  return { filled, border, s };
}

/** @param {Task} task @param {() => number} r @returns {Sample} */
export function sample(task, r) {
  if (task === 'outline') {
    const q = square(r, 3, 6);
    return { x: q.filled, t: q.border };
  }
  if (task === 'big') {
    const q = square(r, 2, 7);
    return { x: q.filled, t: q.s >= 5 ? q.filled : new Float64Array(N * N) };
  }
  const q = square(r, 3, 6);
  return { x: q.filled.map(v => (r() < 0.1 ? 1 - v : v)), t: q.filled };
}
