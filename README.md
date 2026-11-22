# convolutional U-net

The smallest U-net that is still a U-net, learning on 8×8 images, with every neuron and every
weight on screen.

**Live: https://zaffer.github.io/convolutional-u-net/**

```
x (1×8×8) ─conv 3×3, ReLU─▶ e (2×8×8) ──────────── skip connection (copy) ─────────────┐
                              │ max-pool 2×2                                           ▼
                            p (2×4×4) ─conv 3×3, ReLU─▶ b (2×4×4) ─upsample ×2─▶ u (2×8×8)
                                                                      [u, e] ─conv 3×3, ReLU─▶ d (2×8×8)
                                                                                d ─conv 1×1, sigmoid─▶ y (1×8×8)
```

135 weights. Trained one sample at a time by plain gradient descent, w ← w − α·∂L/∂w, on
binary cross-entropy.

## What to do with it

- **➡ step** walks one sample through 14 stages: the forward pass layer by layer, the backward
  pass layer by layer, then the update. Tap any neuron or weight and the inspector writes out the
  exact arithmetic: the 3×3 windows times the kernels, the sum, the bias, the ReLU; on the way
  back, every term of every gradient; for a weight, its gradient summed over every position it
  is used at, and the update.
- **🎯 receptive field** marks every input pixel that can reach the selected neuron, coloured by
  the road it takes: through the bottleneck, or through the skip connection.
- **Why the U.** Each task draws one square. 🔲 outline needs fine detail, so it needs the skip
  connection. 🔍 big only needs to see the whole square, so it needs the bottleneck. Switch
  either path off, press ↺ new weights, ▶ play, and watch which task stalls.

## Files

```
index.html   the panels: native controls in fieldsets, bound to the store with data-* attributes
app.js       the store (plainpanel signals), the stages and actions, the effects that draw
net.js       the network as pure functions: conv, pool, upsample, each with its backward pass
explain.js   the inspector's text and the highlights, as pure functions of the store
data.js      the three tasks
view.js      the drawings: SVG behind a narrow API (setMaps, setWeights, setHighlight)
style.css    layout only; the page works with it switched off (view → 🚫 no CSS)
```

No build step. [plainpanel](https://github.com/Zaffer/plainpanel) loads from jsDelivr, pinned in
the import map. To run it locally, serve the folder, for example `python3 -m http.server`, and
open http://localhost:8000.

The gradients are checked against finite differences (worst error 2·10⁻⁹), and each task's
claim about which path it needs was measured: three seeds each, 20,000 samples, 50 test images.
