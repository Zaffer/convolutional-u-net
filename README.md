# Convolutional U-Net

The smallest U-Net that is still a U-Net, learning on 8×8 images, with every neuron and every
weight on screen.

**Live: https://zaffer.github.io/convolutional-u-net/**

```
x (1×8×8) ─conv 3×3, ReLU─▶ e (2×8×8) ───────────── skip connection (copy) ──────────────┐
                              │ max-pool 2×2                                             ▼
                            p (2×4×4) ─conv 3×3, ReLU─▶ b (4×4×4) ─up-conv 2×2─▶ u (2×8×8)
                                                                      [u, e] ─conv 3×3, ReLU─▶ d (2×8×8)
                                                                                d ─conv 1×1, sigmoid─▶ y (1×8×8)
```

207 weights. As in real U-Nets, the channels double where the resolution halves, and a learned
up-convolution (a 2×2 transposed convolution) halves them again on the way back up. Trained one
sample at a time by plain gradient descent, w ← w − α·∂L/∂w, on binary cross-entropy.

## What to do with it

- **➡️ step** walks one sample through 14 stages: the forward pass layer by layer, the backward
  pass layer by layer, then the update. Tap any neuron or weight and the inspector writes out the
  exact arithmetic: the 3×3 windows times the kernels, the sum, the bias, the ReLU; on the way
  back, every term of every gradient; for a weight, its gradient summed over every position it
  is used at, and the update.
- **✏️ edit a weight.** Tap any weight or bias, then drag its slider. The current sample's
  forward pass reruns with the new value, so every map, the output, the loss and the test images
  respond while you drag. ↩️ puts it back; ⏪ undoes every edit.
- **🎯 receptive field** marks every input pixel that can reach the selected neuron, coloured by
  the road it takes: through the bottleneck, or through the skip connection.
- **Why the U.** Each task draws one square. 🔍 big only needs to see the whole square, so it
  needs the bottleneck: cut it and every run stalls near 95% of pixels. 🔲 outline needs fine
  detail: cut the skip and it gets to about 97% but rarely every pixel. Switch either path off,
  press 🔄 new weights, ▶️ play, and watch.
- About one start in three on 🔲 outline settles on copying its input (89% of pixels right)
  and stays there. 🔄 new weights.

## Files

```
index.html   the panels: native controls in fieldsets, bound to the store with data-* attributes
app.js       the store (plainpanel signals), the stages and actions, the effects that draw
net.js       the network as pure functions: conv, pool, up-conv, each with its backward pass
explain.js   the inspector's text and the highlights, as pure functions of the store
data.js      the three tasks
view.js      the drawings: SVG behind a narrow API (setMaps, setWeights, setHighlight)
style.css    layout only; the page works with it switched off (view → 🚫 no CSS)
```

On a wide screen the page is four panels: controls on the left, the network in the centre, the
weight editor and the inspector on the right, and the weights, loss and test images along the
bottom. Drag the bars between them to resize them. On a phone the panels stack. Either way no panel
takes its size from text that changes while training, so nothing moves under your finger.

No build step. [plainpanel](https://github.com/Zaffer/plainpanel) loads from jsDelivr, pinned in
the import map. To run it locally, serve the folder, for example `python3 -m http.server`, and
open http://localhost:8000.

The gradients are checked against finite differences (worst error 2·10⁻⁹), and each task's
claim about which path it needs was measured: three seeds each, 20,000 samples, 50 test images.
