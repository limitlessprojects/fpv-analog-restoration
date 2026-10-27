# Analog FPV Restoration Pipeline

A classic CV (no neural nets) restoration pipeline for analog FPV video, developed
against a physically-motivated simulator with ground truth, and driven by an
automated propose -> apply -> gate-check -> blind-judge -> keep/revert loop.

**Honest scope note:** all quantitative results below use simulated damage (FM click
noise, dropout bands, vsync roll, lost frames, multipath ghost, per-line jitter,
bandwidth limit) applied to real video content (imageio cockatoo, a Mixkit drone clip).
The damage model was written by the same author as the pipeline: self-benchmark bias
is expected. Real raw FPV sources are being sourced separately; the pipeline has not
been validated on real RF-decoded footage yet.

## Results

Groups: `good` = quality>0.7 clean frames, `fades` = quality<0.5, `roll` = vsync loss,
`lost` = complete signal loss. Flicker metrics are GT-assisted temporal-consistency
measures (lower is better; see `code/metrics2.py`).

### Held-out test clip (scored once after locking, start=60 seed=7)

| Method | all PSNR/SSIM | good | fades | roll | lost | flick-static-good |
|---|---|---|---|---|---|---|
| degraded input | 18.43 / 0.214 | 22.15 / 0.346 | 15.17 / 0.106 | 11.51 / 0.060 | 7.49 / 0.010 | 28.38 |
| hqdn3d (strong) | 19.51 / 0.311 | 23.77 / 0.531 | 15.77 / 0.134 | 11.81 / 0.079 | 7.60 / 0.010 | 22.09 |
| temporal NLM h25 | 22.71 / 0.779 | 25.51 / 0.879 | 20.27 / 0.703 | 13.25 / 0.538 | 9.48 / 0.078 | 17.91 |
| median3 + NLM | 23.27 / 0.770 | 25.24 / 0.862 | 21.44 / 0.678 | 20.03 / 0.624 | 14.41 / 0.303 | 9.25 |
| **this pipeline (locked)** | **27.73 / 0.778** | 30.09 / 0.878 | 25.68 / 0.684 | 25.85 / 0.677 | 20.91 / 0.639 | **2.67** |

### Real-content demo (Mixkit drone clip through the same link, seed=21)

| | all | good | fades | roll | lost | flick-static-good |
|---|---|---|---|---|---|---|
| degraded | 16.81 / 0.320 | 19.12 / 0.465 | 14.05 / 0.181 | 9.84 / 0.030 | 9.56 / 0.013 | 34.19 |
| restored | 25.32 / 0.804 | 26.62 / 0.872 | 24.39 / 0.756 | 24.28 / 0.753 | 21.88 / 0.616 | 3.08 |

`media/real_sidebyside.mp4` is the degraded | restored comparison.

## Pipeline (7 stages, all in `code/restore.py`)

1. **Sync recovery**: per-line pilot template match (Gaussian CDF edge, subpixel grid
   search + parabolic refine), 3-row median outlier rejection, line re-timing; vsync
   via ring template match on the row-median profile of the left strip; frame-loss
   detection from pilot-fit RMS + Immerkaer sigma. Never uses inter-frame matching for
   roll (camera vertical motion causes false positives).
2. **Motion-compensated stacking** (median5 + Gaussian guide, Farneback flow).
3. **Impulse/dropout masks** (residual threshold, row-energy test, directional grow,
   fill from warped neighbors).
4. **Motion-adaptive temporal mean** (Gaussian weight on intensity delta).
5. **Pass 2** re-estimates guides/flow from pass-1 output, wider window.
6. **NLM spatial polish** (noise-adaptive h).
7. **Lost-frame flow midpoint interpolation** + event-flicker smoother (the one change
   the VLM judge loop kept: light temporal smoothing at detected quality-event steps).

## Evaluation loop

`code/iterate.py` implements: hard gates first (G0 beats baseline floor, G1 shape,
G2 no core regression on roll/lost, G3 PSNR floor vs champion, G4 flicker caps,
G5 runtime), then score gate, then a blind VLM judge (`code/judge.py`, anonymized
A/B sheets, randomized assignment, 3 trials, majority vote) on the tuning clip only.
The held-out test clip was scored exactly once. 21 proposals went through the loop:
19 rejected (over-smoothers that won PSNR but lost the blind judge; hard-gate
failures), 1 kept, 1 initial seed.

## Repo layout

```
code/    simulator (sim.py), pipeline (restore.py), metrics, baselines, loop, judge
results/ champion config, score JSONs, append-only experiment ledger
media/   comparison sheets + side-by-side video
.github/workflows/  CI smoke test + runner sweeps (screening only; VLM judge stays local)
```

## Running

```bash
pip install numpy opencv-python-headless scipy scikit-image imageio imageio-ffmpeg
python3 code/sim.py                          # builds test + tune clips
python3 code/run_baselines.py                # baseline outputs
python3 code/evaluate_final.py               # reproduce the test-clip table
python3 code/iterate.py --seed               # rebuild champion from defaults
python3 code/iterate.py NAME '<json-delta>'  # one loop round
```

## Known weaknesses

- Damage is simulated by the same author (self-benchmark bias; real RF footage untested).
- Ghost removal is disabled (blind gain estimation failed; oracle gain only +0.4 dB).
- Deep-fade frames retain partial dropout-band residue; band detection is the weakest stage.
- Multipath ghost in fades is beyond blind correction without a reference (no GCR in FPV).
- Speed: ~50 s / 60 frames at 480x360 single-core; not yet real-time.
- The VLM judge is 3 trials majority vote: fragile, but every decision it made is in
  `results/ledger.json`.
