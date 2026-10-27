# Finding F1: gradient-scaled likelihood fusion loses to thresholding when loss correlates with texture

**Status:** measured negative result (session 3 screening, tune clip s150 k11). Closed; code kept
behind `fz=0` in restore.py for the record. This document exists so the result is citable
instead of buried in the ledger.

## The hypothesis (Prompt 4)

Replace the fusion weight `w = exp(-d^2 / 2h^2)` (a threshold on the guide difference `d`)
with a likelihood model that separates the two error sources inside `d`:

    z^2 = d^2 / (2*sigma_n^2 + (|grad I| * delta_fb)^2)
    w   = exp(-z^2 / 2)

where `sigma_n` is the per-frame noise estimate, `|grad I|` the guide gradient magnitude, and
`delta_fb` the forward-backward optical-flow disagreement (px). The theory is standard: on a
textured pixel with flow error, a large guide diff is *expected* (the gradient amplifies
misalignment), so the model should tolerate it instead of discarding the neighbour; on a flat
pixel the same diff is noise-dominated and gets weighted normally. A forward-backward
occlusion cap (`fz_occ`, weight ceiling on high-disagreement pixels) was added as a melting
guard, plus a "stable guide" variant (weights computed on the fused median instead of the
single noisy frame, BM3D stage-2 pilot logic).

## Result (tune clip, champion P22 = 27.36 dB / 0.814 SSIM / flick 7.477)

| variant                          | p.all | flick_static_good | ssim  | verdict |
|----------------------------------|-------|-------------------|-------|---------|
| B  fz full (stable guide + z^2)  | 26.02 | 8.656             | 0.798 | REVERT  |
| B1 fz_guide only (stable guide)  | 26.45 | 8.850             | 0.805 | REVERT  |
| B2 fz_z2 only (z^2 + occ cap)    | 26.45 | 8.711             | 0.803 | REVERT  |

Every component loses PSNR *and* makes flicker worse. The occlusion cap does not rescue the
z^2 term; the stable guide does not rescue anything either.

## Mechanism (measured, not speculated)

1. **The prescription's premise is false on this data.** The model assumes
   `P(loss | texture, flow error) ~ independent of texture given delta_fb`. Measured: the
   correlation between per-pixel restoration loss and frame texture is **+0.68**. Farneback
   flow on analog-damaged footage fails *at texels* (it snaps to the wrong texture phase), so
   on textured pixels the guide difference is content mismatch, not amplified-alignment noise.
   The likelihood model's tolerance band opens exactly where the diff is most likely to be
   real error, and tolerance there is melting. The FB disagreement cap only trims the extreme
   tail of that; the body of the distribution stays poisoned.
2. **The stable guide is circular.** The fused-median guide is biased toward the very
   neighbours being weighted, so the measured diffs shrink for *wrong* neighbours too. Weights
   get looser where they should get tighter. Guide noise is high-frequency while the residual
   flicker problem is low-frequency (see diag_target_check), so guide re-noising was never the
   binding constraint in the first place.
3. **The per-frame noisy guide + tight h is doing real work.** It is a crude but honest
   outlier test: "does this neighbour agree with the current frame *as observed*". Any
   smoothing of the guide or widening of the kernel inherits the circularity in (2).

## Why this is worth recording

The z^2 form is the textbook-correct generalization (it is the standard likelihood ratio for
"noise vs misalignment" and appears in BM3D-style pairwise fusion literature). The failure is
not the algebra; it is that the flow-error term `delta_fb` is not a sufficient statistic for
"this diff is misalignment" when the loss distribution itself is texture-dependent. On real
analog damage, loss concentrates on texture, so any per-pixel model that becomes more
tolerant on textured pixels imports the damage the pipeline exists to remove.

**Generalizable rule extracted:** widening fusion tolerance proportional to local texture is
only safe if the flow error estimate is conditionally independent of texture. With a
dense-on-damage estimator (Farneback here), it is not. Either estimate flow error from
clean-ish references only, or keep the threshold.

## Caveats (explicit)

- Measured on the simulated tune clip, N=1 screening, no seed replication. The *sign* of the
  effect is large (-1.3 dB) and mechanistically explained, but the magnitude is
  simulator-specific and unreplicated.
- This result cannot distinguish "z^2 weighting is bad" from "z^2 weighting is bad *with
  Farneback flow on this simulator*". A higher-quality flow estimator could change the
  texture-loss correlation; that combination was not tested (it would also change the
  compute budget).
- Flags under the standing rules: this is a single-clip screening verdict, not a
  distribution-level measurement.

Ledger entries: `P4_fz_full`, `P4a_fz_guide_only`, `P4b_z2_only` (all REVERT, session 3).
Code: `restore.py` knobs `fz, fz_guide, fz_z2, fz_occ, fz_occ_max, fz_smooth` (default `fz=0`).
Screening scripts: `screen_p45.py`, `screen_p45_r2.py`; mechanism diagnosis: `diag_p45.py`.
