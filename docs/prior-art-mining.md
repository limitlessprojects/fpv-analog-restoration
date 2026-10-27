# Prior-art deep mining: FPV_DVR_CLEANER (flugbild) and AnaHD

Both repos cloned and read line-by-line. Licenses verified from the LICENSE files:
both MIT (FPV_DVR_CLEANER: David-Andreas Tornow / contributors, 2026; AnaHD: Dominic Salas, 2026).
Everything below is extracted implementation fact, not marketing.

## FPV_DVR_CLEANER (src/flugbild/)

### OSD removal (osd.py) - fully extracted

Detection (per clip, 40 probes spread over the cut + a 6-frame sliding window for transient overlays):
- `_Coherence`: running per-channel sums of Sobel gradient VECTORS (int16 Sobel, half the memory)
  and gradient LENGTHS. Stillness = |sum of vectors| / sum of lengths > 0.7 (direction held)
  AND mean length > 80 (clearly drawn). Edge that points the same way every frame while the
  scene changes direction = overlay.
- Shape assembly: erase letterbox border edges, remove long straight runs (frame borders),
  dilate edges into text lines (rect W/64 x 3), close, then per-contour acceptance:
  reject if it spans the whole picture, hugs the left/right edge while being narrow,
  or is too big (h > 22% height). Too-big merged shapes keep only their still edges,
  no fill. Shapes dilated 7x7 ellipse.
- `_text_rows` second gate for sliding-window detections: minAreaRect tilt <= 8 deg and
  inside the shape >= 30% of strong edges are UPRIGHT (vertical strokes). This is how they
  avoid eating scene edges that slide along themselves during flight.
- Guide = per-pixel edge orientation in whole degrees inside the mask. "Is the OSD shown
  now?" = per element, share of guide edges found in the current frame within +-20 deg and
  strength > 40; shown at >= 0.35 share, then HOLD 150 frames.

Removal (`_Patcher`):
- DIS optical flow (FAST preset) from current frame into each of 6 neighbours, computed at
  <=320px width. Overlays hold flow at zero, so flow inside blocked areas is replaced by the
  mean of the surrounding open-area flow (41x41 box blur of flow*mask divided by blurred mask).
- Per connected region (10px context box): rim = 9px dilated ring outside the mask. For each
  neighbour (ordered by distance): warp the neighbour patch, reject if >5% of the region
  lands on overlay/picture-edge, fit the rim difference with a brightness PLANE
  (a + b*x + c*y per channel, lstsq, slopes clipped to +-40 - a single offset leaves a visible
  patch on sky/vignette gradients), fit score = mean |rim diff - plane|. Best neighbour below
  FIT_LIMIT=14 wins; early-exit at half the limit. No rim -> direct paste.
- Unresolved regions -> cv2.inpaint. Finally blendLinear with a feather that runs OUTSIDE the
  graphic only (edge = max(blur(dilate(mask)), mask)); inside the mask the refill stays at
  full weight or the overlay shines through along its edge.

### HDZero breakup repair (digital.py) - fully extracted

- Per 16px block: fine chroma energy at half size (chroma minus 3x3 blur, Cr+Cb) + speck share
  (pixels with fine energy > 24). Confetti is made of fine colour detail; real content and
  motion blur have little.
- Broken block: energy > threshold/2 AND > 4x calm + 1.5, or specks > 0.02 AND > 3x clear + 0.02,
  where calm/clear = per side the MIN over the +-1 window (confetti must be NEW vs both sides,
  so persistent colour like OSD does not fire). Whole-frame confetti: median energy > 2.5x
  neighbours + 0.5, or > 35% blocks broken (MIN_BLOCKS=30 otherwise it is just digits).
- Repair: nearest clean frame each side (>=90% of broken blocks calm), flow measured BETWEEN
  the two clean frames only (never through the broken one), backward warp both to the centre
  with share weighting, blend into the blocks with a Gaussian-softened mask (sigma = BLOCK/4).
  White OSD text (Y > 190, |Cr|+|Cb| < 24) is protected from the fill.

### Fusion (upscale.py _Fuser) - exact weights

- 5 frames, DIS flow at 320px, per-pixel MEDIAN reference.
- weights = exp(-gap^2 / tolerance^2), tolerance = 6 + 14*calm (calm 0..1 user knob);
  current frame weight multiplied by own_weight = 1 + 3*(1 - calm).
- Per-pixel trust gate on the FUSED result: spread = median over neighbours of |layer - ref|;
  trust = exp(-spread^2 / (1.5*tolerance)^2); fused = present + (blended - present) * trust.
  "Poor agreement means the motion was not followed, so keep the frame as it is." This is the
  same idea as our Prompt 4 occlusion cap, expressed as a pull-back-to-present.
- Optional lift: unsharp on the fused luma, detail clamped to +-LIFT_LIMIT.

### Analog chroma repair (analog.py) - exact formulas

- Coarse repair grid (W/4, H/2); chroma as ONE COMPLEX PLANE (Cr real, Cb imag).
- Hue tear score: |angle(z[row+1] * conj(z[row]))| (phase jump between rows) blurred (15,1),
  gated on saturation > 10 and level > 10/18 ramp, jumps > 0.6 ramp; dilated vertically.
- Row colour-dropout: per-row mean |chroma|*(1-tears) -> (rows - 1.5)/2.5 clipped 0..1
  (absolute test: a truly grey scene stays grey).
- Alignment: DIS flow at 320px; moved_fit = exp(-((luma_nb - luma_now)/16)^2).
  STILL-FIT TRICK: if the unmoved fit (comparing neighbour's own luma to current luma without
  warping) beats the moved fit, take the neighbour's chroma UNMOVED - static content and OSD
  are served exactly, and flow failure cannot corrupt them.
- Lasting-tear consensus: a tear edge that appears identically in the aligned neighbours is
  picture, not carrier damage: own_tears *= (1 - mean(neighbour tears)).
- Weighted chroma estimate with floor (fit + 0.05 keeps colour available during fast motion),
  then 2 IRLS iterations with CAUCHY reweighting: w *= 1/(1 + (|sample - est|/10)^2).
- No backing (no usable neighbour) -> estimate fades to NEUTRAL GREY, never spreads a stripe.
- Strength = max(drained, stray, own_tears, calming) with area-fade (Gaussian sigma 6 * 0.9)
  so pixel-wise decisions do not leave blotches.
- Unbacked bands (colour edge without brightness edge, 31x1 run detector) -> luma-guided
  chroma prediction: local linear model colour = a*luma + b on a 61x25 window fitted on
  trusted pixels only, coverage gate 0.3, slopes smoothed (31,13).
- Final guided filter (radius 4x2, eps = 20 + 120*amount, blend 0.6*amount): chroma follows
  luma contours because analog chroma carries little detail of its own.

### GPU/CPU hybrid (gpu.py)

- OpenCL via cv2.UMat transparent API - same code path for CPU and GPU, no vendor SDK.
- Small/branchy work (flow at reduced size, masks, bookkeeping) STAYS on CPU threads; only
  heavy per-pixel full-res arithmetic (median networks, remap, weight math) goes to the GPU.
- BoundedSemaphore slots (default 2 frames on the GPU at once) - more would just queue;
  the other pool threads keep the CPU busy with flow.
- OPENCV_OPENCL_BUFFERPOOL_LIMIT=1GB set BEFORE the first GPU call (without it allocation
  dominates and serialises threads).
- Self-test on startup: the same warp+exp chain must agree CPU vs GPU to 1e-3, else CPU only.
- Env kill switch FLUGBILD_GPU=0.

### Simple restore path (restoration.py) - for reference

- Farneback at <=640px for alignment. Analog: chroma-only temporal blend, gate =
  (luma_gap < 14 + amount*0.12) & (chroma_gap > max(2.0, fingerprint.farbrauschen*1.3)),
  strength = 0.18 + amount*0.005. Digital: 8px blocks, damage = block diff > max(18, 37 -
  0.16*amount + blocknaht*0.8) & (edges > 1.7*old + 12 | variance < 0.22*old), outer 1/12
  bands guarded for OSD, 5x5 blur on the mask.

## AnaHD (src/anahd/)

### Fusion (clean.py) - exact weights

- radius 3 ("the 2026-10-12 blind-test winner"), DIS flow (MEDIUM preset) at flow_scale 0.5.
- match = 1/(1 + (diff/match_sigma)^4), match_sigma = 7.0 luma levels, diff measured on
  5x5-blurred (sigma 1.2) warped vs reference luma (blur before compare = noise-tolerant).
- weight = match * valid * keep * max(0, 1 - 0.1*|distance|); valid = inside frame AND not
  from the neighbour's OSD (remapped keep-mask, borderValue=0 - outside pixels are unusable).
- Chroma at half res: flows halved, cmatch = 1/(1 + (cdiff/chroma_sigma)^4) with
  chroma_sigma = 6.0, cdiff = max(|dU|, |dV|) after 3x3 blur.
- OSD of the CURRENT frame is kept as-is (weight 1 baseline); neighbour OSD pixels are dropped
  via the remapped keep mask, so text cannot smear into the picture.

### Destripe / derainbow (clean.py)

- destripe: horizontal line offsets = the part of each row's fine vertical detail that stays
  constant along the row. vertical = Gaussian (1,7) sigmaY 1.5; detail = clip(y - vertical,
  +-4); per-row-mean over a 241px window with OSD masked out; subtract.
- derainbow: fine chroma detail cut where luma is finely textured (cross-colour): texture =
  blurred |yh - blur(yh, 1.0)| sigma 2; cut = 0.8 * clip(texture/6.0, 0, 1); chroma = low +
  (c - low)*(1 - cut) with low = Gaussian sigma 1.5.

### Steadying (steady.py) - NOT vsync recovery

- goodFeaturesToTrack (300 pts, masked away from OSD + static_mask: props, arms) -> LK pyr
  (21x21) -> estimateAffinePartial2D RANSAC 3px. Needs >= 30 inliers and >= 0.3 ratio.
- Path = cumsum of per-frame (dx, dy, da); correction = Gaussian-smoothed path (sigma 4
  frames) minus path; strength = ramp of frame motion between calm_shift 1.5% and busy_shift
  4% of width per frame (and 0.5..1.5 deg/frame), window-min over +-2 sigma, smoothed sigma 2,
  capped at 1% width / 0.5 deg; zoom = smallest (max 1.04) hiding the corrected edges.
- Rolls/flips fade steadying OUT (90 deg/s >> busy_angle) - confirmed: AnaHD has NO analog
  vsync/roll recovery at all. Consistent with our bench: it collapses on rolled frames.

### Colour clean-up + deflicker (remaster/stage.py, GPU path)

- Chroma temporal clean-up: aligned neighbours (warp y/u/v by stored flows), weight =
  amount * inside * (1 - FALLOFF*|d|) / (1 + (diff/MATCH)^4) * keep, same quartic family.
- Global chroma deflicker: per-frame mean chroma over non-OSD pixels stored on the frame;
  target = mean of the (c-1, c, c+1) means; u += amount * (target_u - mean_u) * keep.
  3-frame, global, chroma-only - cheap and it targets brightness pumping.
- Tone: per-clip 256-entry LUT from sampled levels (black/white points, strength-blended),
  s-curve contrast: t + contrast*(t-0.5)*(1-|2t-1|). "so brightness doesn't pump" = the
  per-clip curve instead of per-frame normalisation.
- OSD stays fixed while the picture warps: inpaint OSD before the affine, paste the OSD back
  from the unmoved frame with a feathered alpha.

## What we take from this

1. anaHD's quartic match 1/(1+(d/sigma)^4) and flugbild's Gaussian exp(-d^2/tol^2) are the two
   canonical fusion kernels; our exp(-d^2/2h^2) sits in the same family. Our Prompt 4 upgrade
   (separate noise from flow error via |grad|*delta, FB occlusion cap) exists in NEITHER repo -
   flugbild's trust gate (median spread -> pull back to present) is the closest relative.
2. flugbild's still-fit trick (prefer the unmoved neighbour when it matches better) is a cheap
   guard for static scenes that our stack lacks; candidate follow-up.
3. flugbild's brightness-PLANE rim correction (a+bx+cy per channel, slopes clipped) beats a
   plain offset when pasting patches over gradients - relevant if we ever paste fills.
4. Cauchy IRLS on the temporal estimate (2 iterations) is flugbild's answer to outlier
   neighbours; our median fill is the robust equivalent, but the fusion mean is not - the z^2
   weighting plays that role instead.
5. The 3-frame global chroma deflicker is a possible cheap add if chroma flicker shows up on
   real footage; our current residual budget is luma, so not now.
6. Neither repo has: sync/vsync recovery, pilot-based line re-timing, damage-driven masking,
   provenance tracking. Our edge stands; their fusion kernels are cleaner than our old one.
