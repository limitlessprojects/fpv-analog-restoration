# Session 4 (2026-10-27): close the evaluation gaps, then the loop

Directive: close Gap 1 (fair chroma test), Gap 2 (bar reset), Gap 3 (N expansion) before
running improvement iterations; then one gated temporal_post iteration (detail band).

## F1 finding published first

The fz negative result is now a standalone document:
`docs/finding-fz-likelihood-fusion.md` (mechanism, numbers, caveats, generalizable rule).

## Gap 3 / Gap 2 tooling (run in CI: workflow eval-suite-n)

- `code/run_suite.py`: 23 windows x 3 seeds = 69 paired runs; systems = naive med3+nlm,
  bar (sync+med3+nlm = run A two-stage), ours (champion P22). Sources: imageio ships exactly
  ONE standard video (cockatoo.mp4, 280 frames) so content diversity is limited to one
  source; adjacent stride-10 windows share up to 50 frames. This limitation is stated here
  and handled by super-window clustering in the bootstrap.
- `code/aggregate_suite.py`: paired deltas with cluster bootstrap (super-windows of
  non-overlapping 60-frame spans), per-clip win rates, per-damage margins (roll/lost/fades).
- 12 parallel CI shards.

## Gap 1 tooling (run in CI: workflow chroma-fair)

- `code/sim_chroma.py`: chroma-dominant damage profile with EVERYTHING the competitors do
  not claim to fix removed (no roll, no lost, no dropout bands, no line jitter) and the
  chroma path stressed instead: hue instability (fast alternating + slow drift), saturation
  pumping, heavy chroma noise (mild luma noise), chroma-only streaks with horizontal
  bleeding, chroma-only multipath echo, comb-filter leakage at sharp luma edges.
- `code/metrics_chroma.py`: per-plane PSNR (Y/Cr/Cb), chroma flicker (static-pair std on
  chroma planes), hue bias/spread (circular), saturation ratio.
- `code/run_chroma.py`: tune grid on clip 0 (selected on CHROMA PSNR, their home metric),
  one-shot scoring on the rest; systems = degraded, naive, bar, ours, flugbild, anahd.

Local smoke of the full chroma path (2 clips, both engines) produced a plausible split:
flugbild won Y-plane PSNR (24.9 vs ours 23.7) and saturation; ours won chroma PSNR,
chroma flicker, hue spread; AnaHD classic hurt luma badly (Y 19.5). Final numbers come
from the CI run on 8 clips.

## Loop iteration 1: detail-band temporal_post -> REVERT with mechanism

Hypothesis: metric-domain residual flicker on calm static pairs is 64% detail band
(diag_target_check re-verified this session: 69.58 detail vs 39.34 lowpass, matching
session 3), so a flow-gated temporal median restricted to the detail band kills flicker
without touching the low band (the P5A stab lesson).

Screening (tune clip, champion 27.36 / 7.477 / 0.814):

| arm                  | p.all | flick_static_good | ssim |
|----------------------|-------|-------------------|------|
| T1 band r2 f0.30     | 27.27 | 7.511             | 0.817|
| T2 band r2 f0.45     | 27.26 | 7.511             | 0.817|
| T3 band r3 f0.30     | 27.28 | 7.480             | 0.818|
| T4 full r2 f0.30     | 27.32 | 7.422             | 0.817|

All four REVERT (no improvement; T1-T3 cost ~0.08-0.09 dB with flicker flat).

Mechanism (diag_tp.py, measured): gate coverage is not the limiter (median 36% of pixels
pass); the post-polish residual on gated GT-static pixels is correlated rho=0.705 across
consecutive frames because the fusion stages share warped data across the window. Pair-diff
variance on gated static pixels: 8.90 (champ) -> 8.95 (T1). An iid residual would allow a
2.2x variance cut (11.22 -> 5.10); at rho=0.705 a 5-frame median is structurally an
identity operation on it. Conclusion: NO multi-frame median/average can touch the remaining
flicker; the levers are estimation-and-subtraction (stab family, already failed once) or
pair-specific blending (EFS, already ON). The flicker axis is at a local optimum for the
median family; recorded as a finding, no test-clip scoring (did not survive the tune clip).

## No-invention gate groundwork (loop priority 3)

- `restore.py` new knobs (default OFF, champion unchanged): `loo` (leave-one-out fill
  instability probe), `save_masks` (per-frame impulse masks into the log).
- `log["provenance"]`: per-frame observed / warped / interpolated fractions.
- `code/no_invention.py`: scores provenance + LOO fill instability + HF retention on
  observed good pixels; LPIPS section auto-enables when torch+lpips are installed.
- Champion baseline on tune clip (measured this session):
  - provenance: 2 interpolated frames, mean observed 0.917, mean warped 0.049
  - loo_fill: 114 frame-passes with fills, mean instability 4.29 gray, p95 mean 15.09,
    p95 max 54.23 (fills that lean on a single observation; flag for the gate)
  - hf_retention (good, observed): 0.829 (hf out 9.37 vs clean 11.54 vs degraded 15.85)
- Gate wiring into iterate.py happens AFTER the suite establishes the distribution of these
  metrics across clips (thresholds must not be set from a single clip).

## Status of the three gaps at time of writing

- Gap 1 + Gap 3: tooling pushed, CI workflows queued (chroma-fair, eval-suite-n).
- Gap 2: analysis runs the moment the suite artifacts land (aggregate_suite.py output is
  the Gap-2 verdict: paired ours-minus-bar deltas with cluster CIs, win rates, and whether
  the lost/roll/fades edge survives SSIM/flicker parity).

## RESULTS (all measured this session on GitHub Actions runners)

### Gap 3: N expansion (eval-suite-n, 69 paired runs, 23 windows x 3 seeds)

System means: naive 23.32 / 0.791 SSIM / flick_static 9.75; bar 26.32 / 0.840 / 1.76;
ours 28.00 / 0.808 / 3.78. Ours runtime mean 54.1 s per clip (2-core runner).

Paired ours-minus-bar (cluster bootstrap 95% CI, 4 content clusters, 69 pairs):

| metric            | delta            | win rate |
|-------------------|------------------|----------|
| p.all             | +1.69 [+1.27, +2.13] | 23/23 windows |
| p.good            | +1.84 [+1.27, +2.44] | 23/23 |
| p.fades           | +1.37 [+0.94, +1.78] | 23/23 |
| p.roll            | +1.30 [+0.76, +1.84] | 23/23 |
| p.lost            | +3.37 [+2.98, +3.74] | 23/23 |
| s.all             | -0.032 [-0.041, -0.020] | 0/23 |
| s.good            | +0.001 [-0.014, +0.016] | 11/23 (parity) |
| flick_static_good | +2.02 [+1.20, +2.91] | 0/23 (bar wins) |
| flick_tdiff_good  | -0.37 [-0.54, -0.20] | 23/23 |

Every point estimate is resolved well beyond the ~0.3 dB single-clip resolution limit.
Content caveat, stated plainly: one 280-frame source (imageio ships exactly one sample
video), adjacent windows share up to 50 frames, 4 non-overlapping content clusters. The
seeds sample damage randomness, not scene diversity. Sync value replicated: bar - naive
+2.99 [+2.81, +3.17] here vs +2.96 on the single test clip.

### Gap 2: bar verdict

The two-stage bar is NOT beaten on its own axes: it wins overall SSIM (-0.032, driven by
damaged frames; good-frame SSIM is exact parity at +0.001) and static-region flicker
(1.76 vs 3.78). Ours wins PSNR everywhere INCLUDING good frames (+1.84) and wins
tdiff flicker (-0.37). So the edge survives as: (a) the largest margin is still lost
(+3.37), but (b) there is a real good-frame PSNR edge the attribution run underestimated,
and (c) static flicker parity is NOT reached - the bar remains the static-flicker
specialist. Under the standing rules the blind A/B on real footage is the arbiter; these
axes now have honest error bars for that comparison.

### Gap 1: fair chroma test (chroma-fair, 8 clips, tune-clip grid selection)

Settings tuned on clip 0 by CHROMA PSNR (their home metric): flugbild rm50 fb50,
AnaHD no_look=True. 7 clips scored once. Degraded chroma PSNR 14.87 (chroma is the fight).

| system   | chroma | Y     | chroma flick | hue spread | SSIM |
|----------|--------|-------|--------------|------------|------|
| degraded | 14.87  | 25.97 | 25.27        | 77.1       | 0.151|
| naive    | 16.79  | 25.80 | 3.29         | 61.0       | 0.620|
| bar      | 16.84  | 25.81 | 3.27         | 60.3       | 0.631|
| ours     | 17.46  | 26.68 | 5.13         | 58.4       | 0.666|
| flugbild | 18.00  | 27.46 | 13.09        | 73.6       | 0.359|
| anahd    | 16.29  | 19.73 | 9.43         | 71.2       | 0.479|

Split verdict, as expected and now measured:
- flugbild WINS its envelope: chroma PSNR +0.54 mean over ours (clearly better on 4/7
  clips), best Y (27.46), best saturation ratio. NOT beaten on chroma magnitude fidelity.
- OURS WINS temporal+structural: chroma flicker 5.13 vs 13.09, hue spread 58 vs 74 deg,
  SSIM 0.666 vs 0.359 (flugbild's fusion wrecks structure on this damage).
- AnaHD classic is BELOW naive even on chroma-dominant damage (chroma 16.29 vs 16.79;
  Y 19.73 is catastrophic at 480x360 - its upscale+sharpen path is destructive here).
- naive and bar are tied on chroma: sync recovery adds nothing when there is no sync
  damage (sanity check passes).

Caveats: competitor outputs pass through their own encoders (crf 10-14) vs our
pre-encoder evaluation; the slow hue drift is unfixable by every system (shared
handicap); N=7 scored clips for the competitors.
