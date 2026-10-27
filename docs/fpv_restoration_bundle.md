# Analog FPV restoration: bundle v2 (agent-loop edition)

Status: final config locked via a propose -> apply -> blind-judge -> gate-check -> keep/revert
loop. All tuning on a separate clip (start=150, seed=11). Test clip (start=60, seed=7) scored
ONCE at the end. A real-content clip (mixkit drone footage, same simulated link, seed=21) was
added as a generalization check. Everything is still SIMULATED damage; content is real in the
demo clip. Treat numbers as indicative, not as proof about real RF links.

## Final results, TEST clip (held out, scored once)

| Method | All | Good | Fades | Roll | Lost |
|---|---|---|---|---|---|
| Degraded input | 18.43 / 0.214 | 22.15 / 0.346 | 15.17 / 0.106 | 11.51 / 0.060 | 7.49 / 0.010 |
| hqdn3d strong | 19.51 / 0.311 | 23.77 / 0.531 | 15.77 / 0.134 | 11.81 / 0.079 | 7.60 / 0.010 |
| Temporal NLM h25 | 22.71 / 0.779 | 25.51 / 0.879 | 20.27 / 0.703 | 13.25 / 0.538 | 9.48 / 0.078 |
| Median3 + NLM | 23.27 / 0.770 | 25.24 / 0.862 | 21.44 / 0.678 | 20.03 / 0.624 | 14.41 / 0.303 |
| **MINE (locked)** | **27.73 / 0.778** | **30.09 / 0.878** | **25.68 / 0.684** | **25.85 / 0.677** | **20.91 / 0.639** |

Each cell PSNR dB / SSIM. PSNR/SSIM reproduce the previous session's numbers (reproducibility
confirmed on a rebuilt harness).

## NEW: flicker / temporal-consistency metrics (GT-assisted, eval only)

- flick_static_good: std of consecutive-frame output diffs on pixels the ground truth holds
  static, restricted to good-frame pairs. Measures residual grain + wiggle where the eye locks.
- flick_tdiff_good: mean | |out_t - out_{t-1}| - |gt_t - gt_{t-1}| | over good pairs. Penalizes
  added flicker AND lost real motion (both directions bad).

| Method | flick_static_good | flick_tdiff_good | flick_static_all | flick_tdiff_all |
|---|---|---|---|---|
| Degraded | 28.38 | 15.52 | 40.83 | 25.79 |
| Median3 + NLM | 9.25 | 6.56 | 10.05 | 7.71 |
| NLM h25 | 17.91 | 7.26 | 25.09 | 11.64 |
| **MINE (locked)** | **2.67** | **5.70** | **4.77** | **6.41** |

The static-flicker win is 3.5x over the best baseline. This was the biggest evaluation gap from
the previous session and it is now measured and won.

## Real-content demo (mixkit drone clip, simulated link, seed=21)

| Method | All | Good | Fades | Roll | Lost | flick_static_good |
|---|---|---|---|---|---|---|
| Degraded | 16.81 / 0.320 | 19.12 / 0.465 | 14.05 / 0.181 | 9.84 / 0.030 | 9.56 / 0.013 | 34.19 |
| **MINE** | **25.32 / 0.804** | 26.62 / 0.872 | 24.39 / 0.756 | 24.28 / 0.753 | 21.88 / 0.616 | **3.08** |

+8.5 dB over degraded input, flicker cut 11x. Margins are LARGER than on cockatoo: real drone
scenes (trees, gravel, repeated texture) are friendlier to motion-compensated temporal fusion.
Video: real_sidebyside.mp4 (degraded | restored). Still simulated damage on real content.

## The agent loop (this session's method)

Protocol per iteration: propose a config delta -> run pipeline on the TUNING clip -> hard gates
-> score gate -> blind VLM judge -> keep or revert. State in champion.json; append-only
ledger.json records every decision with reasons.

Hard gates (checked in order, any failure = immediate revert):
- G1 shape/dtype/finite.
- G0 floor: PSNR_all >= best baseline on tuning clip (23.29 dB).
- G2 core: roll-group PSNR >= champ - 0.5 dB; lost-group >= champ - 0.5 dB (protects the design win).
- G3 PSNR floor: all-frames >= champ - 0.10 dB.
- G4 flicker: tdiff <= champ x 1.05 AND static <= champ x 1.15.
- G5 runtime <= 300 s.

Score gate: keep only if PSNR improves > 0.02 dB, OR ties within 0.02 with SSIM/flicker gains,
OR a large flicker win (>= 0.15 tdiff) buys a PSNR cost no worse than 0.05 dB.

Blind judge: z-ai vision (GLM) on anonymized [reference | A | B] sheets, 3 trials (flicker on
static region, detail on motion region, flicker on motion region), A/B randomized per trial,
majority vote. Candidate must not lose. The judge vetoed two PSNR-positive variants (windowed
NLM polish +0.06 dB, wider pass 2 +0.02 dB) for over-smoothing. It has real teeth.

Loop tally: 21 iterations, 1 keep, 19 reverts (gate kills: flow damping, vertical inpaint fill,
sharpening, spread-gated EFS, wrong-segment ghost cross-fade; VLM kills: 2 over-smoothers;
score kills: no-op knobs). The kept change: event-flicker smoother, light setting.

## Kept change: event-flicker smoother (EFS)

After polish: for each consecutive pair, gate pixels static across 3 consecutive pairs (both
flow directions < 0.5 px). Pair diff on gated pixels; robust spike threshold (median + 4 MAD + 2).
Blend both frames of up to 6 spike pairs at w=0.15. Rationale: impairments switch on/off in
single frames (ghost segment boundaries, dropout-band churn); these steps dominate residual
flicker. Forensics: median per-pair residual sigma is 1.57 but the worst pairs reach 20.

## Failed directions (do not retry blindly)

- Blind ghost deconvolution: detection collapses in deep fades (tune clip echo at 28-44 sits at
  q 0.02-0.4; lag z-score fires on noise). Boundary cross-fade from that detector hurt PSNR.
- Static-region temporal median post-pass: residual flicker is temporally CORRELATED low-frequency
  error from noisy-region fusion, not independent grain; medians of 5 contaminated frames stay
  contaminated.
- Flow-magnitude damping of temporal weights: intensity weights already collapse during motion
  jerks; damping weakens normal fusion (roll PSNR -1.4 dB).
- VCR-style vertical inpaint fill for band rows: TELEA smearing costs 0.15 dB vs warped-neighbour fill.
- Stronger EFS blends (w 0.20-0.35): motion-jerk pairs get double-exposed; PSNR cost exceeds the
  flicker-trade allowance.
- Sharpening: PSNR gate rejects before the judge sees it.

## Known weaknesses (unchanged and honest)

- Simulator and pipeline share an author. Real FPV has interlacing, OSD overlays, AGC pumping,
  DVR compression artifacts; none are simulated.
- Fades PSNR (25.68 on test) still has the most headroom; dropout-band concealment is partial.
- Lost frames are interpolated (no real information recovery).
- The blind judge is 3 trials; 2-1 votes are fragile. Use 5 trials for final calls.
- Learned damage detection would likely beat the residual thresholds; not tested here.

## Code map (code/ directory, authoritative)

- sim.py: simulator (unchanged from v1; builds test + tune clips).
- restore.py: config-driven pipeline. New knobs: efs/efs_w/efs_max/efs_k/efs_floor/efs_frac,
  temporal_post (kept off; see failed directions), polish_multi (off; VLM vetoed),
  flow_damp/fill_mag/vfill_frac (off; failed), ghost_smooth (off), sharpen (off).
  Locked config: {"efs": true, "efs_w": 0.15, "efs_max": 6}.
- metrics2.py: PSNR + flicker metrics. evaluate_final.py: test-clip tables.
- iterate.py: loop driver. judge.py: blind VLM judge (z-ai vision CLI).
- run_baselines.py, tune_baselines.py, baselines.py: baselines.
- real_demo.py: mixkit real-content demo. ledger.json: full decision log.
- diag_flicker.py, diag_grain_vs_wiggle.py: the diagnostics that located the flicker sources.

## Run order from scratch

python3 sim.py  ->  python3 iterate.py --seed  ->  python3 tune_baselines.py
->  python3 iterate.py NAME '<cfg json>' (repeat)  ->  python3 -c 'restore test clip with locked cfg; save mine_final.npy'
->  python3 run_baselines.py  ->  python3 evaluate_final.py  ->  python3 real_demo.py

Requires: Python 3, numpy, opencv-python, scikit-image, imageio, ffmpeg, z-ai CLI for the judge.
Approx 50 s per 60-frame restore at 480x360 on 2 CPU cores.
