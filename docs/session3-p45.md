# Session 3: prior-art mining + Prompt 4 / Prompt 5A screening

## Prior-art deep mining (docs/prior-art-mining.md)

Both competitor repos cloned, LICENSE files verified (both MIT), and read line-by-line.
Extracted: flugbild OSD detection/removal (coherence sums, brightness-plane rim fit,
orientation guide), HDZero confetti repair (fine-chroma block energy, clean-side flow),
exact fusion kernels (Gaussian vs quartic match, trust gates, Cauchy IRLS, still-fit trick),
AnaHD steady (feature-tracked affine, fades out on flips - NO vsync recovery, confirming
our bench), destripe/derainbow/deflicker formulas, GPU/CPU hybrid architecture.
Everything is in docs/prior-art-mining.md with constants.

## Screening (tune clip s150 k11, champion A = 27.36 / flick 7.477 / ssim 0.814)

| variant | p.all | flick_static_good | verdict |
|---|---|---|---|
| B fz (guide+z2) | 26.02 | 8.656 | REVERT |
| B1 fz_guide only | 26.45 | 8.850 | REVERT |
| B2 fz_z2 only | 26.45 | 8.711 | REVERT |
| C stab raw | 27.10 | 7.915 | REVERT |
| C2 stab + trajectory smooth | 27.19 | 7.553 | REVERT |

## Why Prompt 4 failed (mechanism, not noise)

- z2 term: loss correlates +0.68 with frame texture. The prescription assumed a guide diff on
  textured pixels with flow error is "flow noise, not content" -> be tolerant. On OUR data
  Farneback error on texture samples the WRONG TEXELS, so the diff IS content mismatch and
  tolerance = melting. The FB occlusion cap only covers the extreme tail.
- Stable guide: the fused-median guide is biased toward the very neighbours being weighted,
  so diffs shrink for wrong neighbours too (circularity). Guide noise is high-frequency and
  the residual flicker is low-frequency: guide noise was never the culprit.
- Conclusion: the per-frame noisy guide + tight h is doing real work. Prompt 4 closed as
  empirically dead; code kept behind fz=0 for the record.

## Why Prompt 5A failed (target moved)

- Raw stab injected flicker: the correction field itself wobbles 0.81 gray/frame on calm
  static pairs (diagnosed by reconstructing the b-field). Trajectory smoothing (stab_ts=2,
  the spec step round 1 omitted) recovered most of it (7.915 -> 7.553) but still nets
  negative.
- Target re-check (diag_target_check.py): on the tune clip with the CURRENT champion the
  residual is 90% slow TEMPORALLY (48% <1Hz + 42% 1-5Hz) but only ~41% lowpass SPATIALLY,
  and the metric-domain pair-diff variance is 64% DETAIL band (69.6 vs 39.4 lowpass). The
  test-clip finding (93% lowpass) does not transfer. A luma-only low-band stabilizer can
  address at most a third of the metric here and loses to its own injection.
- Next flicker hypothesis (recorded in ledger): re-aim at the detail band on calm static
  pairs - the temporal_post family (flow-gated temporal median), which the champion has OFF,
  is the existing lever; a fresh hard-gated iteration on that is cheaper than a new stage.

## Code state

- restore.py: fz (fz/fz_guide/fz_z2/fz_occ/fz_occ_max/fz_smooth) and stab
  (stab/stab_sigma/stab_tile/stab_rad/stab_max/stab_flow/stab_dev/stab_persist/stab_ts)
  stages, all default OFF; champion unchanged (P22).
- screen_p45.py / screen_p45_r2.py / diag_p45.py / diag_target_check.py: screening + diagnosis
  scripts, rerunnable.
- results/ledger.json: 6 new entries (5 REVERT verdicts + 1 diagnostic finding).
