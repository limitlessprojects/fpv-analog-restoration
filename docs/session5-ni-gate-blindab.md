# Session 5 (2026-10-27): editportal datapoint, NI gate wired, real-footage blind A/B

## editportal.net Quality Boost (one clip, browser-run)
- Clip: suite clip 005 (cockatoo w10 k23), 60 frames 480x360, identical scoring stack as run_suite (results/editportal_scores.json).
- Ran the actual site in headed Chromium under Xvfb via Playwright (client-side processing, no upload): had to bypass showOpenFilePicker (init-script kill + direct setInputFiles on hidden #qbInput), force-click routes fail because the AI mode buttons are DISABLED without WebGPU (no adapter in sandbox even with swiftshader/vulkan flags) => only Classic mode (denoise 35 / sharpen 60 / contrast 6 / vibrance 10 / Original scale) is scoreable.
- Result: +3.25 dB over degraded on all frames, ties naive on good frames, but p.lost 7.98 < degraded 10.09 (sharpens dropout garbage) and p.roll 12.37 vs naive 19.89. Every dedicated baseline beats it. Static flicker 26.83 vs naive 10.57 (its denoise barely helps temporally). Ledger: editportal_classic_datapoint.
- Lesson for the class: sharpen-on-damage is the exact failure mode sync+mask+fill prevents.

## No-invention gate wired from a distribution (not a single clip)
- Distribution: 8 suite windows (start 0..210 stride 30, seed 7), champion cfg + loo + save_masks (results/ni_dist/, code/ni_dist.py, ~52s/window).
- Thresholds (results/ni_thresholds.json): observed_floor 0.8937 (dist_min - 0.02), interp_ceiling 2, loo_p95_max_ceiling 120.9 (dist_max * 1.2), hf_band [0.7488, 1.02] ([dist_min - 0.08, 1.02]; >1.0 = more HF than clean on untouched pixels = invention). LPIPS not gated (no distribution).
- Self-check: champion passes its own gate on the tune clip (obs 0.917, interp 2, hf 0.829).
- Wiring: iterate.py run_gates gained G6_ni_observed / G6_ni_interp / G6_ni_loo (when candidate enables loo) / G6_ni_hf; candidates run with save_masks=1 so provenance + HF retention are always available. Ledger: ni_gate_wired.

## Real-footage blind A/B (the arbiter) — FINDING, reorders the loop
- mixkit drone content, simulated link seed 21; blind VLM judge, 5 randomized trials per A/B, ties favor status quo.
- AB1 bar vs ours: BAR WINS 3-2, zero ties. 4/5 trials: loser is "overly smoothed/blurred, losing texture". The suite's SSIM/static-flicker winner is also the blind-look winner on real content.
- AB2 degraded vs ours: ours wins 3-2, zero ties; judges still credit degraded texture in 2/5.
- Full writeup: docs/finding-blind-ab-real.md (mechanism: hf_retention 0.829 = the smoothing signature; PSNR edge lives on damaged frames, texture cost lives on healthy frames). Ledger: blind_ab_real_footage.
- Consequence: binding constraint is now texture preservation on observed pixels (HF retention / LPIPS axis). A champion that loses the blind A/B to its own two-piece baseline should not be the display champion.

## Housekeeping
- tune_clean.npy / tune_deg.npy / tune_meta.pkl rebuilt deterministically (start=150, seed 11) after sandbox wipe; setup_local.py reran clean; tune floor reproduced 23.286 (matches session record 23.29).
- New: code/ni_dist.py, code/ni_thresholds.py, code/real_ab.py, code/score_editportal.py, code/append_ledger_s5.py; results/ni_dist/*.json, results/ni_thresholds.json, results/editportal_scores.json, results/real_ab.json; media/real_ab_sidebyside.mp4 + judge sheets.
