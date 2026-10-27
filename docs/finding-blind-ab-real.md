# Finding: the blind judge prefers bar on real footage

Session 5, 2026-10-27. Status: MEASURED (mixed-content caveat below). This is a
finding, not a ledger footnote: it reorders the loop priorities.

## What was run

mixkit drone clip (real content), pushed through the same simulated analog link
(seed 21), 60 frames at 480x360. Two blind A/Bs, 5 randomized trials each
(2 static-flicker, 2 motion-detail, 1 motion-flicker), A/B assignment randomized
per trial, ties favor the second system (status-quo bias: a noisy judge must
EVIDENTIALLY prefer the candidate). The judge never knows which system is which.

GT-assisted scores for context (same clip):

| system   | P all | P good | P lost | S all | S good | flick static | flick tdiff |
|----------|-------|--------|--------|-------|--------|--------------|-------------|
| degraded | 19.14 | 22.04  | 7.85   | 0.164 | 0.267  | 24.47        | 15.93       |
| bar      | 28.78 | 30.13  | 19.60  | 0.748 | 0.791  | 0.89         | 0.99        |
| ours     | 29.15 | 31.35  | 22.27  | 0.739 | 0.835  | 1.81         | 1.27 |

## Results

**A/B 1: bar vs ours → bar wins 3-2, zero ties.**
Four of five trials describe the same failure with the same words: the loser is
"overly smoothed / overly blurred, losing texture"; bar is praised for
"preserving fine vertical lines and texture". On real content, the champion's
extra denoising reads as texture destruction to a blind judge.

**A/B 2: degraded vs ours → ours wins 3-2, zero ties.**
Restoration visibly reduces noise and flicker, but in 2 of 5 trials the judge
still credits the degraded frames with finer texture. Restoration helps, but it
is not a runaway win against doing nothing.

## Why this reorders the loop

The suite (N=69, CI-measured) said: ours = PSNR specialist (+1.69 dB paired,
+3.37 on lost), bar = SSIM-all (+0.032) and static-flicker specialist. The open
question since then was whether the PSNR margins survive the arbiter. They do
not. On real footage the blind judge sides with bar.

Mechanically this is consistent with the no-invention baseline: ours keeps only
0.829 of the clean high-frequency energy on pixels it never touched. That
missing ~17% is exactly the "overly smoothed" texture the judge keeps voting
against. The PSNR edge lives on damaged frames; the texture cost lives on
healthy frames, and healthy frames dominate the visual field.

## Consequences

1. The binding constraint is now texture preservation on observed pixels
   (HF retention / LPIPS axis), not PSNR margins. Candidates that trade more
   smoothing for dB are dead on arrival against this arbiter.
2. The display-champion question is open again: a pipeline that loses the blind
   A/B to its own two-piece baseline should not be the one shown to strangers.
3. Next iterations should target: less pass-2 smoothing where the mask says the
   signal is healthy, sharper fill-on-damaged-only, and HF-retention as a first
   class gate metric (already wired as G6_ni_hf, band [0.7488, 1.02] — note the
   champion sits at 0.829, inside the band but on the soft side).

## Caveats

- Content is real; the DAMAGE is simulated. This A/B says how restoration
  choices look on real content under our damage model, not how the pipeline
  handles native analog artifacts (no GT exists for that; it is the next
  measurement after the loop converges).
- Judge is a VLM, 5 trials per A/B, single clip. 3-2 with zero ties is a real
  signal, not a landslide; widening to more clips × seeds with the same sheet
  machinery is cheap and should happen before any champion swap.
- Scores are GT-assisted; the judge sees only sheets.
