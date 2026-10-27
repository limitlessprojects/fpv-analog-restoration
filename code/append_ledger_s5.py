"""Append session-5 entries to results/ledger.json (append-only JSONL)."""
import json

entries = [
 {"name": "editportal_classic_datapoint", "verdict": "MEASURED",
  "setup": "editportal.net Quality Boost, Classic mode (denoise 35, sharpen 60, contrast 6, vibrance 10, scale Original), run in a real browser (Playwright + Xvfb, headed; client-side processing, no upload). Input: suite clip 005 (cockatoo w10 k23), 60 frames 480x360, identical scoring stack as run_suite. AI Fast / AI Pro buttons are DISABLED when WebGPU is unavailable (sandbox has no GPU adapter), so only Classic is scoreable.",
  "scores": {"degraded": {"p.all": 18.08, "s.all": 0.201, "flkS": 30.40},
             "editportal_classic": {"p.all": 21.33, "s.all": 0.600, "flkS": 26.83, "p.lost": 7.98, "p.roll": 12.37},
             "naive": {"p.all": 23.17, "s.all": 0.791, "flkS": 10.57},
             "bar": {"p.all": 25.99, "s.all": 0.837, "flkS": 1.88},
             "ours": {"p.all": 28.35, "s.all": 0.812, "flkS": 4.18}},
  "answer": "Generic enhancer helps a little (+3.25 dB over degraded on all frames, ties naive on good frames) but AMPLIFIES analog-specific damage: p.lost 7.98 is WORSE than degraded 10.09 (sharpening dropout garbage), p.roll 12.37 vs naive 19.89. Every dedicated baseline in the stack beats it. Sharpen-on-damage is exactly the failure mode the sync+mask+fill class exists to prevent.",
  "location": "results/editportal_scores.json; input/output mp4s kept outside the repo"},

 {"name": "ni_gate_wired", "verdict": "MEASURED",
  "setup": "NI distribution: 8 suite windows (start 0..210 stride 30, seed 7) through champion cfg + loo + save_masks; metrics provenance / loo_fill / hf_retention per window (results/ni_dist/). Thresholds = status-quo guards: observed_floor = dist_min - 0.02 = 0.8937; interp_ceiling = 2; loo_p95_max_ceiling = dist_max * 1.2 = 120.9; hf_band = [dist_min - 0.08, 1.02] = [0.7488, 1.02]. LPIPS not gated (no distribution yet).",
  "self_check": "champion on tune clip passes all four checks (obs 0.917, interp 2, hf 0.829; loo check applies only when a candidate enables loo). G6_* gates live in iterate.py run_gates; candidates run with save_masks=1 so provenance + hf retention are always available at ~zero cost.",
  "caveat": "single content source (cockatoo), thresholds are champion-relative guards (catch invention worse than champion's worst measured window), not perceptual absolutes",
  "location": "results/ni_thresholds.json; code/ni_dist.py; code/ni_thresholds.py; iterate.py G6"},

 {"name": "blind_ab_real_footage", "verdict": "MEASURED",
  "setup": "mixkit drone content (real footage), simulated analog link seed 21, 60 frames. Blind VLM judge, 5 randomized trials per A/B (2 static flicker, 2 motion detail, 1 motion flicker), randomized A/B assignment, ties favor the second system (status-quo bias). Judge never knows identity. Sheets: 150px regions at 2x, reference column ignored.",
  "ab1_bar_vs_ours": {"votes": {"bar": 3, "ours": 2, "tie": 0},
   "reasons": "4/5 trials complain the loser is 'overly smoothed / overly blurred, losing texture'; bar praised for preserving fine vertical texture. Ours = smoother, bar = noisier but textured."},
  "ab2_degraded_vs_ours": {"votes": {"degraded": 2, "ours": 3, "tie": 0},
   "reasons": "restoration visibly reduces noise/flicker, but judges still credit degraded frames with finer texture in 2/5 trials."},
  "answer": "The arbiter does NOT crown the champion. On real content the blind judge prefers bar over ours 3-2 (no ties). The suite already said bar wins SSIM-all (-0.032) and static flicker (+2.02); the blind look test confirms it perceptually. Ours' PSNR edge (+1.69 suite, +0.37 here) is bought with smoothing that a blind stranger can see as texture loss. hf_retention 0.829 = the smoothing signature. IMPLICATION for the loop: the binding constraint is now texture preservation on observed pixels (LPIPS / HF retention axis), not PSNR margins. A champion that cannot beat bar blind on real footage should not be called the display champion.",
  "scores_context": {"degraded": {"p.all": 19.14, "s.all": 0.164}, "bar": {"p.all": 28.78, "s.all": 0.748, "flkS": 0.89}, "ours": {"p.all": 29.15, "s.all": 0.739, "flkS": 1.81}},
  "location": "results/real_ab.json; docs/finding-blind-ab-real.md; media/real_ab_sidebyside.mp4"}
]
with open("../results/ledger.json", "a") as f:
    for e in entries:
        f.write(json.dumps(e) + "\n")
print("appended", len(entries), "entries")
